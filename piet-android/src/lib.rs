#![cfg(target_os = "android")]
mod grapheme;

use unwrap::*;

use jni;

use jni_android_sys::android::graphics::{
    self, Bitmap, Bitmap_Config, Canvas, Color as AColor, LinearGradient, Paint, Paint_Style,
    Path as APath, Shader, Shader_TileMode, Typeface,
};
use jni_android_sys::java::lang::String as JavaString;
use jni_android_sys::java::nio::{Buffer, ByteBuffer};
use jni_glue::{self, Argument, Env, Global, Local, PrimitiveArray, Ref as JNIRef, VM};
use jni_sys::JNIEnv;
use std::borrow::Cow;
use std::error::Error as StdError;
use std::fmt;
use std::io::Read;
use std::marker::PhantomData;
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

use unicode_segmentation::UnicodeSegmentation;

use crate::grapheme::point_x_in_grapheme;

use piet::kurbo::{Affine, PathEl, Point, QuadBez, Rect, Shape};

use piet::{
    self, new_error, Color, Error, ErrorKind, FixedGradient, Font, FontBuilder, HitTestMetrics,
    HitTestPoint, HitTestTextPosition, ImageFormat, InterpolationMode, IntoBrush, LineCap,
    LineJoin, RenderContext, RoundInto, StrokeStyle, Text, TextLayout, TextLayoutBuilder,
};

thread_local! {
    static ENV: RefCell<Option<*mut JNIEnv>> = RefCell::new(None);
}

pub fn set_current_env(env: &Env) {
    let jnienv = env.as_jni_env();
    ENV.with(|tls_env| {
        let mut tls_env = tls_env.borrow_mut();
        *tls_env = Some(jnienv);
    });
}

fn with_current_env<F, R>(f: F) -> R
where
    F: FnOnce(&Env) -> R,
{
    ENV.with(|jnienv| {
        let env: &Env = unsafe {
            Env::from_ptr(
                jnienv
                    .borrow()
                    .expect("Env isn't set in our global environment"),
            )
        };
        f(env)
    })
}

pub struct AndroidDevice {}

pub struct AndroidBitmap(pub Global<Bitmap>);

impl AndroidBitmap {
    fn with_bitmap<'a: 'env, 'env: 'subenv, 'subenv, F, R>(&'a self, env: &'env Env, f: F) -> R
    where
        F: FnOnce(&Bitmap) -> R,
    {
        let bitmap_ref: JNIRef<'env, Bitmap> = self.0.with(env);
        let bitmap: &Bitmap = &bitmap_ref;
        f(bitmap)
    }
}

fn format_to_bitmap_config<'env>(
    env: &'env Env,
    format: ImageFormat,
) -> Local<'env, Bitmap_Config> {
    match format {
        ImageFormat::RgbaSeparate | ImageFormat::RgbaPremul => {
            Bitmap_Config::ARGB_8888(env).expect("Create bitmap config failed")
        }
        _ => panic!("Unhandled image format"),
    }
}

#[derive(Debug)]
pub struct AndroidError(String);
impl StdError for AndroidError {}
impl fmt::Display for AndroidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct AndroidBitmapReader<'a>(JNIRef<'a, Bitmap>, usize);

impl AndroidBitmap {
    pub fn create(
        env: &Env,
        format: ImageFormat,
        width: i32,
        height: i32,
    ) -> Result<AndroidBitmap, AndroidError> {
        let config = format_to_bitmap_config(env, format);
        Bitmap::createBitmap_int_int_Config(env, width, height, &config as &Bitmap_Config)
            .map(|bm| AndroidBitmap(unwrap!(bm).into()))
            .map_err(|e| AndroidError(format!("Failed to create Bitmap: {:?}", e)))
    }

    pub fn create_with_buf(
        env: &Env,
        format: ImageFormat,
        width: i32,
        height: i32,
        buf: &[u8],
    ) -> Result<AndroidBitmap, AndroidError> {
        // Convert &[u8] to &[i8] for jni
        let buf = unsafe { &*(buf as *const [u8] as *const [i8]) };
        let java_bytes: Local<jni_glue::ByteArray> = jni_glue::PrimitiveArray::from(env, &buf);
        let byte_buffer = unwrap!(unwrap!(ByteBuffer::wrap_byte_array(
            env,
            Some(&java_bytes as &jni_glue::ByteArray)
        )));

        let config = format_to_bitmap_config(env, format);
        Bitmap::createBitmap_int_int_Config(env, width, height, &config as &Bitmap_Config)
            .map(|bm| {
                let bm = unwrap!(bm);
                bm.copyPixelsFromBuffer(Some(&byte_buffer as &Buffer));
                AndroidBitmap(bm.into())
            })
            .map_err(|e| AndroidError(format!("Failed to create Bitmap: {:?}", e)))
    }

    pub fn byte_count(&self, env: &Env) -> usize {
        let bitmap = self.0.with(env);
        let width = bitmap.getWidth().unwrap() as usize;
        let height = bitmap.getHeight().unwrap() as usize;
        width * height * 4
    }

    /// Returns the pixel data as is from the bitmap buffer
    /// Non premul
    /// This could be better with direct write to a buffer
    pub fn get_bytes<'a>(&'a self, env: &'a Env) -> Vec<u8> {
        let bitmap = self.0.with(env);

        let byte_count = bitmap.getByteCount().unwrap();
        let java_byte_array: Local<jni_glue::ByteArray> =
            jni_glue::PrimitiveArray::new(env, byte_count as usize);
        let bb = ByteBuffer::wrap_byte_array(env, Some(&java_byte_array as &jni_glue::ByteArray))
            .unwrap()
            .unwrap();
        bitmap
            .copyPixelsToBuffer(Some(&bb as &Buffer))
            .expect("Copy pixels to buffer failed");

        let mut v: Vec<u8> = vec![0u8; byte_count as usize];
        {
            let u8_buf = &mut v[..];
            // Java bytes are i8, so we pretend we are looking at an i8 slice.
            // Basically &[u8] into &[i8]
            let i8_buf = unsafe { &mut *(u8_buf as *mut [u8] as *mut [i8]) };
            java_byte_array.get_region(0, i8_buf);
        }

        v
    }
}

impl Read for AndroidBitmapReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        let bitmap = &self.0;
        let read_so_far = &mut self.1;
        let width = bitmap.getWidth().unwrap();
        let height = bitmap.getHeight().unwrap();
        let pixels_read = (*read_so_far / 4 as usize) as i32;
        let x_so_far = pixels_read % width;
        let y_so_far = pixels_read / width;
        let mut bytes_written = 0;
        let buf_len = buf.len();

        for y in y_so_far..height {
            for x in 0..width {
                if y == y_so_far && x < x_so_far {
                    continue;
                }
                let pixel = bitmap.getPixel(x, y).unwrap();
                // In order of ABGR
                for color in 0..4 {
                    if bytes_written == buf_len {
                        return Ok(bytes_written);
                    }
                    buf[bytes_written + color] = (pixel >> (24 - color * 8)) as u8 & 0xff;
                    bytes_written += 1;
                    *read_so_far += 1;
                }
            }
        }
        return Ok(bytes_written);
    }
}

#[derive(Clone)]
pub struct AndroidFont {
    // typeface: Global<Typeface>,
    paint: Rc<Global<Paint>>,
}

fn into_java_string<'a>(env: &'a Env, text: &str) -> Local<'a, JavaString> {
    let raw_env = env.as_jni_env();
    let jni_env: jni::JNIEnv = unsafe { jni::JNIEnv::from_raw(raw_env).unwrap() };
    let text: jni_sys::jobject = jni_env.new_string(text).unwrap().into_inner();
    let text: Local<JavaString> = unsafe { Local::from_env_object(raw_env, text) };
    text
}

impl AndroidFont {
    fn measure_text(&self, text: &str) -> f32 {
        // jni_glue doesn't let us create new strings
        // This should be fine since it comes from our thread and we move it to JNIEnv right away
        with_current_env(|env| {
            let paint = self.paint.with(env);
            let text = into_java_string(env, &text);
            paint
                .measureText_String(Some(&text as &JavaString))
                .unwrap()
        })
    }

    fn measure_text_bounded(&self, text: &str, start: i32, end: i32) -> f32 {
        with_current_env(|env| {
            let paint = self.paint.with(env);
            let text = into_java_string(env, &text);
            paint
                .measureText_String_int_int(Some(&text as &JavaString), start, end)
                .unwrap()
        })
    }
}

pub enum FontWeight {
    Thin = 100,
    ExtraLight = 200,
    Light = 300,
    Normal = 400,
    Medium = 500,
    SemiBold = 600,
    Bold = 700,
    ExtraBold = 800,
    Black = 900,
}

pub struct AndroidFontBuilder {
    family: String,
    weight: FontWeight,
    is_italic: bool,
    size: f64,
}

pub struct AndroidText;

pub struct AndroidTextLayout {
    font: AndroidFont,
    text: String,
}

pub struct AndroidTextLayoutBuilder(AndroidTextLayout);

#[derive(Clone)]
pub struct AndroidBrush(Rc<Global<Paint>>);

impl AndroidBrush {
    fn with_paint<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Paint) -> R,
    {
        with_current_env(|env| {
            let paint: &Paint = &self.0.with(env);
            f(paint)
        })
    }
}

impl Default for AndroidBrush {
    fn default() -> Self {
        let paint = with_current_env(|env| {
            let paint: Global<Paint> = Paint::new_int(env, Paint::ANTI_ALIAS_FLAG).unwrap().into();
            paint
        });
        AndroidBrush(Rc::new(paint))
    }
}

impl From<FixedGradient> for AndroidBrush {
    fn from(gradient: FixedGradient) -> AndroidBrush {
        let brush = AndroidBrush::default();
        match gradient {
            FixedGradient::Linear(piet::FixedLinearGradient { start, end, stops }) => {
                with_current_env(|env| {
                    let start_x = start.x as f32;
                    let start_y = start.y as f32;
                    let end_x = end.x as f32;
                    let end_y = end.y as f32;

                    let colors: Vec<i32> = stops
                        .iter()
                        .map(|stop| {
                            let color: AndroidColor = stop.color.clone().into();
                            let color: i32 = color.into();
                            color
                        })
                        .collect();
                    let pos: Vec<f32> = stops.iter().map(|stop| stop.pos).collect();
                    let colors: Local<jni_glue::IntArray> =
                        jni_glue::PrimitiveArray::from(env, &colors);
                    let pos: Local<jni_glue::FloatArray> =
                        jni_glue::PrimitiveArray::from(env, &pos);
                    let tile_mode = Shader_TileMode::CLAMP(env).unwrap();

                    let android_gradient =
                        LinearGradient::new_float_float_float_float_int_array_float_array_TileMode(
                            env,
                            start_x,
                            start_y,
                            end_x,
                            end_y,
                            Some(&colors as &jni_glue::IntArray),
                            Some(&pos as &jni_glue::FloatArray),
                            Some(&tile_mode as &Shader_TileMode),
                        )
                        .unwrap();
                    brush.with_paint(|paint| {
                        paint.setShader(Some(&android_gradient as &Shader)).unwrap();
                    })
                })
            }
            FixedGradient::Radial(piet::FixedRadialGradient {
                center,
                origin_offset,
                radius,
                stops,
            }) => unimplemented!("TODO add Radial offset"),
        }
        brush
    }
}

pub struct CanvasContext<'draw>(Global<Canvas>, PhantomData<&'draw ()>);

impl<'draw> CanvasContext<'draw> {
    pub fn new(surface: &AndroidBitmap) -> CanvasContext<'draw> {
        with_current_env(|env: &Env| {
            let bitmap_ref: &Bitmap = &surface.0.with(env);
            let canvas: Local<'_, Canvas> =
                Canvas::new_Bitmap(env, Some(bitmap_ref)).expect("Failed to create Canvas");
            CanvasContext(canvas.into(), Default::default())
        })
    }

    pub fn scale(&self, sx: f32, sy: f32) {
        self.with_canvas(|canvas| {
            canvas.scale_float_float(sx, sy).expect("Scale failed");
        })
    }

    pub fn with_canvas<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Canvas) -> R,
    {
        with_current_env(|env| {
            let canvas: &Canvas = &self.0.with(env);
            f(canvas)
        })
    }

    pub fn with_env_canvas<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Env, &Canvas) -> R,
    {
        with_current_env(|env| {
            let canvas: &Canvas = &self.0.with(env);
            f(env, canvas)
        })
    }
}

pub struct AndroidRenderContext<'a, 'draw> {
    canvas: &'a mut CanvasContext<'draw>,
    text: AndroidText,
}

impl AndroidRenderContext<'_, '_> {
    pub fn new<'a, 'draw>(canvas: &'a mut CanvasContext<'draw>) -> AndroidRenderContext<'a, 'draw> {
        AndroidRenderContext {
            canvas,
            text: AndroidText,
        }
    }
}

impl Font for AndroidFont {}

impl TextLayoutBuilder for AndroidTextLayoutBuilder {
    type Out = AndroidTextLayout;
    fn build(self) -> Result<Self::Out, Error> {
        Ok(self.0)
    }
}

impl FontBuilder for AndroidFontBuilder {
    type Out = AndroidFont;
    fn build(self) -> Result<Self::Out, Error> {
        with_current_env(|env| {
            let font_family = into_java_string(env, &self.family);
            let weight = self.weight as i32;
            let typeface =
                Typeface::create_String_int(&env, Some(&font_family as &JavaString), weight)
                    .unwrap()
                    .unwrap();
            let typeface = Typeface::create_Typeface_int_boolean(
                &env,
                &typeface as &Typeface,
                weight,
                self.is_italic,
            )
            .unwrap()
            .unwrap();
            let paint = Paint::new_int(&env, Paint::ANTI_ALIAS_FLAG).unwrap();
            paint.setTypeface(&typeface as &Typeface).unwrap();
            paint.setTextSize(self.size as f32).unwrap();
            Ok(AndroidFont {
                paint: Rc::new(paint.into()),
            })
        })
    }
}

impl Text for AndroidText {
    type FontBuilder = AndroidFontBuilder;
    type Font = AndroidFont;
    type TextLayoutBuilder = AndroidTextLayoutBuilder;
    type TextLayout = AndroidTextLayout;

    fn new_font_by_name(&mut self, name: &str, size: f64) -> Self::FontBuilder {
        AndroidFontBuilder {
            family: String::from(name),
            weight: FontWeight::Normal,
            is_italic: false,
            size,
        }
    }

    fn new_text_layout(&mut self, font: &Self::Font, text: &str) -> Self::TextLayoutBuilder {
        AndroidTextLayoutBuilder(AndroidTextLayout {
            text: String::from(text),
            font: font.clone(),
        })
    }
}

impl TextLayout for AndroidTextLayout {
    fn width(&self) -> f64 {
        self.font.measure_text(&self.text) as f64
    }

    // first assume one line.
    // TODO do with lines
    fn hit_test_point(&self, point: Point) -> HitTestPoint {
        // internal logic is using grapheme clusters, but return the text position associated
        // with the border of the grapheme cluster.

        // null case
        if self.text.len() == 0 {
            return HitTestPoint::default();
        }

        // get bounds
        // TODO handle if string is not null yet count is 0?
        let end = UnicodeSegmentation::graphemes(self.text.as_str(), true).count() - 1;
        let end_bounds = match self.get_grapheme_boundaries(end) {
            Some(bounds) => bounds,
            None => return HitTestPoint::default(),
        };

        let start = 0;
        let start_bounds = match self.get_grapheme_boundaries(start) {
            Some(bounds) => bounds,
            None => return HitTestPoint::default(),
        };

        // first test beyond ends
        if point.x > end_bounds.trailing {
            let mut res = HitTestPoint::default();
            res.metrics.text_position = self.text.len();
            return res;
        }
        if point.x <= start_bounds.leading {
            return HitTestPoint::default();
        }

        // then test the beginning and end (common cases)
        if let Some(hit) = point_x_in_grapheme(point.x, &start_bounds) {
            return hit;
        }
        if let Some(hit) = point_x_in_grapheme(point.x, &end_bounds) {
            return hit;
        }

        // Now that we know it's not beginning or end, begin binary search.
        // Iterative style
        let mut left = start;
        let mut right = end;
        loop {
            // pick halfway point
            let middle = left + ((right - left) / 2);

            let grapheme_bounds = match self.get_grapheme_boundaries(middle) {
                Some(bounds) => bounds,
                None => return HitTestPoint::default(),
            };

            if let Some(hit) = point_x_in_grapheme(point.x, &grapheme_bounds) {
                return hit;
            }

            // since it's not a hit, check if closer to start or finish
            // and move the appropriate search boundary
            if point.x < grapheme_bounds.leading {
                right = middle;
            } else if point.x > grapheme_bounds.trailing {
                left = middle + 1;
            } else {
                unreachable!("hit_test_point conditional is exhaustive");
            }
        }
    }

    fn hit_test_text_position(&self, text_position: usize) -> Option<HitTestTextPosition> {
        // Using substrings, but now with unicode grapheme awareness

        let text_len = self.text.len();

        if text_position == 0 {
            return Some(HitTestTextPosition::default());
        }

        if text_position as usize >= text_len {
            return Some(HitTestTextPosition {
                point: Point {
                    x: self.font.measure_text(&self.text) as f64,
                    y: 0.0,
                },
                metrics: HitTestMetrics {
                    text_position: text_len,
                },
            });
        }

        // Already checked that text_position > 0 and text_position < count.
        // If text position is not at a grapheme boundary, use the text position of current
        // grapheme cluster. But return the original text position
        // Use the indices (byte offset, which for our purposes = utf8 code units).
        let grapheme_indices = UnicodeSegmentation::grapheme_indices(self.text.as_str(), true)
            .take_while(|(byte_idx, _s)| text_position >= *byte_idx);

        if let Some((byte_idx, _s)) = grapheme_indices.last() {
            let point_x = self.font.measure_text(&self.text[0..byte_idx]);

            Some(HitTestTextPosition {
                point: Point {
                    x: point_x as f64,
                    y: 0.0,
                },
                metrics: HitTestMetrics { text_position },
            })
        } else {
            // iterated to end boundary
            Some(HitTestTextPosition {
                point: Point {
                    x: self.font.measure_text(&self.text) as f64,
                    y: 0.0,
                },
                metrics: HitTestMetrics {
                    text_position: text_len,
                },
            })
        }
    }
}

impl<'draw> IntoBrush<AndroidRenderContext<'_, 'draw>> for AndroidBrush {
    fn make_brush<'b>(
        &'b self,
        _piet: &mut AndroidRenderContext,
        _bbox: impl FnOnce() -> Rect,
    ) -> std::borrow::Cow<'b, AndroidBrush> {
        Cow::Borrowed(self)
    }
}

struct AndroidPath<'env>(Local<'env, APath>);

impl<'env> AndroidPath<'env> {
    fn from_shape(env: &'env Env, shape: impl Shape) -> AndroidPath<'env> {
        let path = APath::new(env).unwrap();
        for el in shape.to_bez_path(1e-3) {
            let res = match el {
                PathEl::MoveTo(p) => path.moveTo(p.x as f32, p.y as f32),
                PathEl::LineTo(p) => path.lineTo(p.x as f32, p.y as f32),
                PathEl::QuadTo(p1, p2) => {
                    path.quadTo(p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32)
                }
                PathEl::CurveTo(p1, p2, p3) => path.cubicTo(
                    p1.x as f32,
                    p1.y as f32,
                    p2.x as f32,
                    p2.y as f32,
                    p3.x as f32,
                    p3.y as f32,
                ),
                PathEl::ClosePath => path.close(),
            };
            res.expect("Adding to path failed");
        }

        AndroidPath(path)
    }
}
// impl<'env> AndroidPath<'env> {
//     fn ()
// }

/// Android uses argb, so let's have a helper function to convert from rgba
#[derive(Debug)]
struct AndroidColor(u32);

impl Into<i32> for AndroidColor {
    fn into(self) -> i32 {
        self.0 as i32
    }
}

impl From<Color> for AndroidColor {
    fn from(color: Color) -> Self {
        let color = color.as_rgba_u32();
        // Just the bottom byte
        let alpha = color as u8;
        let color = (color >> 8) | ((alpha as u32) << 24);
        AndroidColor(color)
    }
}

impl<'draw> RenderContext for AndroidRenderContext<'_, 'draw> {
    type Brush = AndroidBrush;
    type Text = AndroidText;
    type TextLayout = AndroidTextLayout;
    type Image = AndroidBitmap;

    fn status(&mut self) -> Result<(), Error> {
        // TODO
        Ok(())
    }

    fn clear(&mut self, color: Color) {
        let color: AndroidColor = color.into();
        self.canvas.with_canvas(|canvas| {
            canvas.drawColor_int(color.into()).unwrap();
        })
    }

    fn solid_brush(&mut self, color: Color) -> Self::Brush {
        let brush = AndroidBrush::default();
        let color: AndroidColor = color.into();
        brush.with_paint(|paint| paint.setColor(color.into()).unwrap());
        brush
    }

    fn gradient(&mut self, gradient: impl Into<FixedGradient>) -> Result<Self::Brush, Error> {
        let brush = gradient.into().into();
        Ok(brush)
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.canvas.with_env_canvas(|env, canvas| {
            let android_path = AndroidPath::from_shape(env, shape);
            brush.with_paint(|paint| {
                paint
                    .setStyle(Some(&*Paint_Style::FILL(env).unwrap()))
                    .unwrap();
                canvas
                    .drawPath(Some(&android_path.0 as &APath), Some(paint))
                    .unwrap();
            })
        })
    }

    fn fill_even_odd(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.canvas.with_env_canvas(|env, canvas| {
            let android_path = AndroidPath::from_shape(env, shape);
            brush.with_paint(|paint| {
                paint
                    .setStyle(Some(&*Paint_Style::FILL(env).unwrap()))
                    .unwrap();
                android_path
                    .0
                    .setFillType(Some(&graphics::Path_FillType::EVEN_ODD(env).unwrap()
                        as &graphics::Path_FillType))
                    .unwrap();
                canvas
                    .drawPath(Some(&android_path.0 as &APath), Some(paint))
                    .unwrap();
            })
        })
    }

    fn clip(&mut self, shape: impl Shape) {
        self.canvas.with_env_canvas(|env, canvas| {
            let android_path = AndroidPath::from_shape(env, shape);
            canvas
                .clipPath_Path(Some(&android_path.0 as &APath))
                .unwrap();
        })
    }

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.canvas.with_env_canvas(|env, canvas| {
            let android_path = AndroidPath::from_shape(env, shape);
            brush.with_paint(|paint| {
                paint
                    .setStyle(Some(&*Paint_Style::STROKE(env).unwrap()))
                    .unwrap();
                paint.setStrokeWidth(width as f32).unwrap();
                canvas
                    .drawPath(Some(&android_path.0 as &APath), Some(paint))
                    .unwrap();
            })
        })
    }

    fn stroke_styled(
        &mut self,
        shape: impl Shape,
        brush: &impl IntoBrush<Self>,
        width: f64,
        style: &StrokeStyle,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.canvas.with_env_canvas(|env, canvas| {
            let android_path = AndroidPath::from_shape(env, shape);
            brush.with_paint(|paint| {
                paint.setStrokeWidth(width as f32).unwrap();
                paint
                    .setStyle(Some(&*Paint_Style::STROKE(env).unwrap()))
                    .unwrap();

                if let Some(line_join) = style.line_join {
                    let paint_join: Local<graphics::Paint_Join> = match line_join {
                        LineJoin::Miter => graphics::Paint_Join::MITER(env),
                        LineJoin::Round => graphics::Paint_Join::ROUND(env),
                        LineJoin::Bevel => graphics::Paint_Join::BEVEL(env),
                    }
                    .unwrap();
                    paint
                        .setStrokeJoin(Some(&paint_join as &graphics::Paint_Join))
                        .unwrap();
                }

                if let Some(line_cap) = style.line_cap {
                    let paint_cap: Local<graphics::Paint_Cap> = match line_cap {
                        LineCap::Butt => graphics::Paint_Cap::BUTT(env),
                        LineCap::Round => graphics::Paint_Cap::ROUND(env),
                        LineCap::Square => graphics::Paint_Cap::SQUARE(env),
                    }
                    .unwrap();
                    paint
                        .setStrokeCap(Some(&paint_cap as &graphics::Paint_Cap))
                        .unwrap();
                }

                if let Some(dash) = &style.dash {
                    let (intervals, phase) = dash;
                    let intervals_f32: Vec<f32> = intervals.iter().map(|f| *f as f32).collect();
                    let intervals: Local<jni_glue::FloatArray> =
                        PrimitiveArray::from(env, &intervals_f32);
                    let dash_effect = graphics::DashPathEffect::new(
                        env,
                        Some(&intervals as &jni_glue::FloatArray),
                        *phase as f32,
                    )
                    .unwrap();
                    paint
                        .setPathEffect(Some(&dash_effect as &graphics::PathEffect))
                        .unwrap();
                }

                if let Some(miter_limit) = &style.miter_limit {
                    paint.setStrokeMiter(*miter_limit as f32).unwrap();
                }

                android_path
                    .0
                    .setFillType(Some(&graphics::Path_FillType::EVEN_ODD(env).unwrap()
                        as &graphics::Path_FillType))
                    .unwrap();
                canvas
                    .drawPath(Some(&android_path.0 as &APath), Some(paint))
                    .unwrap();
            })
        })
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.text
    }

    fn draw_text(
        &mut self,
        layout: &Self::TextLayout,
        pos: impl Into<Point>,
        brush: &impl IntoBrush<Self>,
    ) {
        // TODO: bounding box for text
        let brush = brush.make_brush(self, || Rect::ZERO);
        let Point { x, y } = pos.into();

        self.canvas.with_env_canvas(|env, canvas| {
            brush.with_paint(|paint| {
                let font_paint = layout.font.paint.with(env);
                paint.set(Some(&font_paint as &Paint)).unwrap();
                let java_str = into_java_string(env, &layout.text);

                canvas
                    .drawText_String_float_float_Paint(
                        Some(&java_str as &JavaString),
                        x as f32,
                        y as f32,
                        Some(&paint as &Paint),
                    )
                    .unwrap();
            });
        })
    }

    fn save(&mut self) -> Result<(), Error> {
        // TODO propagate errors
        self.canvas.with_canvas(|canvas| {
            canvas.save().unwrap();
        });
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Error> {
        // TODO propagate errors
        self.canvas.with_canvas(|canvas| {
            canvas.restore().unwrap();
        });
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        // Not sure what should happen here
        Ok(())
    }

    fn transform(&mut self, transform: Affine) {
        self.canvas.with_env_canvas(|env, canvas| {
            let floats = transform.as_coeffs();
            let mut floats: Vec<f32> = floats.iter().map(|f| *f as f32).collect();
            // Android expects the full 9x9 matrix
            floats.push(0f32);
            floats.push(0f32);
            floats.push(1f32);
            let java_floats: Local<jni_glue::FloatArray> =
                jni_glue::PrimitiveArray::from(env, &floats);

            let matrix = graphics::Matrix::new(env).unwrap();
            matrix
                .setValues(Some(&java_floats as &jni_glue::FloatArray))
                .unwrap();
            canvas
                .setMatrix(Some(&matrix as &graphics::Matrix))
                .unwrap();
        });
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        self.canvas.with_env_canvas(|env, _| {
            let android_bitmap =
                AndroidBitmap::create_with_buf(env, format, width as i32, height as i32, buf)
                    .unwrap();
            Ok(android_bitmap)
        })
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        self.canvas.with_env_canvas(|env, canvas| {
            let rect = rect.into();
            let android_rect = graphics::Rect::new_int_int_int_int(
                env,
                rect.x0 as i32,
                rect.y0 as i32,
                rect.x1 as i32,
                rect.y1 as i32,
            )
            .unwrap();
            let bitmap = image.0.with(env);
            let paint = Paint::new_int(env, Paint::ANTI_ALIAS_FLAG).unwrap();
            if interp == InterpolationMode::Bilinear {
                paint.setFlags(Paint::FILTER_BITMAP_FLAG).unwrap();
            }
            canvas
                .drawBitmap_Bitmap_Rect_Rect_Paint(
                    Some(&bitmap as &Bitmap),
                    None,
                    Some(&android_rect as &graphics::Rect),
                    Some(&paint as &Paint),
                )
                .unwrap();
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
