//! Support for piet Android back-end.

// use std::marker::PhantomData;

// use cairo::{Context, Format, ImageSurface};

use jni_glue::Env;
use piet::{ErrorKind, ImageFormat};

#[doc(hidden)]
pub use piet_android::*;

/// The `RenderContext` for the Android backend, which is selected.
pub type Piet<'a, 'draw> = AndroidRenderContext<'a, 'draw>;

/// The associated brush type for this backend.
///
/// This type matches `RenderContext::Brush`
pub type Brush = AndroidBrush;

/// The associated text factory for this backend.
///
/// This type matches `RenderContext::Text`
pub type PietText<'a> = AndroidText;

/// The associated font type for this backend.
///
/// This type matches `RenderContext::Text::Font`
pub type PietFont = AndroidFont;

/// The associated font builder for this backend.
///
/// This type matches `RenderContext::Text::FontBuilder`
pub type PietFontBuilder<'a> = AndroidFontBuilder;

/// The associated text layout type for this backend.
///
/// This type matches `RenderContext::Text::TextLayout`
pub type PietTextLayout = AndroidTextLayout;

/// The associated text layout builder for this backend.
///
/// This type matches `RenderContext::Text::TextLayoutBuilder`
pub type PietTextLayoutBuilder<'a> = AndroidTextLayoutBuilder;

/// The associated image type for this backend.
///
/// This type matches `RenderContext::Image`
/// Should be an Android Bitmap
pub type Image = AndroidBitmap;

/// A struct that can be used to create bitmap render contexts.
pub struct Device<'env> {
    env: &'env Env,
}

/// A struct provides a `RenderContext` and then can have its bitmap extracted.
pub struct BitmapTarget<'a, 'draw> {
    surface: AndroidBitmap,
    cr: CanvasContext<'draw>,
    env: &'a Env,
}

impl<'env> Device<'env> {
    /// Create a new device.
    pub fn new(env: &'env Env) -> Result<Device<'env>, piet::Error> {
        Ok(Device { env })
    }

    /// Create a new bitmap target.
    pub fn bitmap_target(
        &self,
        width: usize,
        height: usize,
        pix_scale: f64,
    ) -> Result<BitmapTarget, piet::Error> {
        let surface = AndroidBitmap::create(
            self.env,
            ImageFormat::RgbaSeparate,
            width as i32,
            height as i32,
        )
        .unwrap();
        let cr = CanvasContext::new(&surface);
        let pix_scale = pix_scale as f32;
        cr.scale(pix_scale, pix_scale);
        Ok(BitmapTarget {
            surface,
            cr,
            env: self.env,
        })
    }
}

impl<'a, 'draw> BitmapTarget<'a, 'draw> {
    /// Get a piet `RenderContext` for the bitmap.
    ///
    /// Note: caller is responsible for calling `finish` on the render
    /// context at the end of rendering.
    pub fn render_context<'b>(&'b mut self) -> AndroidRenderContext<'b, 'draw> {
        AndroidRenderContext::new(&mut self.cr)
    }

    /// Get raw RGBA pixels from the bitmap.
    pub fn into_raw_pixels(self, fmt: ImageFormat) -> Result<Vec<u8>, piet::Error> {
        // TODO: convert other formats.
        if fmt != ImageFormat::RgbaSeparate {
            return Err(piet::new_error(ErrorKind::NotSupported));
        }
        std::mem::drop(self.cr);
        let android_bitmap = self.surface;
        let byte_count = android_bitmap.byte_count(self.env);
        let mut reader = android_bitmap.to_reader(self.env);
        use std::io::Read;
        let mut raw_data = vec![0u8; byte_count];
        reader.read(&mut raw_data[..]);
        Ok(raw_data)
    }
}
