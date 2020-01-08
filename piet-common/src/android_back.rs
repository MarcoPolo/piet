//! Support for piet Android back-end.

use jni_android_sys::android::graphics::Bitmap;
use jni_glue::{Env, Global};
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
pub struct BitmapTarget<'env, 'draw> {
    surface: AndroidBitmap,
    cr: CanvasContext<'draw>,
    env: &'env Env,
}

impl<'env> Device<'env> {
    /// Create a new device.
    pub fn new(env: &'env Env) -> Result<Device<'env>, piet::Error> {
        piet_android::set_current_env(env);
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

impl<'draw> BitmapTarget<'_, 'draw> {
    /// Get a piet `RenderContext` for the bitmap.
    ///
    /// Note: caller is responsible for calling `finish` on the render
    /// context at the end of rendering.
    pub fn render_context(&mut self) -> AndroidRenderContext<'_, 'draw> {
        AndroidRenderContext::new(&mut self.cr)
    }

    pub fn into_bitmap(self) -> Global<Bitmap> {
        self.surface.0
    }

    /// Get raw RGBA pixels from the bitmap.
    pub fn to_raw_pixels(&self, fmt: ImageFormat) -> Result<Vec<u8>, piet::Error> {
        // TODO: convert other formats.
        if fmt != ImageFormat::RgbaSeparate {
            return Err(piet::new_error(ErrorKind::NotSupported));
        }
        let android_bitmap = &self.surface;
        Ok(android_bitmap.get_bytes(self.env))
    }
}
