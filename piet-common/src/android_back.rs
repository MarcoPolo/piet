//! Support for piet Android back-end.

use std::marker::PhantomData;

use cairo::{Context, Format, ImageSurface};

use piet::{ErrorKind, ImageFormat};

#[doc(hidden)]
pub use piet_android::*;

/// The `RenderContext` for the Android backend, which is selected.
pub type Piet<'a> = AndroidRenderContext<'a>;

/// The associated brush type for this backend.
///
/// This type matches `RenderContext::Brush`
pub type Brush = piet_android::Brush;

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
pub type Image = ImageSurface;

/// A struct that can be used to create bitmap render contexts.
pub struct Device;

/// A struct provides a `RenderContext` and then can have its bitmap extracted.
pub struct BitmapTarget<'a> {
    surface: ImageSurface,
    cr: Context,
    phantom: PhantomData<&'a ()>,
}

impl Device {
    /// Create a new device.
    pub fn new() -> Result<Device, piet::Error> {
        Ok(Device)
    }

    /// Create a new bitmap target.
    pub fn bitmap_target(
        &self,
        width: usize,
        height: usize,
        pix_scale: f64,
    ) -> Result<BitmapTarget, piet::Error> {
        let surface = ImageSurface::create(Format::ARgb32, width as i32, height as i32).unwrap();
        let cr = Context::new(&surface);
        cr.scale(pix_scale, pix_scale);
        let phantom = Default::default();
        Ok(BitmapTarget {
            surface,
            cr,
            phantom,
        })
    }
}

impl<'a> BitmapTarget<'a> {
    /// Get a piet `RenderContext` for the bitmap.
    ///
    /// Note: caller is responsible for calling `finish` on the render
    /// context at the end of rendering.
    pub fn render_context<'b>(&'b mut self) -> AndroidRenderContext<'b> {
        AndroidRenderContext::new(&mut self.cr)
    }

    /// Get raw RGBA pixels from the bitmap.
    pub fn into_raw_pixels(mut self, fmt: ImageFormat) -> Result<Vec<u8>, piet::Error> {
        // TODO: convert other formats.
        if fmt != ImageFormat::RgbaPremul {
            return Err(piet::new_error(ErrorKind::NotSupported));
        }
        std::mem::drop(self.cr);
        self.surface.flush();
        let stride = self.surface.get_stride() as usize;
        let width = self.surface.get_width() as usize;
        let height = self.surface.get_height() as usize;
        let mut raw_data = vec![0; width * height * 4];
        let buf = self
            .surface
            .get_data()
            .map_err(|e| Into::<Box<dyn std::error::Error>>::into(e))?;
        for y in 0..height {
            let src_off = y * stride;
            let dst_off = y * width * 4;
            for x in 0..width {
                raw_data[dst_off + x * 4 + 0] = buf[src_off + x * 4 + 2];
                raw_data[dst_off + x * 4 + 1] = buf[src_off + x * 4 + 1];
                raw_data[dst_off + x * 4 + 2] = buf[src_off + x * 4 + 0];
                raw_data[dst_off + x * 4 + 3] = buf[src_off + x * 4 + 3];
            }
        }
        Ok(raw_data)
    }
}
