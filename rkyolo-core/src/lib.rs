pub mod drawing;
mod postprocess;
// 从子模块中重新导出关键的公共类型，方便外部使用
pub use drawing::{draw_results, load_labels};
pub use image;
pub use image::{Rgb, RgbImage, imageops, open};
pub use postprocess::{BoundingBox, Detection, post_process_i8};
pub use rknn_ffi;
pub use rknn_ffi::RknnContext;

use std::error::Error;
use std::fmt;
use std::path::Path;

/// 存储 Letterbox 预处理的相关信息
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    scale: f32,
    pad_x: u32,
    pad_y: u32,
}
/// 自定义错误类型，用于封装来自 RKNN FFI 调用的 i32 错误码。
#[derive(Debug)]
pub struct RknnError(pub i32);

impl RknnError {
    /// 从 i32 错误码创建 RknnError
    pub fn new(code: i32) -> Self {
        RknnError(code)
    }
}

// 实现 From trait，允许从 i32 自动转换为 RknnError
impl From<i32> for RknnError {
    fn from(code: i32) -> Self {
        RknnError(code)
    }
}

// 实现 Display trait，以便能够友好地打印错误信息。
impl fmt::Display for RknnError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RKNN API Error with code: {}", self.0)
    }
}
// 实现 Error trait，这样 RknnError 就可以被当作一个标准的错误类型来处理，
// 并且可以被放入 Box<dyn Error> 中。
impl Error for RknnError {}

/// 使用 `image` crate 和 Letterboxing 技术预处理图像。
pub fn preprocess_letterbox_quantize(
    image_path: &Path,
    target_width: u32,
    target_height: u32,
    zp: i32,
    scale: f32,
) -> Result<(Vec<i8>, LetterboxInfo), image::ImageError> {
    // 1. Letterbox 几何变换
    let img = image::open(image_path)?.to_rgb8();
    let (original_w, original_h) = img.dimensions();
    let scale_val =
        (target_width as f32 / original_w as f32).min(target_height as f32 / original_h as f32);
    let new_w = (original_w as f32 * scale_val) as u32;
    let new_h = (original_h as f32 * scale_val) as u32;
    let resized_img = imageops::resize(&img, new_w, new_h, imageops::FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(target_width, target_height, Rgb([114, 114, 114]));
    let pad_x = (target_width - new_w) / 2;
    let pad_y = (target_height - new_h) / 2;
    imageops::overlay(&mut canvas, &resized_img, pad_x.into(), pad_y.into());
    let info = LetterboxInfo {
        scale: scale_val,
        pad_x,
        pad_y,
    };
    let u8_data = canvas.into_raw();

    // 2. 【关键】手动归一化并量化
    let i8_data: Vec<i8> = u8_data
        .into_iter()
        .map(|val_u8| {
            // 正确的公式: q = round( (f_val / scale) + zp )
            // 其中 f_val 是归一化后的浮点值
            let f_val = val_u8 as f32 / 255.0; // <--- 之前遗漏的归一化步骤！
            let q_val = (f_val / scale + zp as f32).round() as i32;
            q_val.clamp(i8::MIN as i32, i8::MAX as i32) as i8
        })
        .collect();

    Ok((i8_data, info))
}
