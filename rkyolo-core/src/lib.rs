pub mod drawing;
mod postprocess;
// 从子模块中重新导出关键的公共类型，方便外部使用
pub use drawing::{draw_results, load_labels};
pub use image;
pub use image::{Rgb, RgbImage, imageops, open};
pub use log::{debug, trace};
pub use postprocess::{BoundingBox, Detection, post_process_i8};
pub use rknn_ffi;
pub use rknn_ffi::{RknnContext, get_type_string};
use std::error::Error;
use std::fmt;
use std::path::Path;

/// 【新增】定义支持的语言选项，作为公共API
#[derive(Clone, Debug, Default, clap::ValueEnum)]
pub enum Lang {
    #[default]
    En, // 英文
    Zh, // 中文
}

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
    debug!(
        "Preprocessing image: original_dims=({}x{}), target_dims=({}x{})",
        original_w, original_h, target_width, target_height
    );

    let scale_val =
        (target_width as f32 / original_w as f32).min(target_height as f32 / original_h as f32);
    let new_w = (original_w as f32 * scale_val) as u32;
    let new_h = (original_h as f32 * scale_val) as u32;
    trace!(
        "Calculated scale: {}, new_dims: ({}x{})",
        scale_val, new_w, new_h
    );

    let resized_img = imageops::resize(&img, new_w, new_h, imageops::FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(target_width, target_height, Rgb([114, 114, 114]));
    let pad_x = (target_width - new_w) / 2;
    let pad_y = (target_height - new_h) / 2;
    trace!("Padding: x={}, y={}", pad_x, pad_y);

    imageops::overlay(&mut canvas, &resized_img, pad_x.into(), pad_y.into());
    let info = LetterboxInfo {
        scale: scale_val,
        pad_x,
        pad_y,
    };
    let u8_data = canvas.into_raw();
    debug!("Image letterboxed and converted to raw u8 buffer.");

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
    debug!("Quantization complete. ZP={}, Scale={}", zp, scale);

    Ok((i8_data, info))
}

/// 【新增】使用 Letterboxing 技术预处理图像，并将量化后的结果直接写入一个 DMA 缓冲区（零拷贝）。
/// 这个版本会处理硬件要求的 w_stride。
pub fn preprocess_letterbox_quantize_zero_copy(
    image_path: &Path,
    target_width: u32,
    target_height: u32,
    w_stride: u32, // <--- 硬件期望的行步长
    zp: i32,
    scale: f32,
    buffer: &mut [u8], // <--- 直接写入的目标缓冲区
) -> Result<LetterboxInfo, image::ImageError> {
    // 1. Letterbox 几何变换 (与之前版本相同)
    let img = image::open(image_path)?.to_rgb8();
    let (original_w, original_h) = img.dimensions();
    debug!(
        "Preprocessing (zero-copy) for image: original_dims=({}x{}), target_dims=({}x{})",
        original_w, original_h, target_width, target_height
    );

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
    debug!("Image letterboxed, ready for quantization and copy.");

    // 2. 量化并【逐行】拷贝到目标缓冲区
    let line_bytes = target_width as usize * 3; // NHWC, so channels=3
    let stride_bytes = w_stride as usize * 3;

    for i in 0..target_height as usize {
        let u8_line = &u8_data[i * line_bytes..(i + 1) * line_bytes];
        let buffer_line = &mut buffer[i * stride_bytes..i * stride_bytes + line_bytes];

        for (src_pixel, dst_pixel) in u8_line.iter().zip(buffer_line.iter_mut()) {
            let f_val = *src_pixel as f32 / 255.0;
            let q_val = (f_val / scale + zp as f32).round() as i32;
            // 因为目标缓冲区是 u8, 但我们量化到 i8, 所以需要转换
            *dst_pixel = (q_val.clamp(i8::MIN as i32, i8::MAX as i32) as i8).to_le_bytes()[0];
        }
    }

    debug!(
        "Quantization and stride-aware copy complete. Line bytes={}, Stride bytes={}",
        line_bytes, stride_bytes
    );

    Ok(info)
}
