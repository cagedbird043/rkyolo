use clap::Parser;
use image::{Rgb, RgbImage, imageops};
use rknn_ffi::RknnContext;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use walkdir::WalkDir;
mod drawing;
mod postprocess;

/// 一个使用 Rust 和 Rockchip NPU 进行 YOLO 模型推理的应用
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// .rknn 模型文件的路径
    #[arg(short, long, default_value = "assets/yolo11.rknn")]
    model: String,

    /// 要进行检测的输入图片路径，也可以是一个包含图片的目录
    #[arg(short, long, default_value = "assets/bus.jpg")]
    input: String,

    /// 包含类别名称的标签文件路径
    #[arg(short, long, default_value = "assets/coco_labels.txt")]
    labels: String,

    /// 输出结果图片的保存路径，如果输入是文件，则为输出文件名；如果输入是目录，则为输出目录
    #[arg(short, long, default_value = "output.jpg")]
    output: String,

    /// 置信度阈值
    #[arg(long, default_value_t = 0.25)]
    conf_thresh: f32,

    /// NMS (非极大值抑制) 的 IoU 阈值
    #[arg(long, default_value_t = 0.45)]
    iou_thresh: f32,
}

/// 存储 Letterbox 预处理的相关信息
#[derive(Debug, Clone, Copy)]
struct LetterboxInfo {
    scale: f32,
    pad_x: u32,
    pad_y: u32,
}
/// 自定义错误类型，用于封装来自 RKNN FFI 调用的 i32 错误码。
#[derive(Debug)]
struct RknnError(i32);

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
fn preprocess_letterbox_quantize(
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

/// 【新增】处理单张图片的完整流程
fn process_single_image(
    ctx: &mut RknnContext,
    image_path: &Path,
    labels: &[String],
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    println!("\n--- Processing Image: {:?} ---", image_path);

    let mut original_image = image::open(image_path)?.to_rgb8();

    let input_attrs = ctx.query_input_attrs().map_err(RknnError)?;
    let input_attr = &input_attrs[0];
    let (model_height, model_width, _) =
        if input_attr.fmt == rknn_ffi::raw::_rknn_tensor_format_RKNN_TENSOR_NHWC {
            (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3])
        } else {
            (input_attr.dims[2], input_attr.dims[3], input_attr.dims[1])
        };

    let (image_data_i8, letterbox_info) = preprocess_letterbox_quantize(
        image_path,
        model_width,
        model_height,
        input_attr.zp,
        input_attr.scale,
    )?;

    let image_data_u8: &[u8] = unsafe {
        std::slice::from_raw_parts(image_data_i8.as_ptr() as *const u8, image_data_i8.len())
    };
    ctx.set_input(
        0,
        rknn_ffi::raw::_rknn_tensor_type_RKNN_TENSOR_INT8,
        input_attr.fmt,
        image_data_u8,
    )
    .map_err(RknnError)?;
    ctx.run().map_err(RknnError)?;

    let outputs_obj = ctx.get_outputs().map_err(RknnError)?;
    let output_attrs = ctx.query_output_attrs().map_err(RknnError)?;
    let outputs_data: Vec<&[u8]> = outputs_obj
        .all()
        .iter()
        .map(|o| unsafe { std::slice::from_raw_parts(o.buf as *const u8, o.size as usize) })
        .collect();

    let detections = postprocess::post_process_i8(
        &outputs_data,
        &output_attrs,
        args.conf_thresh,
        args.iou_thresh,
        letterbox_info,
    );

    drawing::draw_results(&mut original_image, &detections, labels);

    // 决定输出路径
    let output_path = if Path::new(&args.output).is_dir() {
        let file_name = image_path.file_name().ok_or("Invalid image path")?;
        PathBuf::from(&args.output).join(file_name)
    } else {
        PathBuf::from(&args.output)
    };

    println!("Saving result to: {:?}", output_path);
    original_image.save(&output_path)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 解析命令行参数
    let args = Args::parse();
    println!("配置参数: {:?}", args);
    // --- 2. 使用解析出的路径 ---
    let model_path = Path::new(&args.model);
    let input_path = Path::new(&args.input);
    let labels_path = Path::new(&args.labels);
    let output_path = Path::new(&args.output);

    // --- 1. 初始化模型和标签 ---
    let model_data = fs::read(model_path)?;
    let mut ctx = RknnContext::new(&model_data, 0, None).map_err(RknnError)?;
    let labels = drawing::load_labels(labels_path)?;

    // --- 2. 判断输入是文件还是目录 ---
    if input_path.is_file() {
        if output_path.is_dir() {
            // 如果输出是目录，确保它存在
            fs::create_dir_all(output_path)?;
        }
        process_single_image(&mut ctx, input_path, &labels, &args)?;
    } else if input_path.is_dir() {
        // 确保输出目录存在
        fs::create_dir_all(output_path)?;

        for entry in WalkDir::new(input_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                match ext.to_lowercase().as_str() {
                    "jpg" | "jpeg" | "png" | "bmp" => {
                        if let Err(e) = process_single_image(&mut ctx, path, &labels, &args) {
                            eprintln!("Error processing file {:?}: {}", path, e);
                        }
                    }
                    _ => {}
                }
            }
        }
    } else {
        return Err(format!(
            "Input path not found or is not a file/directory: {}",
            args.input
        )
        .into());
    }

    println!("\nAll tasks completed.");
    Ok(())
}
