use clap::{Parser, ValueEnum};
use rkyolo_core::{
    RknnContext, RknnError, draw_results, image, load_labels, post_process_i8,
    preprocess_letterbox_quantize, rknn_ffi,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// 定义支持的语言选项
#[derive(ValueEnum, Clone, Debug, Default)]
enum Lang {
    #[default]
    En, // 英文
    Zh, // 中文
}

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

    /// 增加日志详细程度 (-v -> info, -vv -> debug, -vvv -> trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// 设置日志和输出信息的语言
    #[arg(long, value_enum, default_value_t = Lang::En)]
    lang: Lang,
}

/// 处理单张图片的完整流程
fn process_single_image(
    ctx: &mut RknnContext,
    image_path: &Path,
    labels: &[String],
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    println!("\n--- Processing Image: {:?} ---", image_path);

    let mut original_image = image::open(image_path)?.to_rgb8();

    let input_attrs = ctx.query_input_attrs().map_err(RknnError::from)?;
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

    let detections = post_process_i8(
        &outputs_data,
        &output_attrs,
        args.conf_thresh,
        args.iou_thresh,
        letterbox_info,
    );

    draw_results(&mut original_image, &detections, labels);

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
    // 注意：我们稍后会用 log::debug! 宏来替换下面的 println!
    println!("配置参数: {:?}", args);

    // --- 2. 使用解析出的路径 ---
    let model_path = Path::new(&args.model);
    let input_path = Path::new(&args.input);
    let labels_path = Path::new(&args.labels);
    let output_path = Path::new(&args.output);

    // --- 1. 初始化模型和标签 ---
    let model_data = fs::read(model_path)?;
    let mut ctx = RknnContext::new(&model_data, 0, None).map_err(RknnError)?;
    let labels = load_labels(labels_path)?;

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
