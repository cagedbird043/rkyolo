use clap::Parser;
use log::{debug, error, info};
use rkyolo_core::rknn_ffi::raw::rknn_tensor_attr;
use rkyolo_core::{
    Detection, Lang, RknnContext, RknnError, draw_results, get_type_string, image, load_labels,
    post_process_i8, preprocess_letterbox_quantize, rknn_ffi,
};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

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

    /// 增加日志详细程度 (-v -> debug, -vv -> trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// 设置日志和输出信息的语言
    #[arg(long, value_enum, default_value_t = rkyolo_core::Lang::En)]
    lang: Lang,
}

/// 【新增】辅助函数：记录检测结果摘要
fn log_detection_summary(detections: &[Detection], labels: &[String], lang: &Lang) {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for det in detections {
        *counts.entry(det.class_id).or_insert(0) += 1;
    }

    match lang {
        Lang::En => {
            info!(
                "Post-processing complete. Found {} final objects (after NMS).",
                detections.len()
            );
            if !detections.is_empty() {
                info!("Detection summary:");
                for (class_id, count) in &counts {
                    let label = labels
                        .get(*class_id as usize)
                        .map_or("unknown", |s| s.as_str());
                    info!("  - Class '{}': {} objects", label, count);
                }
            }
        }
        Lang::Zh => {
            info!(
                "后处理完成，共找到 {} 个最终目标（NMS后）。",
                detections.len()
            );
            if !detections.is_empty() {
                info!("检测结果摘要:");
                for (class_id, count) in &counts {
                    let label = labels
                        .get(*class_id as usize)
                        .map_or("未知类别", |s| s.as_str());
                    info!("  - 类别 '{}': {} 个目标", label, count);
                }
            }
        }
    }
}

/// 【新增】辅助函数：记录张量属性
fn log_tensor_attrs(attrs: &[rknn_tensor_attr], name_en: &str, name_zh: &str, lang: &Lang) {
    let (name, _dims_str, format_str, type_str, qnt_str) = match lang {
        Lang::En => (name_en, "Dims", "Format", "Type", "Quantization"),
        Lang::Zh => (name_zh, "维度", "格式", "类型", "量化"),
    };
    info!("- {}: {}", name, attrs.len());
    for (i, attr) in attrs.iter().enumerate() {
        let dims: Vec<String> = attr.dims[..attr.n_dims as usize]
            .iter()
            .map(|d| d.to_string())
            .collect();
        let type_name = get_type_string(attr.type_);
        info!(
            "  - [{}]: {} [{}], {}: {}, {}: {}, {}: ZP={}, Scale={:.4}",
            i,
            attr.name.iter().map(|&c| c as char).collect::<String>(),
            dims.join("x"),
            format_str,
            if attr.fmt == rknn_ffi::raw::_rknn_tensor_format_RKNN_TENSOR_NHWC {
                "NHWC"
            } else {
                "NCHW"
            },
            type_str,
            type_name,
            qnt_str,
            attr.zp,
            attr.scale
        );
    }
}

/// 处理单张图片的完整流程
fn process_single_image(
    ctx: &mut RknnContext,
    image_path: &Path,
    labels: &[String],
    args: &Args,
) -> Result<(), Box<dyn Error>> {
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

    // 【新增】调用辅助函数打印检测摘要
    log_detection_summary(&detections, labels, &args.lang);

    draw_results(&mut original_image, &detections, labels);

    let output_path = if Path::new(&args.output).is_dir() {
        let file_name = image_path.file_name().ok_or("Invalid image path")?;
        PathBuf::from(&args.output).join(file_name)
    } else {
        PathBuf::from(&args.output)
    };

    match args.lang {
        Lang::En => info!("Saving result to: {:?}", output_path),
        Lang::Zh => info!("结果保存至: {:?}", output_path),
    }
    original_image.save(&output_path)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 【修改】默认级别为 Info
    let log_level = match args.verbose {
        0 => log::LevelFilter::Info,
        1 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::new().filter_level(log_level).init();

    debug!("Parsed command line arguments: {:?}", args);

    let model_path = Path::new(&args.model);
    let input_path = Path::new(&args.input);
    let labels_path = Path::new(&args.labels);
    let output_path = Path::new(&args.output);

    let model_data = fs::read(model_path)?;
    let mut ctx = RknnContext::new(&model_data, 0, None).map_err(RknnError)?;
    let labels = load_labels(labels_path)?;

    // 【新增】打印模型信息摘要
    info!(
        "{}",
        match args.lang {
            Lang::En => "--- Model Initialized ---",
            Lang::Zh => "--- 模型初始化完成 ---",
        }
    );
    let input_attrs = ctx.query_input_attrs().map_err(RknnError::from)?;
    let output_attrs = ctx.query_output_attrs().map_err(RknnError::from)?;
    log_tensor_attrs(&input_attrs, "Input Tensors", "输入张量", &args.lang);
    log_tensor_attrs(&output_attrs, "Output Tensors", "输出张量", &args.lang);
    info!("-------------------------");

    if input_path.is_file() {
        if output_path.is_dir() {
            fs::create_dir_all(output_path)?;
        }
        match args.lang {
            Lang::En => info!("\nProcessing single image: {:?}", input_path),
            Lang::Zh => info!("\n处理单张图片: {:?}", input_path),
        }
        process_single_image(&mut ctx, input_path, &labels, &args)?;
    } else if input_path.is_dir() {
        fs::create_dir_all(output_path)?;
        match args.lang {
            Lang::En => info!("\nProcessing all images in directory: {:?}", input_path),
            Lang::Zh => info!("\n处理目录中的所有图片: {:?}", input_path),
        }

        for entry in WalkDir::new(input_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                match ext.to_lowercase().as_str() {
                    "jpg" | "jpeg" | "png" | "bmp" => {
                        match args.lang {
                            Lang::En => info!("\n--- Processing image: {:?} ---", path),
                            Lang::Zh => info!("\n--- 正在处理图片: {:?} ---", path),
                        }
                        if let Err(e) = process_single_image(&mut ctx, path, &labels, &args) {
                            match args.lang {
                                Lang::En => error!("Error processing file {:?}: {}", path, e),
                                Lang::Zh => error!("处理文件 {:?} 时发生错误: {}", path, e),
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    } else {
        let err_msg = format!(
            "Input path not found or is not a file/directory: {}",
            args.input
        );
        error!("{}", &err_msg);
        return Err(err_msg.into());
    }
    match args.lang {
        Lang::En => info!("\nAll tasks completed."),
        Lang::Zh => info!("\n所有任务已完成。"),
    }
    Ok(())
}
