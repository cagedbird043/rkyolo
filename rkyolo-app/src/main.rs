use clap::Parser;
use log::{debug, error, info, warn};
use opencv::{core, highgui, imgproc, prelude::*, videoio};
use rkyolo_core::rknn_ffi::raw::rknn_tensor_attr;
use rkyolo_core::{
    Detection, Lang, RknnContext, RknnError, draw_results, get_type_string, image, load_labels,
    post_process_i8, preprocess_letterbox_quantize_from_buffer,
    preprocess_letterbox_quantize_zero_copy_from_buffer, rknn_ffi, rknn_ffi::RknnTensorMem,
};
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{env, fs};
use walkdir::WalkDir; // 用于测量时间

enum ExecutionMode {
    ZeroCopy {
        input_mem: RknnTensorMem,
        input_attr: rknn_ffi::raw::rknn_tensor_attr,
    },
    Standard,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "assets/yolo11.rknn")]
    model: String,
    #[arg(short, long, default_value = "assets/bus.jpg")]
    source: String,
    #[arg(short, long, default_value = "assets/coco_labels.txt")]
    labels: String,
    #[arg(short, long, default_value = "output.jpg")]
    output: String,
    #[arg(long, default_value_t = 0.25)]
    conf_thresh: f32,
    #[arg(long, default_value_t = 0.45)]
    iou_thresh: f32,
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    #[arg(long, value_enum, default_value_t = rkyolo_core::Lang::En)]
    lang: Lang,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    disable_zero_copy: bool,
    #[arg(long)]
    output_video: Option<String>,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    headless: bool,
}

/// 【全新】处理视频源的主循环
fn process_video_source(
    ctx: &mut RknnContext,
    mode: &mut ExecutionMode,
    source: &str,
    labels: &[String],
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let mut cap = if source.starts_with("/dev/video") {
        let device_id: i32 = source.trim_start_matches("/dev/video").parse().unwrap_or(0);
        info!("Opening camera device with ID: {}", device_id);
        videoio::VideoCapture::new(device_id, videoio::CAP_ANY)?
    } else {
        info!("Opening video file: {}", source);
        videoio::VideoCapture::from_file(source, videoio::CAP_ANY)?
    };

    if !cap.is_opened()? {
        return Err(format!("Failed to open video source: {}", source).into());
    }

    let window_name = "RKYOLO Live";
    if !args.headless {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;
    }

    let mut frame = Mat::default();
    let mut rgb_frame = Mat::default();

    // 【新增】根据 --output-video 参数，有条件地设置 FFmpeg 硬件编码器环境变量
    let _env_guard = if let Some(output_path) = &args.output_video {
        if output_path.ends_with(".mp4") {
            // 或者你可以根据其他条件来判断是否需要硬件编码
            let encoder_options = format!("-codec:v h264_rkmpp"); // 显式指定硬件编码器
            info!(
                "Setting OPENCV_FFMPEG_WRITER_OPTIONS to: '{}'",
                encoder_options
            );
            // env::set_var 会返回一个 Guard，当 Guard 离开作用域时环境变量会被还原
            // 这确保了我们设置的环境变量只对 VideoWriter.new() 调用有效，不会污染全局环境
            Some(unsafe { env::set_var("OPENCV_FFMPEG_WRITER_OPTIONS", encoder_options) })
        } else {
            None
        }
    } else {
        None
    };

    // 【新增】根据命令行参数，有条件地初始化 VideoWriter
    let mut writer = match &args.output_video {
        Some(path) => {
            let fourcc = videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?; // 使用更通用的 H.264 FourCC tag
            let fps = cap.get(videoio::CAP_PROP_FPS)?;
            let size = core::Size::new(
                cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
                cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
            );
            info!("Recording output to '{}' at {} FPS", path, fps);
            Some(videoio::VideoWriter::new(path, fourcc, fps, size, true)?)
        }
        None => None,
    };

    let mut last_fps_update_time = Instant::now();
    let mut frame_count_since_last_update: u32 = 0;
    let mut fps_display_text = String::from("FPS: N/A");

    loop {
        cap.read(&mut frame)?;
        if frame.empty() {
            info!("Video stream ended.");
            break;
        }

        // 【新增】FPS 计算逻辑
        frame_count_since_last_update += 1;
        let elapsed_time = last_fps_update_time.elapsed();

        // 每秒钟更新一次 FPS 显示
        if elapsed_time.as_secs_f64() >= 1.0 {
            let calculated_fps = frame_count_since_last_update as f64 / elapsed_time.as_secs_f64();
            fps_display_text = format!("FPS: {:.2}", calculated_fps); // 直接使用计算结果
            frame_count_since_last_update = 0;
            last_fps_update_time = Instant::now();
        }

        // 1. 颜色空间转换 BGR -> RGB
        imgproc::cvt_color(&frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0)?;
        let frame_width = rgb_frame.cols() as u32;
        let frame_height = rgb_frame.rows() as u32;
        let frame_data = rgb_frame.data_bytes()?;

        // 2. 预处理 + 推理 (根据模式选择不同路径)
        let letterbox_info = match mode {
            ExecutionMode::ZeroCopy {
                input_mem,
                input_attr,
            } => {
                let (model_h, model_w, _) =
                    (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3]);
                preprocess_letterbox_quantize_zero_copy_from_buffer(
                    frame_width,
                    frame_height,
                    frame_data,
                    model_w,
                    model_h,
                    input_attr.w_stride,
                    input_attr.zp,
                    input_attr.scale,
                    input_mem.as_mut_slice(),
                )?
            }
            ExecutionMode::Standard => {
                let input_attrs = ctx.query_input_attrs().map_err(RknnError::from)?;
                let input_attr = &input_attrs[0];
                let (model_h, model_w, _) =
                    (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3]);

                let (image_data_i8, info) = preprocess_letterbox_quantize_from_buffer(
                    frame_width,
                    frame_height,
                    frame_data,
                    model_w,
                    model_h,
                    input_attr.zp,
                    input_attr.scale,
                )?;
                let image_data_u8: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        image_data_i8.as_ptr() as *const u8,
                        image_data_i8.len(),
                    )
                };
                ctx.set_input(
                    0,
                    rknn_ffi::raw::_rknn_tensor_type_RKNN_TENSOR_INT8,
                    input_attr.fmt,
                    image_data_u8,
                )
                .map_err(RknnError)?;
                info
            }
        };

        ctx.run().map_err(RknnError)?;

        // 3. 获取输出并进行后处理
        let outputs_obj = ctx.get_outputs().map_err(RknnError)?;
        let output_attrs = ctx.query_output_attrs().map_err(RknnError)?;
        let outputs_data: Vec<&[u8]> = outputs_obj
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

        // 4. 在原始帧 (BGR) 上绘制结果
        for det in &detections {
            let bbox = &det.bbox;
            let rect = core::Rect::new(
                bbox.x1 as i32,
                bbox.y1 as i32,
                (bbox.x2 - bbox.x1) as i32,
                (bbox.y2 - bbox.y1) as i32,
            );
            imgproc::rectangle(
                &mut frame,
                rect,
                core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            let label = labels
                .get(det.class_id as usize)
                .map_or("unknown", |s| s.as_str());
            let text = format!("{} {:.2}", label, det.confidence);
            let org = core::Point::new(rect.x, rect.y - 10);
            imgproc::put_text(
                &mut frame,
                &text,
                org,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.8,
                core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        if !args.headless {
            // 只有在非无头模式下才需要绘制和显示
            let org = core::Point::new(10, 30); // 绘制位置 (左上角，稍微偏移)
            let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0); // 绿色
            imgproc::put_text(
                &mut frame,
                &fps_display_text,
                org,
                imgproc::FONT_HERSHEY_SIMPLEX, // 字体
                1.0,                           // 字体大小
                color,
                2, // 字体粗细
                imgproc::LINE_8,
                false,
            )?;
        }

        // 【新增】如果 writer 存在，则将当前帧写入视频文件
        if let Some(writer) = &mut writer {
            writer.write(&frame)?;
        }

        // 【修改】根据 --headless 参数，有条件地显示图像并处理键盘事件
        if !args.headless {
            highgui::imshow(window_name, &frame)?;
            let key = highgui::wait_key(1)?;
            if key == 'q' as i32 || key == 27 {
                // 'q' or ESC key
                info!("User pressed 'q' or ESC. Exiting video stream.");
                break;
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    env_logger::Builder::new()
        .filter_level(match args.verbose {
            0 => log::LevelFilter::Info,
            1 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .init();
    debug!("Parsed arguments: {:?}", args);

    let model_data = fs::read(&args.model)?;
    let mut ctx = RknnContext::new(&model_data, 0, None).map_err(RknnError)?;
    let labels = load_labels(Path::new(&args.labels))?;

    let mut execution_mode = if args.disable_zero_copy {
        info!("Zero-copy mode disabled by argument.");
        ExecutionMode::Standard
    } else {
        try_setup_zero_copy(&mut ctx, &args.lang)?
    };

    info!("--- Model Initialized ---");
    let input_attrs = ctx.query_input_attrs().map_err(RknnError::from)?;
    let output_attrs = ctx.query_output_attrs().map_err(RknnError::from)?;
    log_tensor_attrs(&input_attrs, "Input Tensors", "输入张量", &args.lang);
    log_tensor_attrs(&output_attrs, "Output Tensors", "输出张量", &args.lang);
    info!("-------------------------");

    let source_path = Path::new(&args.source);
    let source_str = args.source.as_str();

    if source_str.starts_with("/dev/video")
        || source_str.ends_with(".mp4")
        || source_str.ends_with(".avi")
        || source_str.ends_with(".mov")
    {
        // 视频处理分支
        process_video_source(&mut ctx, &mut execution_mode, source_str, &labels, &args)?;
    } else if source_path.is_file() {
        // 单图片处理分支
        info!("\nProcessing single image: {:?}", source_path);
        process_single_image(&mut ctx, &mut execution_mode, source_path, &labels, &args)?;
    } else if source_path.is_dir() {
        // 目录处理分支
        info!("\nProcessing all images in directory: {:?}", source_path);
        for entry in WalkDir::new(source_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ["jpg", "jpeg", "png", "bmp"].contains(&ext.to_lowercase().as_str()) {
                    info!("\n--- Processing image: {:?} ---", path);
                    if let Err(e) =
                        process_single_image(&mut ctx, &mut execution_mode, path, &labels, &args)
                    {
                        error!("Error processing file {:?}: {}", path, e);
                    }
                }
            }
        }
    } else {
        return Err(format!(
            "Source not found or is not a valid file/directory/video device: {}",
            args.source
        )
        .into());
    }

    info!("\nAll tasks completed.");
    Ok(())
}

// 图片处理函数 (从文件)
fn process_single_image(
    ctx: &mut RknnContext,
    mode: &mut ExecutionMode,
    image_path: &Path,
    labels: &[String],
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let mut original_image = image::open(image_path)?.to_rgb8();
    let letterbox_info = match mode {
        ExecutionMode::ZeroCopy {
            input_mem,
            input_attr,
        } => {
            let (model_h, model_w, _) =
                (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3]);
            rkyolo_core::preprocess_letterbox_quantize_zero_copy(
                image_path,
                model_w,
                model_h,
                input_attr.w_stride,
                input_attr.zp,
                input_attr.scale,
                input_mem.as_mut_slice(),
            )
            .map_err(|e| e.to_string())?
        }
        ExecutionMode::Standard => {
            let input_attrs = ctx.query_input_attrs().map_err(RknnError::from)?;
            let input_attr = &input_attrs[0];
            let (model_h, model_w, _) =
                (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3]);
            let (image_data_i8, info) = rkyolo_core::preprocess_letterbox_quantize(
                image_path,
                model_w,
                model_h,
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
            info
        }
    };

    ctx.run().map_err(RknnError)?;
    let outputs_obj = ctx.get_outputs().map_err(RknnError)?;
    let output_attrs = ctx.query_output_attrs().map_err(RknnError)?;
    let outputs_data: Vec<&[u8]> = outputs_obj
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

    log_detection_summary(&detections, labels, &args.lang);
    draw_results(&mut original_image, &detections, labels);

    let output_path = if Path::new(&args.output).is_dir() {
        image_path.file_name().ok_or("Invalid image path")?.into()
    } else {
        PathBuf::from(&args.output)
    };
    info!("Saving result to: {:?}", output_path);
    original_image.save(&output_path)?;
    Ok(())
}

/// 尝试设置零拷贝执行模式。
/// 如果成功，返回配置好的 ZeroCopy 模式。
/// 如果失败，打印警告并返回 Standard 模式作为降级。
fn try_setup_zero_copy(ctx: &mut RknnContext, lang: &Lang) -> Result<ExecutionMode, RknnError> {
    // 1. 查询原生属性
    let native_input_attrs = match ctx.query_native_input_attrs() {
        Ok(attrs) => attrs,
        Err(e) => {
            match lang {
                Lang::En => warn!(
                    "Failed to query native input attributes (code: {}). Falling back to standard mode.",
                    e
                ),
                Lang::Zh => warn!("查询原生输入属性失败 (错误码: {})。将降级到标准模式。", e),
            }
            return Ok(ExecutionMode::Standard);
        }
    };
    // 我们只关心第一个输入
    let input_attr = if let Some(attr) = native_input_attrs.get(0) {
        attr.clone() // Clone attr so we can own it in ExecutionMode
    } else {
        match lang {
            Lang::En => warn!("No native input attributes found. Falling back to standard mode."),
            Lang::Zh => warn!("未找到原生输入属性。将降级到标准模式。"),
        }
        return Ok(ExecutionMode::Standard);
    };

    // 2. 创建 DMA 内存
    let input_mem = match ctx.create_mem(input_attr.size_with_stride) {
        Ok(mem) => mem,
        Err(e) => {
            match lang {
                Lang::En => warn!(
                    "Failed to create DMA memory (code: {}). Falling back to standard mode.",
                    e
                ),
                Lang::Zh => warn!("创建 DMA 内存失败 (错误码: {})。将降级到标准模式。", e),
            }
            return Ok(ExecutionMode::Standard);
        }
    };

    // 3. 绑定内存
    if let Err(e) = ctx.set_io_mem(&input_mem, &input_attr) {
        match lang {
            Lang::En => warn!(
                "Failed to set IO memory (code: {}). Falling back to standard mode.",
                e
            ),
            Lang::Zh => warn!("设置 IO 内存失败 (错误码: {})。将降级到标准模式。", e),
        }
        return Ok(ExecutionMode::Standard);
    }

    // 如果所有步骤都成功
    match lang {
        Lang::En => info!("Successfully initialized Zero-Copy mode."),
        Lang::Zh => info!("零拷贝模式初始化成功。"),
    }
    Ok(ExecutionMode::ZeroCopy {
        input_mem,
        input_attr,
    })
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
