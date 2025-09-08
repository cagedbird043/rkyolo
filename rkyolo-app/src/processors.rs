use crate::config::{Bbox, InputSource, OutputMode, Prediction}; // <-- 从我们自己的模块导入
use crate::utils::{ExecutionMode, log, log_detection_summary}; // <-- 从我们自己的模块导入
use anyhow::Result;
use log::{info, trace, warn};
use opencv::{core, highgui, imgproc, prelude::*, videoio};
use rkyolo_core::{
    Detection, Lang, RknnContext, RknnError, draw_results, image, post_process_i8,
    preprocess_letterbox_quantize_from_buffer, preprocess_letterbox_quantize_zero_copy_from_buffer,
    rknn_ffi,
};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{env, fs};
use walkdir::WalkDir;

/// 【全新】处理目录中所有图片的主函数
pub fn process_directory(
    ctx: &mut RknnContext,
    mode: &mut ExecutionMode,
    dir_path: &Path,
    labels: &[String],
    conf_thresh: f32,
    iou_thresh: f32,
    output_mode: &OutputMode,
    save_preds_path: &Option<PathBuf>,
    lang: &Lang,
) -> Result<(), Box<dyn Error>> {
    let mut all_predictions: HashMap<String, Vec<Prediction>> = HashMap::new();

    // 使用 walkdir 遍历目录，只处理顶层文件
    for entry in WalkDir::new(dir_path)
        .min_depth(1)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // 检查文件扩展名是否为支持的图片格式
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext.to_lowercase().as_str() {
                "jpg" | "jpeg" | "png" | "bmp" => {
                    info!("-> Processing: {:?}", path);

                    // 根据主输出模式，为 process_single_image 构造临时的输出模式
                    let single_image_output_mode = match output_mode {
                        OutputMode::Directory(output_dir) => {
                            // 保留原始文件名，并构建新的输出路径
                            let output_file_path = output_dir.join(path.file_name().unwrap());
                            OutputMode::File(output_file_path)
                        }
                        _ => OutputMode::None,
                    };

                    match process_single_image(
                        ctx,
                        mode,
                        path,
                        labels,
                        conf_thresh,
                        iou_thresh,
                        &single_image_output_mode,
                        lang,
                    ) {
                        Ok(detections) => {
                            // 如果需要保存预测结果，则收集它们
                            if save_preds_path.is_some() {
                                let image_filename =
                                    path.file_name().unwrap().to_str().unwrap().to_string();
                                let preds: Vec<Prediction> = detections
                                    .into_iter()
                                    .map(|d| Prediction {
                                        class_id: d.class_id,
                                        confidence: d.confidence,
                                        bbox: Bbox {
                                            x1: d.bbox.x1,
                                            y1: d.bbox.y1,
                                            x2: d.bbox.x2,
                                            y2: d.bbox.y2,
                                        },
                                    })
                                    .collect();
                                all_predictions.insert(image_filename, preds);
                            }
                        }
                        Err(e) => {
                            // 错误容忍：记录错误并继续处理下一个文件
                            warn!("   [!] Failed to process {:?}: {}", path, e);
                        }
                    }
                }
                _ => {
                    trace!("Skipping non-image file: {:?}", path);
                }
            }
        }
    }

    // 处理完所有图片后，如果需要，则保存所有预测结果
    if let Some(path) = save_preds_path {
        info!("Saving all predictions to JSON file: {:?}", path);
        let json_string = serde_json::to_string_pretty(&all_predictions)?;
        fs::write(path, json_string)?;
    }

    Ok(())
}

/// 【全新】处理视频源的主循环
pub fn process_video_source(
    ctx: &mut RknnContext,
    mode: &mut ExecutionMode,
    source: &InputSource,
    labels: &[String],
    conf_thresh: f32,
    iou_thresh: f32,
    output_video_path: &Option<PathBuf>,
    is_headless: bool,
) -> Result<()> {
    let mut cap = match source {
        InputSource::CameraDevice(device_id) => {
            info!("Opening camera device with ID: {}", device_id);
            videoio::VideoCapture::new(*device_id, videoio::CAP_ANY)?
        }
        InputSource::VideoFile(path) => {
            info!("Opening video file: {:?}", path);
            videoio::VideoCapture::from_file(path.to_str().unwrap(), videoio::CAP_ANY)?
        }
        // 其他情况理论上不会传入，但在 match 中处理更安全
        _ => anyhow::bail!("Invalid input source type provided to process_video_source"),
    };

    if !cap.is_opened()? {
        // 使用 {:?} 来打印 Debug 信息
        anyhow::bail!("Failed to open video source: {:?}", source);
    }

    let window_name = "RKYOLO Live";
    if !is_headless {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;
    }

    let mut frame = Mat::default();
    let mut rgb_frame = Mat::default();

    // 【新增】根据 --output-video 参数，有条件地设置 FFmpeg 硬件编码器环境变量
    let _env_guard = if let Some(output_path) = &output_video_path {
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
    let mut writer = match output_video_path {
        Some(path) => {
            let fourcc = videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?;
            let fps = cap.get(videoio::CAP_PROP_FPS)?;
            let size = core::Size::new(
                cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
                cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
            );
            info!("Recording output to '{:?}' at {} FPS", path, fps);
            Some(videoio::VideoWriter::new(
                path.to_str().unwrap(),
                fourcc,
                fps,
                size,
                true,
            )?)
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
                )
                .map_err(anyhow::Error::msg)?
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
                )
                .map_err(anyhow::Error::msg)?;
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
            conf_thresh,
            iou_thresh,
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

        if !is_headless {
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
        if !is_headless {
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

// 图片处理函数 (从文件)
pub fn process_single_image(
    ctx: &mut RknnContext,
    mode: &mut ExecutionMode,
    image_path: &Path,
    labels: &[String],
    conf_thresh: f32,
    iou_thresh: f32,
    output_mode: &OutputMode,
    lang: &Lang,
) -> Result<Vec<Detection>, Box<dyn Error>> {
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
        conf_thresh,
        iou_thresh,
        letterbox_info,
    );

    log_detection_summary(&detections, labels, &lang);
    draw_results(&mut original_image, &detections, labels);

    if let OutputMode::File(output_path) = output_mode {
        info!("Saving result to: {:?}", output_path);
        original_image.save(output_path)?;
    }
    Ok(detections)
}
