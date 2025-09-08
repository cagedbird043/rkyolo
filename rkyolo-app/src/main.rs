mod config;
mod processors;
mod utils;

use crate::config::{Args, InputSource, validate_and_build_config};
use crate::processors::{process_directory, process_single_image, process_video_source};
use crate::utils::{ExecutionMode, log_tensor_attrs, try_setup_zero_copy};
use clap::Parser;
use log::{debug, error, info};
use rkyolo_core::{RknnContext, RknnError, load_labels};
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. 解析原始参数
    let args = Args::parse();
    env_logger::Builder::new()
        .filter_level(match args.verbose {
            0 => log::LevelFilter::Info,
            1 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .init();

    // 2. 验证参数并构建配置，如果失败则提前退出
    let config = match validate_and_build_config(args) {
        Ok(c) => c,
        Err(e) => {
            error!("Configuration error: {}", e);
            // 使用 anyhow 的上下文来提供更丰富的错误信息
            for (i, cause) in e.chain().skip(1).enumerate() {
                error!("  Caused by [{}]: {}", i + 1, cause);
            }
            std::process::exit(1);
        }
    };
    debug!("Validated AppConfig: {:?}", config);

    // 3. 初始化模型和标签
    let model_data = fs::read(&config.model_path)?;
    let mut ctx = RknnContext::new(&model_data, 0, None).map_err(RknnError)?;
    let labels = load_labels(&config.labels_path)?;

    // 4. 初始化执行模式 (Zero-Copy 或 Standard)
    let mut execution_mode = if config.disable_zero_copy {
        info!("Zero-copy mode disabled by argument.");
        ExecutionMode::Standard
    } else {
        try_setup_zero_copy(&mut ctx, &config.lang)?
    };

    info!("--- Model Initialized ---");
    let input_attrs = ctx.query_input_attrs().map_err(RknnError::from)?;
    let output_attrs = ctx.query_output_attrs().map_err(RknnError::from)?;
    log_tensor_attrs(&input_attrs, "Input Tensors", "输入张量", &config.lang);
    log_tensor_attrs(&output_attrs, "Output Tensors", "输出张量", &config.lang);
    info!("-------------------------");

    // 5. 根据输入源分发到不同的处理函数
    match &config.input_source {
        InputSource::SingleImage(path) => {
            info!("\nProcessing single image: {:?}", path);
            process_single_image(
                &mut ctx,
                &mut execution_mode,
                path,
                &labels,
                config.conf_thresh,
                config.iou_thresh,
                &config.output_mode,
                &config.lang,
            )?;
        }
        InputSource::ImageDirectory(path) => {
            info!("\nProcessing all images in directory: {:?}", path);
            process_directory(
                &mut ctx,
                &mut execution_mode,
                path,
                &labels,
                config.conf_thresh,
                config.iou_thresh,
                &config.output_mode,
                &config.save_preds_path,
                &config.lang,
            )?;
        }
        InputSource::VideoFile(_) | InputSource::CameraDevice(_) => {
            process_video_source(
                &mut ctx,
                &mut execution_mode,
                &config.input_source,
                &labels,
                config.conf_thresh,
                config.iou_thresh,
                &config.output_video_path,
                config.headless,
            )?;
        }
    }

    info!("\nAll tasks completed.");
    Ok(())
}
