use anyhow::Result;
use clap::Parser;
use log::info;
use rkyolo_core::Lang;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

// 【新增】用于序列化预测结果的数据结构
#[derive(Debug, Serialize, Deserialize)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub class_id: i32,
    pub confidence: f32,
    pub bbox: Bbox,
}

/// 表示解析后的输入源类型
#[derive(Debug, Clone)]
pub enum InputSource {
    SingleImage(PathBuf),
    ImageDirectory(PathBuf),
    VideoFile(PathBuf),
    CameraDevice(i32),
}

/// 表示解析后的输出模式
#[derive(Debug, Clone)]
pub enum OutputMode {
    /// 不保存图片/视频，仅显示（如果适用）或记录日志
    None,
    /// 将单张结果图片保存到指定文件
    File(PathBuf),
    /// 将多张结果图片保存到指定目录
    Directory(PathBuf),
}

/// 存储经过验证和解析后的应用程序配置
#[derive(Debug)]
pub struct AppConfig {
    pub model_path: PathBuf,
    pub labels_path: PathBuf,
    pub conf_thresh: f32,
    pub iou_thresh: f32,
    pub lang: Lang,
    pub disable_zero_copy: bool,
    pub headless: bool,

    // --- 核心配置 ---
    pub input_source: InputSource,
    pub output_mode: OutputMode,
    pub output_video_path: Option<PathBuf>,
    pub save_preds_path: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "assets/yolo11.rknn")]
    model: String,
    #[arg(short, long, default_value = "assets/bus.jpg")]
    source: String,
    #[arg(short, long, default_value = "assets/coco_labels.txt")]
    labels: String,
    #[arg(short, long)]
    output: Option<String>,
    #[arg(long, default_value_t = 0.25)]
    conf_thresh: f32,
    #[arg(long, default_value_t = 0.45)]
    iou_thresh: f32,
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
    #[arg(long, value_enum, default_value_t = rkyolo_core::Lang::En)]
    lang: Lang,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    disable_zero_copy: bool,
    #[arg(long)]
    output_video: Option<String>,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    headless: bool,
    /// 【新增】将检测结果以 JSON 格式保存到文件，用于后续评估
    #[arg(long)]
    save_preds: Option<PathBuf>,
}

/// 验证命令行参数并构建一个清晰的 AppConfig。
/// 这是所有业务逻辑的入口点，确保配置的有效性。
pub fn validate_and_build_config(args: Args) -> Result<AppConfig> {
    let source_path = Path::new(&args.source);
    let model_path = PathBuf::from(&args.model);
    let labels_path = PathBuf::from(&args.labels);

    // --- 基础路径验证 ---
    if !model_path.is_file() {
        anyhow::bail!("Model file not found at: {:?}", model_path);
    }
    if !labels_path.is_file() {
        anyhow::bail!("Labels file not found at: {:?}", labels_path);
    }

    // --- 解析输入源 ---
    let input_source = if source_path.is_file() {
        let ext = source_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" | "png" | "bmp" => InputSource::SingleImage(source_path.to_path_buf()),
            "mp4" | "avi" | "mov" | "mkv" => InputSource::VideoFile(source_path.to_path_buf()),
            _ => anyhow::bail!("Unsupported file type: {:?}", source_path),
        }
    } else if source_path.is_dir() {
        InputSource::ImageDirectory(source_path.to_path_buf())
    } else if args.source.starts_with("/dev/video") {
        let device_id: i32 = args
            .source
            .trim_start_matches("/dev/video")
            .parse()
            .unwrap_or(0);
        InputSource::CameraDevice(device_id)
    } else {
        anyhow::bail!(
            "Source not found or is not a valid file, directory, or camera device: {:?}",
            args.source
        );
    };

    // --- 根据输入源验证输出模式 ---
    let (output_mode, output_video_path, save_preds_path) = match &input_source {
        InputSource::SingleImage(_) => {
            // --output-video 和 --save-preds 对单图片非法
            if args.output_video.is_some() {
                anyhow::bail!("--output-video cannot be used with a single image input.");
            }
            if args.save_preds.is_some() {
                anyhow::bail!("--save-preds is only for directory input for evaluation.");
            }
            // --output 必须是文件
            let mode = match args.output {
                Some(p) => {
                    let path = PathBuf::from(p);
                    if path.is_dir() {
                        anyhow::bail!(
                            "For single image input, --output must be a file path, not a directory."
                        );
                    }
                    OutputMode::File(path)
                }
                None => OutputMode::None, // 默认不保存
            };
            (mode, None, None)
        }
        InputSource::ImageDirectory(_) => {
            // --output-video 对目录非法
            if args.output_video.is_some() {
                anyhow::bail!("--output-video cannot be used with a directory input.");
            }
            // --output 必须是目录
            let mode = match args.output {
                Some(p) => {
                    let path = PathBuf::from(p);
                    // 检查路径是否存在且是否是一个文件
                    if path.is_file() {
                        anyhow::bail!(
                            "Output path {:?} exists but is a file, not a directory.",
                            path
                        );
                    }
                    // 如果路径不存在，则尝试创建它
                    if !path.exists() {
                        info!("Output directory {:?} does not exist. Creating it...", path);
                        // `create_dir_all` 等同于 `mkdir -p`
                        // `?` 操作符会自动处理并传播任何IO错误 (如权限不足)
                        fs::create_dir_all(&path)?;
                    }
                    OutputMode::Directory(path)
                }
                None => OutputMode::None,
            };
            (mode, None, args.save_preds)
        }
        InputSource::VideoFile(_) | InputSource::CameraDevice(_) => {
            // --output 和 --save-preds 对视频源非法
            if args.output.is_some() {
                anyhow::bail!(
                    "-o/--output cannot be used with a video or camera source. Use --output-video instead."
                );
            }
            if args.save_preds.is_some() {
                anyhow::bail!("--save-preds is only for directory input for evaluation.");
            }
            (OutputMode::None, args.output_video.map(PathBuf::from), None)
        }
    };

    Ok(AppConfig {
        model_path,
        labels_path,
        conf_thresh: args.conf_thresh,
        iou_thresh: args.iou_thresh,
        lang: args.lang,
        disable_zero_copy: args.disable_zero_copy,
        // 目录处理本质上是批处理，不应有实时窗口
        headless: args.headless || matches!(input_source, InputSource::ImageDirectory(_)),
        input_source,
        output_mode,
        output_video_path,
        save_preds_path,
    })
}
