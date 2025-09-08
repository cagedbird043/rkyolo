use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
mod map_computer;

/// 定义 --split 参数的选项
#[derive(Clone, Debug, Default, ValueEnum)]
enum Split {
    #[default]
    Val,
    Test,
    Train,
}

// --- 1. 定义命令行接口 ---
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// 数据集配置文件 (dataset.yaml) 的路径
    #[arg(short, long)]
    data: PathBuf,

    /// `rkyolo-app` 生成的预测结果文件路径 (JSON 格式)
    #[arg(short, long)]
    preds: PathBuf,

    /// 【新增】指定要评估的数据集子集
    #[arg(long, value_enum, default_value_t = Split::Val)]
    split: Split,

    /// 设置日志详细程度 (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// 【新增】列出计数差异超过此阈值的图片
    #[arg(long)]
    list_outliers: Option<u32>,
}

// --- 2. 定义与 `dataset.yaml` 匹配的数据结构 ---
#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    path: PathBuf,
    // 我们暂时只关心 test 字段，val 和 train 可以后续添加
    // test 字段的值可能是可选的
    test: Option<PathBuf>,
    val: Option<PathBuf>,
    train: Option<PathBuf>,
    names: HashMap<u32, String>,
}

// --- 3. 定义预测结果的数据结构 (对应 JSON 文件) ---
// 稍后 rkyolo-app 会生成这种格式
#[derive(Debug, Deserialize, Copy, Clone)]
pub struct Bbox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Deserialize)]
pub struct Prediction {
    class_id: u32,
    confidence: f32,
    bbox: Bbox,
}

#[derive(Debug, Clone)]
pub struct GroundTruthBox {
    class_id: u32,
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
}

/// 加载单个图片的 Ground Truth 标签文件。
fn load_ground_truth(labels_dir: &Path, image_filename: &str) -> Result<Vec<GroundTruthBox>> {
    // 1. 推断标签文件的路径 (e.g., "image.jpg" -> "image.txt")
    let label_path = labels_dir.join(Path::new(image_filename).with_extension("txt"));

    // 2. 如果标签文件不存在，说明该图片没有物体，返回一个空 Vec 是正确行为。
    if !label_path.exists() {
        return Ok(Vec::new());
    }

    // 3. 打开并逐行解析文件
    let file = File::open(&label_path)
        .with_context(|| format!("Failed to open ground truth file: {:?}", &label_path))?;
    let reader = BufReader::new(file);
    let mut boxes = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        if parts.len() == 5 {
            boxes.push(GroundTruthBox {
                class_id: parts[0] as u32,
                x_center: parts[1],
                y_center: parts[2],
                width: parts[3],
                height: parts[4],
            });
        }
    }

    Ok(boxes)
}

/// 主函数
fn main() -> Result<()> {
    let args = Args::parse();

    // --- 【新增】初始化日志系统 ---
    env_logger::Builder::new()
        .filter_level(match args.verbose {
            0 => log::LevelFilter::Info,
            1 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace, // 使用 -vv 激活 trace
        })
        .init();

    // --- 加载并解析 dataset.yaml ---
    println!("Loading dataset config from: {:?}", &args.data);
    let f = File::open(&args.data)
        .with_context(|| format!("Failed to open dataset config file: {:?}", &args.data))?;
    let config: DatasetConfig = serde_yaml::from_reader(f)
        .with_context(|| format!("Failed to parse YAML from file: {:?}", &args.data))?;

    println!("Dataset base path: {:?}", &config.path);
    println!("Class names: {:?}", &config.names);

    // --- 确定要评估的图片集路径 ---
    // --- 【修正】根据 --split 参数确定要评估的图片集路径 ---
    let split_path =
        match args.split {
            Split::Val => config
                .val
                .as_ref()
                .context("Dataset config is missing the 'val' key, but --split=val was specified."),
            Split::Test => config.test.as_ref().context(
                "Dataset config is missing the 'test' key, but --split=test was specified.",
            ),
            Split::Train => config.train.as_ref().context(
                "Dataset config is missing the 'train' key, but --split=train was specified.",
            ),
        }?;

    // 解析 YAML 中的 `path` 字段，并将其与 split 路径结合
    let dataset_base_path = args
        .data
        .parent()
        .context("Failed to get parent directory of YAML file")?
        .join(&config.path);
    let test_images_path = dataset_base_path.join(split_path);

    println!("Evaluating on split '{:?}'", args.split);
    println!("Found evaluation image set at: {:?}", test_images_path);

    // --- 加载预测结果 ---
    println!("\nLoading predictions from: {:?}", &args.preds);
    let f = File::open(&args.preds)
        .with_context(|| format!("Failed to open predictions file: {:?}", &args.preds))?;
    // 预测结果是大文件，使用 BufReader 提高效率
    let reader = std::io::BufReader::new(f);
    // 我们的 JSON 将是 { "image_name.jpg": [Prediction, ...], ... } 的格式
    let predictions: HashMap<String, Vec<Prediction>> = serde_json::from_reader(reader)
        .with_context(|| {
            format!(
                "Failed to parse JSON from predictions file: {:?}",
                &args.preds
            )
        })?;

    println!("Loaded predictions for {} images.", predictions.len());

    // --- 【新增】加载 Ground Truth ---
    // 假设 'labels' 目录与 'images' 目录平行。
    // e.g., 'data/test/images' -> 'data/test/labels'
    let mut labels_dir_path = test_images_path.clone();
    if labels_dir_path.ends_with("images") {
        labels_dir_path.pop(); // 移除 'images'
        labels_dir_path.push("labels");
    } else {
        anyhow::bail!(
            "The image directory path does not end with 'images'. Cannot infer labels directory path. Please follow the 'images'/'labels' convention."
        );
    }
    println!("\nExpecting ground truth labels in: {:?}", &labels_dir_path);

    let mut ground_truths: HashMap<String, Vec<GroundTruthBox>> = HashMap::new();
    // 【新增】用于存储每张图片的尺寸 (width, height)
    let mut image_dimensions: HashMap<String, (u32, u32)> = HashMap::new();
    let mut images_with_gt = 0;

    for image_filename in predictions.keys() {
        // 加载 Ground Truth (不变)
        let gt_boxes = load_ground_truth(&labels_dir_path, image_filename)?;
        if !gt_boxes.is_empty() {
            images_with_gt += 1;
        }
        ground_truths.insert(image_filename.clone(), gt_boxes);

        // 【新增】加载图片以获取尺寸
        let image_path = test_images_path.join(image_filename);
        let dims = image::image_dimensions(&image_path)
            .with_context(|| format!("Failed to read dimensions of image: {:?}", &image_path))?;
        image_dimensions.insert(image_filename.clone(), dims);
    }

    println!(
        "Loaded ground truth and dimensions for {} images ({} of which had objects).",
        ground_truths.len(),
        images_with_gt
    );
    // --- 结束新增 ---

    // --- 【新增】调用 mAP 计算 ---
    let num_classes = config.names.len() as u32;
    let iou_threshold = 0.5; // 这是 COCO 标准常用的 IoU 阈值

    let map = map_computer::calculate_map(
        &predictions,
        &ground_truths,
        &image_dimensions,
        num_classes,
        iou_threshold,
    );

    // --- 【新增】计算 R² (决定系数) ---
    let confidence_threshold_for_r2 = 0.25; // 这是一个用于计数的常见置信度阈值
    let r_squared = map_computer::calculate_r_squared(
        &predictions,
        &ground_truths,
        confidence_threshold_for_r2,
        args.list_outliers, // <-- 传递新参数
    );

    println!("\n--- Evaluation Report ---");
    println!("mAP @ IoU={:.2}: {:.4}", iou_threshold, map);

    // 【新增】打印 R² 结果
    if let Some(r2_value) = r_squared {
        println!(
            "R² (Count Accuracy) @ conf>{:.2}: {:.4}",
            confidence_threshold_for_r2, r2_value
        );
    } else {
        println!("R² (Count Accuracy): N/A (not enough data to compute)");
    }

    println!("\nYAML parsing and data loading pub structure is ready.");

    Ok(())
}
