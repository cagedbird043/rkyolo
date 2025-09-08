use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

// --- 1. 定义命令行接口 ---
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// 数据集配置文件 (dataset.yaml) 的路径
    #[arg(short, long)]
    data: PathBuf,

    /// `rkyolo-app` 生成的预测结果文件路径 (JSON 格式)
    #[arg(short, long)]
    preds: PathBuf,
}

// --- 2. 定义与 `dataset.yaml` 匹配的数据结构 ---
#[derive(Debug, Deserialize)]
struct DatasetConfig {
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
#[derive(Debug, Deserialize)]
struct Bbox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}
#[derive(Debug, Deserialize)]
struct Prediction {
    class_id: u32,
    confidence: f32,
    bbox: Bbox,
}

/// 主函数
fn main() -> Result<()> {
    let args = Args::parse();

    // --- 加载并解析 dataset.yaml ---
    println!("Loading dataset config from: {:?}", &args.data);
    let f = File::open(&args.data)
        .with_context(|| format!("Failed to open dataset config file: {:?}", &args.data))?;
    let config: DatasetConfig = serde_yaml::from_reader(f)
        .with_context(|| format!("Failed to parse YAML from file: {:?}", &args.data))?;

    println!("Dataset base path: {:?}", &config.path);
    println!("Class names: {:?}", &config.names);

    // --- 确定要评估的图片集路径 ---
    let test_images_path = match &config.test {
        Some(path) => config.path.join(path),
        None => {
            // 如果 test 字段不存在，我们可以尝试使用 val 字段
            match &config.val {
                Some(path) => config.path.join(path),
                None => anyhow::bail!(
                    "Dataset config must contain a 'test' or 'val' field for evaluation."
                ),
            }
        }
    };
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

    // --- 接下来的步骤将在这里实现 ---
    // 1. 遍历 test_images_path 目录，找到所有图片文件
    // 2. 为每张图片，去对应的 `labels` 目录加载 Ground Truth
    // 3. 将 Ground Truth 和 Predictions 组合起来
    // 4. 调用我们的 mAP 计算函数
    // 5. 打印最终报告

    println!("\nYAML parsing and data loading structure is ready.");

    Ok(())
}
