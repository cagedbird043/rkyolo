# RKYOLO: 一个健壮、高性能的 Rust YOLO 推理与评估框架

[English](./README.md)

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RKYOLO** 是一个从零开始、使用 Rust 构建的现代化、生产就绪的 AI 应用套件。它利用瑞芯微（Rockchip）的 NPU 对 YOLO 模型进行硬件加速推理，并提供了一套用于严格模型评估的完整工具链，是那些脆弱、硬编码的官方 C++ 示例的一个优越替代品。

本项目的诞生源于一种需求：一个能够提供 Rust 内存安全保障、拥有基于 Cargo 的现代构建系统、以及灵活架构的框架。它旨在实现接近硬件极限的**实时视频处理**性能，同时提供科学测量和理解模型表现的工具。

## 🌟 核心特性

- **内存安全 & 健壮**: 使用 Rust 构建，在编译时即可消除如段错误、内存泄漏等一整类 Bug。
- **极致零拷贝性能**: 默认实现零拷贝（Zero-Copy）数据路径，将预处理后的数据直接写入 DMA 缓冲区，最大限度减少 CPU 开销和 I/O 延迟。
- **🎥 多功能输入源**: 无缝处理来自**图片、目录、视频文件和 V4L2 摄像头**（例如 `/dev/video0`）的数据。
- **统一 YAML 配置**: 推理应用和评估工具均可由标准的 `dataset.yaml` 文件驱动，极大地简化了整个工作流。
- **实时视频处理**: 对视频流进行高帧率实时推理，支持实时结果显示、FPS 计数器和可选的硬件加速视频录制。
- **📈 全面的评估套件 (`rkyolo-eval`)**:
  - 计算业界标准的 **mAP@0.5** 来衡量检测精度。
  - 计算 **R² (决定系数)** 来衡量目标计数准确性。
  - 自动识别并报告模型表现不佳的**“离群值”图片**，加速调试与分析过程。
- **清晰的模块化架构**: 采用 Cargo Workspace 组织，在 FFI 绑定层 (`rknn-ffi`)、核心逻辑 (`rkyolo-core`)、推理应用 (`rkyolo-app`) 和评估工具 (`rkyolo-eval`) 之间实现了清晰的职责分离。

## 🚀 快速开始

### 先决条件

- 一块已正确安装官方 SDK 和 RKNN 运行时 (`librknnrt.so`) 的瑞芯微开发板（如 RK3588）。
- 通过 [rustup](https://rustup.rs/) 在设备上安装 Rust 工具链。
- C 语言工具链 (`gcc`)。
- **OpenCV 开发库** (`sudo apt install libopencv-dev`)。
- 一个已转换为 `.rknn` 格式的 YOLO 模型。

### 构建项目

```bash
# 克隆仓库
git clone https://github.com/your-username/rkyolo.git
cd rkyolo

# 以 release 模式构建以获取最佳性能
# 注意：首次构建可能因 opencv crate 而耗时较长。
cargo build --release
```

## 💻 使用流程

推荐的工作流利用 `dataset.yaml` 文件，以实现从推理到评估的无缝衔接。

### 第一步：使用 `rkyolo-app` 运行推理

使用 `dataset.yaml` 指定您的数据集，并将预测结果保存到一个 JSON 文件。

```bash
# 在验证集上运行推理并保存预测结果
./target/release/rkyolo-app \
    -m ./path/to/your_model.rknn \
    -d ./path/to/your/dataset.yaml \
    --split val \
    --conf-thresh 0.10 \
    --iou-thresh 0.57 \
    --save-preds predictions.json
```

### 第二步：使用 `rkyolo-eval` 评估性能

使用相同的 `dataset.yaml` 和上一步生成的 `predictions.json`，获取一份完整的性能报告。

```bash
# 评估已保存的预测结果，并列出计数差异大于 20 的离群图片
./target/release/rkyolo-eval \
    -d ./path/to/your/dataset.yaml \
    -p ./predictions.json \
    --split val \
    --list-outliers 20 \
    -v
```

### 其他 `rkyolo-app` 示例

#### 实时摄像头推理

```bash
./target/release/rkyolo-app -m model.rknn -s /dev/video0 -l labels.txt
```

#### 处理目录并保存可视化结果

```bash
./target/release/rkyolo-app \
    -m model.rknn \
    -s ./input_images_dir \
    -l labels.txt \
    -o ./output_visuals_dir
```

使用 `--help` 查看所有选项。

## 📜 许可证

本项目基于 MIT 许可证。
