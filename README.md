# RKYOLO: A Robust, High-Performance YOLO Inference & Evaluation Framework in Rust

[中文](./README_zh.md)

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RKYOLO** is a modern, production-ready AI application suite built from the ground up in Rust. It leverages Rockchip's NPU for hardware-accelerated inference of YOLO models and provides a comprehensive toolkit for rigorous model evaluation, offering a superior alternative to brittle, hardcoded C++ examples.

This project was born out of the need for a framework that offers Rust's memory safety, a modern Cargo-based build system, and a flexible architecture that achieves near-hardware-limit performance for **real-time video processing** while providing the tools to scientifically measure and understand model performance.

## 🌟 Core Features

- **Memory Safe & Robust**: Built with Rust to eliminate entire classes of bugs like segmentation faults and memory leaks at compile time.
- **Blazing Fast Zero-Copy Performance**: Implements a zero-copy data path by default, writing pre-processed data directly into DMA buffers to minimize CPU overhead and I/O latency.
- **🎥 Versatile Input Sources**: Process data from **images, directories, video files, and V4L2 cameras** (e.g., `/dev/video0`) seamlessly.
- **Unified YAML Configuration**: Both the application and evaluation tool can be driven by a standard `dataset.yaml` file, streamlining the entire workflow.
- **Live Video Processing**: High-framerate, real-time inference on video streams with live result display, FPS counter, and optional hardware-accelerated video recording.
- **📈 Comprehensive Evaluation Suite (`rkyolo-eval`)**:
  - Calculates industry-standard **mAP@0.5** to measure detection accuracy.
  - Calculates **R² (Coefficient of Determination)** to measure object counting accuracy.
  - Automatically identifies and reports on **outlier images** where the model performs poorly, accelerating debugging and analysis.
- **Clean, Modular Architecture**: Organized as a Cargo Workspace with a clear separation of concerns between the FFI layer (`rknn-ffi`), core logic (`rkyolo-core`), inference app (`rkyolo-app`), and evaluation tool (`rkyolo-eval`).

## 🚀 Getting Started

### Prerequisites

- A Rockchip development board (e.g., RK3588) with the official SDK and RKNN runtime (`librknnrt.so`) installed.
- Rust toolchain via [rustup](https://rustup.rs/).
- A C toolchain (`gcc`).
- **OpenCV development libraries** (`sudo apt install libopencv-dev`).
- A YOLO model converted to the `.rknn` format.

### Building the Project

```bash
# Clone the repository
git clone https://github.com/your-username/rkyolo.git
cd rkyolo

# Build in release mode for maximum performance
# Note: The first build may take time due to the opencv crate.
cargo build --release
```

## 💻 Usage Workflow

The recommended workflow leverages the `dataset.yaml` file for a seamless inference-to-evaluation pipeline.

### Step 1: Run Inference with `rkyolo-app`

Use the `dataset.yaml` to specify your dataset and save the prediction results to a JSON file.

```bash
# Run inference on the validation set and save predictions
./target/release/rkyolo-app \
    -m ./path/to/your_model.rknn \
    -d ./path/to/your/dataset.yaml \
    --split val \
    --conf-thresh 0.10 \
    --iou-thresh 0.57 \
    --save-preds predictions.json
```

### Step 2: Evaluate Performance with `rkyolo-eval`

Use the same `dataset.yaml` and the generated `predictions.json` to get a full performance report.

```bash
# Evaluate the saved predictions and list outliers with a count difference > 20
./target/release/rkyolo-eval \
    -d ./path/to/your/dataset.yaml \
    -p ./predictions.json \
    --split val \
    --list-outliers 20 \
    -v
```

### Other `rkyolo-app` Examples

#### Real-time Camera Inference

```bash
./target/release/rkyolo-app -m model.rknn -s /dev/video0 -l labels.txt
```

#### Process a Directory and Save Visual Results

```bash
./target/release/rkyolo-app \
    -m model.rknn \
    -s ./input_images_dir \
    -l labels.txt \
    -o ./output_visuals_dir
```

View all options with `--help`.

## 📜 License

This project is licensed under the MIT License.
