# RKYOLO: A Robust, High-Performance YOLO Inference Framework in Rust for Rockchip NPU

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RKYOLO** is a modern, robust, and high-performance AI application foundation built from the ground up in Rust. It leverages Rockchip's NPU for hardware-accelerated inference of YOLO models, providing a superior alternative to the often brittle and hardcoded official C++ examples.

This project was born out of the need for a production-ready framework that offers Rust's memory safety guarantees, a modern build system with Cargo, and a flexible, model-agnostic architecture that achieves near-hardware-limit performance for **real-time video processing**.

## 🌟 Core Features

- **Memory Safe**: Built with Rust to eliminate entire classes of bugs like segmentation faults and memory leaks at compile time, guaranteed by the compiler.
- **Blazing Fast Zero-Copy Performance**: Implements a zero-copy data path by default, writing pre-processed data directly into DMA buffers to minimize CPU overhead and I/O latency.
- **🎥 Real-time Video & Camera Support**: Process live video feeds from V4L2 devices (e.g., `/dev/video0`) or video files with high framerates, displaying results in a live window.
- **Intelligent Fallback**: Automatically and gracefully falls back to a standard memory-copy mode if zero-copy initialization fails, ensuring maximum performance without sacrificing robustness.
- **Model Agnostic & Adaptive**: Dynamically adapts to different YOLO model architectures (e.g., varying number of outputs, different class counts) without requiring any code changes. It just works.
- **Professional Logging**: Features a multi-level (`-v`, `-vv`) and multi-lingual (`--lang en/zh`) logging system for clear, informative feedback and easy debugging.
- **Clean, Modular Architecture**: Organized as a Cargo Workspace with a clear separation between the low-level, modular FFI layer (`rknn-ffi`), the core logic (`rkyolo-core`), and the application itself (`rkyolo-app`).

## 🚀 Getting Started

### Prerequisites

- A Rockchip development board (e.g., RK3588) with the official SDK and RKNN runtime (`librknnrt.so`) correctly installed.
- Rust toolchain installed on the device via [rustup](https://rustup.rs/).
- A C toolchain (`gcc`).
- **OpenCV development libraries** (e.g., `sudo apt install libopencv-dev`).
- A YOLO model converted to the `.rknn` format.

### Building the Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/cagedbird043/rkyolo.git
    cd rkyolo
    ```

2.  **Build the application in release mode for maximum performance:**
    _Note: The first build may take a significant amount of time due to the `opencv` crate._

    ```bash
    cargo build --release
    ```

    The final executable will be located at `target/release/rkyolo-app`.

## 💻 Usage Examples

The application is controlled via a universal `--source` (or `-s`) argument.

### 1. Real-time Camera Inference (Flagship Feature)

This command opens the default camera (`/dev/video0`), performs real-time inference on each frame using the custom tassel model, and displays the results. Press 'q' or ESC in the window to quit.

```bash
./target/release/rkyolo-app \
    -m ./MTDC-UAV.rknn \
    -s /dev/video0 \
    -l ./tassel_labels.txt \
    --conf-thresh 0.45 \
    --iou-thresh 0.50 \
    --lang zh
```

### 2. Video File Inference

Process a pre-recorded video file and display the results in real-time.

```bash
./target/release/rkyolo-app \
    -m model.rknn \
    -s my_video.mp4 \
    -l labels.txt
```

### 3. Single Image Inference

Run inference on a single image and save the output.

```bash
./target/release/rkyolo-app \
    -m model.rknn \
    -s image.jpg \
    -l labels.txt \
    -o result.jpg
```

### 4. Advanced Control (Logging & Performance)

Force the use of standard memory-copy mode and enable verbose debug logs.

```bash
./target/release/rkyolo-app \
    -s /dev/video0 \
    -m model.rknn \
    -l labels.txt \
    --disable-zero-copy \
    -vv
```

### All Options

You can view all available options and their default values by running:

```bash
./target/release/rkyolo-app --help
```

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
