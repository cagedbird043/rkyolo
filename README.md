# RKYOLO: A Robust, High-Performance YOLO Inference Framework in Rust for Rockchip NPU

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RKYOLO** is a modern, robust, and high-performance AI application foundation built from the ground up in Rust. It leverages Rockchip's NPU for hardware-accelerated inference of YOLO models, providing a superior alternative to the often brittle and hardcoded official C++ examples.

This project was born out of the need for a production-ready framework that offers Rust's memory safety guarantees, a modern build system with Cargo, and a flexible, model-agnostic architecture.

## 🌟 Core Features

- **Memory Safe**: Built with Rust to eliminate entire classes of bugs like segmentation faults at compile time.
- **High Performance**: Directly calls `librknnrt` via FFI to unleash the full power of the Rockchip NPU.
- **Model Agnostic**: Dynamically adapts to different YOLO model architectures (e.g., varying number of outputs, different class counts) without requiring code changes.
- **Robust Tooling**: Features a professional command-line interface (CLI) for easy configuration and use.
- **Clean Architecture**: Organized as a Cargo Workspace with a clear separation between the low-level FFI layer (`rknn-ffi`) and the application logic (`rkyolo-app`).

## 🚀 Getting Started

### Prerequisites

- A Rockchip development board (e.g., RK3588) with the official SDK and RKNN runtime (`librknnrt.so`) correctly installed.
- Rust toolchain installed on the device. You can install it via [rustup](https://rustup.rs/).
- A C toolchain (`gcc`) for compiling dependencies.
- A YOLO model converted to the `.rknn` format.

### Building the Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/cagedbird043/rkyolo.git
    cd rkyolo
    ```

2.  **Build the application:**
    ```bash
    cargo build --release
    ```
    The final executable will be located at `target/release/rkyolo-app`.

## 💻 Usage

The application is controlled via a command-line interface.

### Basic Inference

Run inference on a single image with default parameters:

```bash
./target/release/rkyolo-app \
    --model /path/to/your/model.rknn \
    --image /path/to/your/image.jpg \
    --labels /path/to/labels.txt \
    --output result.jpg
```

### All Options

You can view all available options by running:

```bash
./target/release/rkyolo-app --help
```

This will display a full list of arguments, including how to set confidence and NMS thresholds.

### Example: Running on the Corn Tassel Model

```bash
./target/release/rkyolo-app \
    --model ./MTDC-UAV.rknn \
    --image ./corn_field.jpg \
    --labels ./tassel_labels.txt \
    --output ./tassel_result.jpg \
    --conf-thresh 0.26 \
    --iou-thresh 0.57
```

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
