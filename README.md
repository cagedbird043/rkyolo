# RKYOLO: A Robust, High-Performance YOLO Inference Framework in Rust for Rockchip NPU

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RKYOLO** is a modern, robust, and high-performance AI application foundation built from the ground up in Rust. It leverages Rockchip's NPU for hardware-accelerated inference of YOLO models, providing a superior alternative to the often brittle and hardcoded official C++ examples.

This project was born out of the need for a production-ready framework that offers Rust's memory safety guarantees, a modern build system with Cargo, and a flexible, model-agnostic architecture that achieves near-hardware-limit performance.

## 🌟 Core Features

- **Memory Safe**: Built with Rust to eliminate entire classes of bugs like segmentation faults and memory leaks at compile time, guaranteed by the compiler.
- **Blazing Fast Zero-Copy Performance**: Implements a zero-copy data path by default, writing pre-processed data directly into DMA buffers to minimize CPU overhead and I/O latency. Achieves significant speedups over standard memory-copy methods.
- **Intelligent Fallback**: Automatically and gracefully falls back to a standard memory-copy mode if zero-copy initialization fails, ensuring maximum performance where possible without sacrificing robustness.
- **Model Agnostic & Adaptive**: Dynamically adapts to different YOLO model architectures (e.g., varying number of outputs, different class counts) without requiring any code changes. It just works.
- **Professional Logging**: Features a multi-level (`-v`, `-vv`) and multi-lingual (`--lang en/zh`) logging system for clear, informative feedback and easy debugging.
- **Clean, Modular Architecture**: Organized as a Cargo Workspace with a clear separation between the low-level, modular FFI layer (`rknn-ffi`), the core logic (`rkyolo-core`), and the application itself (`rkyolo-app`).

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

2.  **Build the application in release mode for maximum performance:**
    ```bash
    cargo build --release
    ```
    The final executable will be located at `target/release/rkyolo-app`.

## 💻 Usage

The application is controlled via a powerful and user-friendly command-line interface.

### Basic Inference

Run inference on a single image. The application will automatically attempt to use the high-performance zero-copy mode.

```bash
./target/release/rkyolo-app \
    --model /path/to/your/model.rknn \
    --input /path/to/your/image.jpg \
    --labels /path/to/labels.txt \
    --output result.jpg
```

### Advanced Usage Examples

**1. Increase Log Verbosity and Switch to Chinese:**
Use the `-v` flag for INFO-level, `-vv` for DEBUG-level logs.

```bash
./target/release/rkyolo-app \
    -m model.rknn -i image.jpg -l labels.txt -o out.jpg \
    -v --lang zh
```

**2. Disable Zero-Copy for Debugging or Compatibility:**
Force the application to use the standard (safer but slower) memory-copy mode.

```bash
./target/release/rkyolo-app \
    -m model.rknn -i image.jpg -l labels.txt -o out.jpg \
    --disable-zero-copy
```

**3. Running on the Corn Tassel Model:**
A real-world example demonstrating the model-agnostic capabilities of RKYOLO.

```bash
./target/release/rkyolo-app \
    --model ./MTDC-UAV.rknn \
    --image ./corn_field.jpg \
    --labels ./tassel_labels.txt \
    --output ./tassel_result.jpg \
    --conf-thresh 0.26 \
    --iou-thresh 0.57
```

### All Options

You can view all available options and their default values by running:

```bash
./target/release/rkyolo-app --help
```

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
