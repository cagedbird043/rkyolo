use rknn_ffi::RknnContext;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

/// 自定义错误类型，用于封装来自 RKNN FFI 调用的 i32 错误码。
#[derive(Debug)]
struct RknnError(i32);

// 实现 Display trait，以便能够友好地打印错误信息。
impl fmt::Display for RknnError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RKNN API Error with code: {}", self.0)
    }
}

// 实现 Error trait，这样 RknnError 就可以被当作一个标准的错误类型来处理，
// 并且可以被放入 Box<dyn Error> 中。
impl Error for RknnError {}

// --- 您的 preprocess_image 函数 ---
/// 使用 `image` crate 预处理图像。
///
/// 1. 从路径加载图片。
/// 2. 转换为 RGB8 格式。
/// 3. 暴力缩放到目标尺寸。
/// 4. 将像素数据提取为扁平化的 Vec<u8>。
fn preprocess_image(
    image_path: &Path,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, image::ImageError> {
    let img = image::open(image_path)?;
    let rgb_img = img.to_rgb8();
    let resized = image::imageops::resize(
        &rgb_img,
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );
    Ok(resized.into_raw())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RKNN YOLO Rust Demo - 启动");

    // --- 1. 定义路径 ---
    let model_path = Path::new("./yolo11.rknn"); // 假设模型在当前目录
    let image_path = Path::new("bus.jpg"); // TODO: 准备一张名为 bus.jpg 的图片

    // --- 2. 加载模型 ---
    println!("正在加载模型: {:?}", model_path);
    let model_data = fs::read(model_path)?;

    // --- 3. 初始化 RKNN 上下文 ---
    println!("正在初始化 RKNN 上下文...");
    let mut ctx = RknnContext::new(&model_data, 0, None).map_err(RknnError)?;
    println!("RKNN 上下文初始化成功!");

    // --- 4. 查询模型输入属性 ---
    println!("正在查询模型输入属性...");
    let input_attrs = ctx.query_input_attrs().map_err(RknnError)?;
    let input_attr = &input_attrs[0]; // 假设只有一个输入

    // 从属性中获取模型期望的输入尺寸和格式
    // 注意：dims 数组的布局依赖于模型的 format (NHWC vs NCHW)
    let (model_height, model_width, model_channels) =
        if input_attr.fmt == rknn_ffi::raw::_rknn_tensor_format_RKNN_TENSOR_NHWC {
            (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3])
        } else {
            // NCHW
            (input_attr.dims[2], input_attr.dims[3], input_attr.dims[1])
        };
    println!(
        "模型输入尺寸: {}x{}x{}",
        model_width, model_height, model_channels
    );

    // --- 5. 预处理输入图片 ---
    println!("正在预处理图片: {:?}", image_path);
    let image_data = preprocess_image(image_path, model_width, model_height)?;

    // --- 6. 设置输入并执行推理 ---
    println!("正在设置输入数据并执行推理...");
    ctx.set_input(0, input_attr.type_, input_attr.fmt, &image_data)
        .map_err(RknnError)?;
    ctx.run().map_err(RknnError)?;
    println!("推理完成!");

    // --- 7. 获取输出 ---
    println!("正在获取输出...");
    let outputs = ctx.get_outputs().map_err(RknnError)?;
    println!("成功获取 {} 个输出张量。", outputs.all().len());

    // 暂时只打印每个输出的大小
    for (i, output) in outputs.all().iter().enumerate() {
        println!("  - 输出 {}: 大小 = {} 字节", i, output.size);
    }

    println!("Demo 运行结束。");
    Ok(())
}
