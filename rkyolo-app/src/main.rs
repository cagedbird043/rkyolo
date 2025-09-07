use image::{Rgb, RgbImage, imageops};
use rknn_ffi::RknnContext;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path; // 确保 RgbImage 和 Rgb 被导入
mod drawing;
mod postprocess;

/// 存储 Letterbox 预处理的相关信息
#[derive(Debug, Clone, Copy)]
struct LetterboxInfo {
    scale: f32,
    pad_x: u32,
    pad_y: u32,
}
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

/// 使用 `image` crate 和 Letterboxing 技术预处理图像。
fn preprocess_letterbox(
    image_path: &Path,
    target_width: u32,
    target_height: u32,
) -> Result<(Vec<u8>, LetterboxInfo), image::ImageError> {
    let img = image::open(image_path)?.to_rgb8();
    let (original_w, original_h) = img.dimensions();

    // 1. 计算缩放比例
    let scale_w = target_width as f32 / original_w as f32;
    let scale_h = target_height as f32 / original_h as f32;
    let scale = scale_w.min(scale_h);

    // 2. 计算缩放后尺寸
    let new_w = (original_w as f32 * scale) as u32;
    let new_h = (original_h as f32 * scale) as u32;

    // 缩放图像
    let resized_img = imageops::resize(&img, new_w, new_h, imageops::FilterType::Lanczos3);

    // 3. 创建灰色画布
    let mut canvas = RgbImage::from_pixel(target_width, target_height, Rgb([114, 114, 114]));

    // 4. 计算填充
    let pad_x = (target_width - new_w) / 2;
    let pad_y = (target_height - new_h) / 2;

    // 5. 粘贴图像
    imageops::overlay(&mut canvas, &resized_img, pad_x.into(), pad_y.into());

    // 6. 返回数据和信息
    let info = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };
    Ok((canvas.into_raw(), info))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RKNN YOLO Rust Demo - 启动");

    // --- 1. 定义路径 ---
    let model_path = Path::new("./yolo11.rknn");
    let image_path = Path::new("bus.jpg");
    let labels_path = Path::new("coco_labels.txt");

    // 加载原始图片用于绘图
    let mut original_image = image::open(image_path)?.to_rgb8();

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
    // 调用新的 letterbox 函数
    let (image_data, letterbox_info) = preprocess_letterbox(image_path, model_width, model_height)?;

    // --- 6. 设置输入并执行推理 ---
    println!("正在设置输入数据并执行推理...");
    ctx.set_input(0, input_attr.type_, input_attr.fmt, &image_data)
        .map_err(RknnError)?;
    ctx.run().map_err(RknnError)?;
    println!("推理完成!");

    // --- 7. 获取输出 ---
    println!("正在获取输出...");
    let outputs_obj = ctx.get_outputs().map_err(RknnError)?; // 重命名以示区分
    let output_attrs = ctx.query_output_attrs().map_err(RknnError)?;
    println!("成功获取 {} 个输出张量。", outputs_obj.all().len());

    // --- 诊断信息：打印输出张量的维度 ---
    println!("--- 输出张量属性诊断 ---");
    for (i, attr) in output_attrs.iter().enumerate() {
        println!(
            "  - Attr {}: name={}, fmt={:?}, dims=[{}, {}, {}, {}]",
            i,
            std::str::from_utf8(&attr.name).unwrap_or(""), // C字符串转Rust字符串
            attr.fmt,
            attr.dims[0],
            attr.dims[1],
            attr.dims[2],
            attr.dims[3]
        );
    }
    println!("--------------------------");

    // --- 8. 准备后处理函数的输入 ---
    // 从 RknnOutputs 对象中提取出原始的字节切片
    let outputs_data: Vec<&[u8]> = outputs_obj
        .all()
        .iter()
        .map(|o| unsafe { std::slice::from_raw_parts(o.buf as *const u8, o.size as usize) })
        .collect();

    // --- 9. 执行后处理 ---
    let detections = postprocess::post_process_i8(
        &outputs_data,
        &output_attrs,
        0.25,
        0.45,
        letterbox_info, // <-- 新增参数
    );

    // --- 10. 打印结果 ---
    for det in &detections {
        println!(
            "类别: {}, 置信度: {:.2}, 框: {:?}",
            det.class_id, det.confidence, det.bbox
        );
    }

    // --- 11. 加载标签并绘制结果 ---
    println!("正在加载标签文件: {:?}", labels_path);
    let labels = drawing::load_labels(labels_path)?;

    println!("正在将检测结果绘制到图片上...");
    drawing::draw_results(&mut original_image, &detections, &labels);

    // --- 12. 保存结果图片 ---
    let output_path = Path::new("output.jpg");
    println!("正在保存结果图片到: {:?}", output_path);
    original_image.save(output_path)?;
    println!("结果已保存！");

    println!("Demo 运行结束。");
    Ok(())
}
