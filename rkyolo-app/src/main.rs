use clap::Parser;
use image::{Rgb, RgbImage, imageops};
use rknn_ffi::RknnContext;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
mod drawing;
mod postprocess;

/// 一个使用 Rust 和 Rockchip NPU 进行 YOLO 模型推理的应用
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// .rknn 模型文件的路径
    #[arg(short, long, default_value = "assets/yolo11.rknn")]
    model: String,

    /// 要进行检测的输入图片路径
    #[arg(short, long, default_value = "assets/bus.jpg")]
    image: String,

    /// 包含类别名称的标签文件路径
    #[arg(short, long, default_value = "assets/coco_labels.txt")]
    labels: String,

    /// 输出结果图片的保存路径
    #[arg(short, long, default_value = "output.jpg")]
    output: String,

    /// 置信度阈值
    #[arg(long, default_value_t = 0.25)]
    conf_thresh: f32,

    /// NMS (非极大值抑制) 的 IoU 阈值
    #[arg(long, default_value_t = 0.45)]
    iou_thresh: f32,
}

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
fn preprocess_letterbox_quantize(
    image_path: &Path,
    target_width: u32,
    target_height: u32,
    zp: i32,
    scale: f32,
) -> Result<(Vec<i8>, LetterboxInfo), image::ImageError> {
    // 1. Letterbox 几何变换
    let img = image::open(image_path)?.to_rgb8();
    let (original_w, original_h) = img.dimensions();
    let scale_val =
        (target_width as f32 / original_w as f32).min(target_height as f32 / original_h as f32);
    let new_w = (original_w as f32 * scale_val) as u32;
    let new_h = (original_h as f32 * scale_val) as u32;
    let resized_img = imageops::resize(&img, new_w, new_h, imageops::FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(target_width, target_height, Rgb([114, 114, 114]));
    let pad_x = (target_width - new_w) / 2;
    let pad_y = (target_height - new_h) / 2;
    imageops::overlay(&mut canvas, &resized_img, pad_x.into(), pad_y.into());
    let info = LetterboxInfo {
        scale: scale_val,
        pad_x,
        pad_y,
    };
    let u8_data = canvas.into_raw();

    // 2. 【关键】手动归一化并量化
    let i8_data: Vec<i8> = u8_data
        .into_iter()
        .map(|val_u8| {
            // 正确的公式: q = round( (f_val / scale) + zp )
            // 其中 f_val 是归一化后的浮点值
            let f_val = val_u8 as f32 / 255.0; // <--- 之前遗漏的归一化步骤！
            let q_val = (f_val / scale + zp as f32).round() as i32;
            q_val.clamp(i8::MIN as i32, i8::MAX as i32) as i8
        })
        .collect();

    Ok((i8_data, info))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 解析命令行参数
    let args = Args::parse();
    println!("配置参数: {:?}", args);
    // --- 2. 使用解析出的路径 ---
    let model_path = Path::new(&args.model);
    let image_path = Path::new(&args.image);
    let labels_path = Path::new(&args.labels);

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
    let input_attrs = ctx.query_input_attrs().map_err(RknnError)?;
    let input_attr = &input_attrs[0];
    let input_zp = input_attr.zp;
    let input_scale = input_attr.scale;
    println!("模型输入量化参数: zp={}, scale={}", input_zp, input_scale);
    let (model_height, model_width, _) =
        if input_attr.fmt == rknn_ffi::raw::_rknn_tensor_format_RKNN_TENSOR_NHWC {
            (input_attr.dims[1], input_attr.dims[2], input_attr.dims[3])
        } else {
            (input_attr.dims[2], input_attr.dims[3], input_attr.dims[1])
        };

    // --- 5. 预处理输入图片并手动量化 ---
    println!("正在预处理图片并执行手动量化...");
    let (image_data_i8, letterbox_info) = preprocess_letterbox_quantize(
        image_path,
        model_width,
        model_height,
        input_zp,
        input_scale,
    )?;

    // --- 6. 设置输入并执行推理 ---
    println!("正在设置INT8输入数据并执行推理...");
    // 将 Vec<i8> 的数据安全地转换为 &[u8] 以传递给 FFI
    let image_data_u8: &[u8] = unsafe {
        std::slice::from_raw_parts(image_data_i8.as_ptr() as *const u8, image_data_i8.len())
    };
    // 将输入类型明确设置为 INT8
    ctx.set_input(
        0,
        rknn_ffi::raw::_rknn_tensor_type_RKNN_TENSOR_INT8,
        input_attr.fmt,
        image_data_u8,
    )
    .map_err(RknnError)?;
    ctx.run().map_err(RknnError)?;
    println!("推理完成!");

    // --- 7. 获取输出 ---
    println!("正在获取输出...");
    let outputs_obj = ctx.get_outputs().map_err(RknnError)?; // 重命名以示区分
    let output_attrs = ctx.query_output_attrs().map_err(RknnError)?;
    println!("成功获取 {} 个输出张量。", outputs_obj.all().len());

    // // --- 诊断信息：打印输出张量的维度 ---
    // println!("--- 输出张量属性诊断 ---");
    // for (i, attr) in output_attrs.iter().enumerate() {
    //     println!(
    //         "  - Attr {}: name={}, fmt={:?}, dims=[{}, {}, {}, {}]",
    //         i,
    //         std::str::from_utf8(&attr.name).unwrap_or(""), // C字符串转Rust字符串
    //         attr.fmt,
    //         attr.dims[0],
    //         attr.dims[1],
    //         attr.dims[2],
    //         attr.dims[3]
    //     );
    // }
    // println!("--------------------------");

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
        args.conf_thresh,
        args.iou_thresh,
        letterbox_info, // <-- 新增参数
    );

    // --- 10. 打印结果 ---
    println!("--- 检测结果 ({} 个) ---", detections.len());
    // 提前加载标签用于打印
    let labels_for_print = drawing::load_labels(labels_path)?;
    for det in &detections {
        let label = labels_for_print
            .get(det.class_id as usize)
            .map_or("unknown", |s| s.as_str());
        println!(
            "类别: {} ({}), 置信度: {:.4}, 框: {:?}", // 打印类别名和更精确的置信度
            det.class_id, label, det.confidence, det.bbox
        );
    }
    println!("--------------------------");

    // --- 11. 加载标签并绘制结果 ---
    println!("正在加载标签文件: {:?}", labels_path);
    let labels = drawing::load_labels(labels_path)?;

    println!("正在将检测结果绘制到图片上...");
    drawing::draw_results(&mut original_image, &detections, &labels);

    // --- 12. 保存结果图片到指定的输出路径 ---
    let output_path = Path::new(&args.output);
    println!("正在保存结果图片到: {:?}", output_path);
    original_image.save(output_path)?;
    println!("结果已保存！");
    Ok(())
}
