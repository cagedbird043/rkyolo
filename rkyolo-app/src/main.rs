use rknn_ffi::RknnContext; // 导入我们创建的安全封装
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RKNN YOLO Rust Demo - 启动");

    // --- 1. 定义模型路径 ---
    // TODO: 请将此路径替换为您实际的模型文件路径
    let model_path = Path::new("./yolo11.rknn");
    if !model_path.exists() {
        eprintln!("错误: 模型文件未找到: {:?}", model_path);
        // 返回一个自定义的错误信息
        return Err(format!("Model file not found: {}", model_path.display()).into());
    }

    // --- 2. 加载模型文件 ---
    println!("正在加载模型: {:?}", model_path);
    let model_data = fs::read(model_path)?;

    // --- 3. 初始化 RKNN 上下文 ---
    // 使用 rknn-ffi crate 中的 RknnContext::new 方法
    println!("正在初始化 RKNN 上下文...");
    let ctx = match RknnContext::new(&model_data, 0, None) {
        Ok(ctx) => {
            println!("RKNN 上下文初始化成功!");
            ctx
        }
        Err(e) => {
            eprintln!("错误: RKNN 上下文初始化失败，错误码: {}", e);
            return Err(format!("RKNN init failed with code {}", e).into());
        }
    };

    // --- 4. 查询并打印模型信息 ---
    println!("正在查询模型输入/输出数量...");
    match ctx.query_io_num() {
        Ok(io_num) => {
            println!("查询成功:");
            println!("  - 模型输入数量 (n_input): {}", io_num.n_input);
            println!("  - 模型输出数量 (n_output): {}", io_num.n_output);
        }
        Err(e) => {
            eprintln!("错误: 查询 IO 数量失败，错误码: {}", e);
            return Err(format!("Query IO num failed with code {}", e).into());
        }
    }

    println!("Demo 运行结束。当 main 函数返回时，RknnContext 将被自动销毁。");
    // 当 `ctx` 离开作用域时，它的 Drop 实现会被调用，自动执行 rknn_destroy
    Ok(())
}
