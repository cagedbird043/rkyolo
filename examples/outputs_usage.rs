use rknn_ffi::context::RknnContext;

fn example_usage() -> Result<(), Box<dyn std::error::Error>> {
    // 假设你有一个已初始化的 RKNN 上下文
    let model_data = std::fs::read("model.rknn")?;
    let mut ctx = RknnContext::new(&model_data, 0, None)?;

    // ... 设置输入和运行推理 ...

    // 获取输出
    let outputs = ctx.get_outputs()?;

    // 方法 1: 使用显式的 getter 方法
    println!("输出数量: {}", outputs.len());
    println!("是否为空: {}", outputs.is_empty());

    // 访问特定索引的输出
    if let Some(first_output) = outputs.get(0) {
        println!("第一个输出的索引: {}", first_output.index);
        println!("第一个输出的大小: {}", first_output.size);
    }

    // 获取对 Vec 的引用（如果你需要调用 Vec 特有的方法）
    let outputs_vec = outputs.outputs();
    println!("Vec 长度: {}", outputs_vec.len());

    // 方法 2: 通过 Deref trait 自动解引用（推荐）
    // 由于实现了 Deref，可以直接调用 Vec 的方法
    println!("输出数量（通过 Deref）: {}", outputs.len());
    println!("是否为空（通过 Deref）: {}", outputs.is_empty());

    // 直接索引访问（通过 Deref）
    if !outputs.is_empty() {
        let first_output = &outputs[0];
        println!("第一个输出的索引: {}", first_output.index);
    }

    // 方法 3: 使用迭代器
    for (i, output) in outputs.iter().enumerate() {
        println!("输出 {}: 索引={}, 大小={}", i, output.index, output.size);
    }

    // 方法 4: 使用现有的 all() 方法获取切片
    let all_outputs = outputs.all();
    println!("通过 all() 获取的输出数量: {}", all_outputs.len());

    Ok(())
}
