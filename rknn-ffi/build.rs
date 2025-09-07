use std::env;
use std::path::PathBuf;

fn main() {
    // 告诉 Cargo 需要链接 rknnrt 动态库。
    // Cargo 会在链接阶段自动添加 -lrknnrt 参数。
    println!("cargo:rustc-link-lib=dylib=rknnrt");

    // 告诉 Cargo 如果 C 头文件发生了变化，需要重新运行构建脚本。
    println!("cargo:rerun-if-changed=include/rknn_api.h");

    // --- bindgen 的配置 ---
    // 创建一个 bindgen 的 Builder
    let bindings = bindgen::Builder::default()
        // 指定要为其生成绑定的头文件
        .header("include/rknn_api.h")
        // 告诉 bindgen 在重新生成绑定时使 cargo 构建脚本无效
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // `__u128` 在某些目标上可能与 C 不兼容，通常建议禁用它
        .blocklist_item("__u128")
        // 生成 inline 函数的绑定
        .generate_inline_functions(true)
        // 完成配置并生成绑定
        .generate()
        // 如果生成失败，则 panic
        .expect("Unable to generate bindings");

    // --- 将生成的绑定写入文件 ---
    // 获取 Cargo 的输出目录路径
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // 将绑定写入到 `OUT_DIR/bindings.rs` 文件中
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
