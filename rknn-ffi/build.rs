fn main() {
    // 告诉 Cargo 需要链接 Rockchip 的 rknnrt 动态库。
    // Cargo 会在链接阶段自动添加 -lrknnrt 参数。
    println!("cargo:rustc-link-lib=dylib=rknnrt");

    // 告诉 Cargo 如果 build.rs 文件发生变化，需要重新运行构建脚本。
    println!("cargo:rerun-if-changed=build.rs");

    // 提示：我们很快会在这里添加 bindgen 的逻辑，
    // 用来从 rknn_api.h 自动生成 Rust 绑定代码。
    // 目前，我们先确保链接配置是正确的。
}