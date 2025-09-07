// 允许 bindgen 生成的不规范代码
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// 1. 声明模块，Rust 编译器会自动寻找同名的 .rs 文件
pub mod context;
pub mod io;
pub mod mem;
pub mod utils;

// 2. 将原始绑定放在它们自己的模块中，保持隔离
pub mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// 3. 重新导出 (re-export) 公共 API
pub use context::RknnContext;
pub use io::RknnOutputs;
pub use utils::get_type_string;

// 注意：我们暂时不导出 RknnTensorMem，因为它还不存在
