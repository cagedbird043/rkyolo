#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// 使用 include! 宏，在编译时将 `OUT_DIR/bindings.rs` 的内容直接“粘贴”到这里。
// 这使得生成的绑定成为我们 crate 的一部分。
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
