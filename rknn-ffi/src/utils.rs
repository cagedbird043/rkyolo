use crate::raw;
/// 将 RKNN 张量类型转换为可读的字符串
///
/// 这是对 C 头文件中 `get_type_string` 函数的 Rust 实现。
pub fn get_type_string(tensor_type: raw::rknn_tensor_type) -> &'static str {
    match tensor_type {
        raw::_rknn_tensor_type_RKNN_TENSOR_FLOAT32 => "FP32",
        raw::_rknn_tensor_type_RKNN_TENSOR_FLOAT16 => "FP16",
        raw::_rknn_tensor_type_RKNN_TENSOR_INT8 => "INT8",
        raw::_rknn_tensor_type_RKNN_TENSOR_UINT8 => "UINT8",
        raw::_rknn_tensor_type_RKNN_TENSOR_INT16 => "INT16",
        raw::_rknn_tensor_type_RKNN_TENSOR_UINT16 => "UINT16",
        raw::_rknn_tensor_type_RKNN_TENSOR_INT32 => "INT32",
        raw::_rknn_tensor_type_RKNN_TENSOR_UINT32 => "UINT32",
        raw::_rknn_tensor_type_RKNN_TENSOR_INT64 => "INT64",
        raw::_rknn_tensor_type_RKNN_TENSOR_BOOL => "BOOL",
        _ => "UNKNOWN",
    }
}
