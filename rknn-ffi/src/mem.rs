use crate::raw;

/// 一个安全的 Rust 封装，用于表示一个由 RKNN 管理的 DMA 内存缓冲区。
///
/// 它遵循 RAII 原则：当 `RknnTensorMem` 实例被创建时，它拥有一个由 `rknn_create_mem` 分配的缓冲区；
/// 当它离开作用域时，会通过 `Drop` trait 自动调用 `rknn_destroy_mem` 来释放资源。
pub struct RknnTensorMem {
    /// 存储由 `rknn_create_mem` 返回的指向内存描述符的裸指针。
    /// `pub(crate)` 使其在 crate 内部可见，但对外部 crate 隐藏。
    pub(crate) mem: *mut raw::rknn_tensor_mem,
    /// 持有创建此内存的上下文句柄，以便在 Drop 时可以正确释放。
    pub(crate) ctx: raw::rknn_context,
}

impl RknnTensorMem {
    /// 将底层的 DMA 缓冲区暴露为一个安全的可变字节切片。
    /// 上层代码可以通过这个切片安全地读写输入数据。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            // 我们信任 rknn_create_mem 返回的 mem 指针和其中的 size 是有效的
            let raw_mem = &*self.mem;
            std::slice::from_raw_parts_mut(raw_mem.virt_addr as *mut u8, raw_mem.size as usize)
        }
    }

    /// 提供对底层 `rknn_tensor_mem` 结构体的安全只读访问。
    pub fn as_raw(&self) -> &raw::rknn_tensor_mem {
        unsafe { &*self.mem }
    }
}

impl Drop for RknnTensorMem {
    fn drop(&mut self) {
        println!("Dropping RknnTensorMem and calling rknn_destroy_mem...");
        unsafe {
            raw::rknn_destroy_mem(self.ctx, self.mem);
        }
    }
}
