use crate::raw;
use std::ops::Deref;
/// 一个安全封装，用于管理由 `rknn_outputs_get` 分配的输出张量。
///
/// 这个结构体“拥有” RKNN 输出缓冲区，并遵循 RAII 原则。
/// 当 `RknnOutputs` 实例离开作用域时，它会自动调用 `rknn_outputs_release`
/// 来释放底层内存，从而防止内存泄漏。
pub struct RknnOutputs {
    /// 持有从 C API 返回的 rknn_output 结构体数组。
    outputs: Vec<raw::rknn_output>,
    /// 持有分配这些输出的上下文句柄，以便在 Drop 时可以正确释放。
    ctx: raw::rknn_context,
}

impl RknnOutputs {
    /// 创建一个新的 RknnOutputs 实例。
    ///
    /// 这个方法仅供内部使用，用于在 RKNN FFI 内部创建输出封装。
    ///
    /// # Arguments
    /// * `outputs` - 从 C API 获取的输出张量数组
    /// * `ctx` - RKNN 上下文句柄，用于后续释放资源
    pub(crate) fn new(outputs: Vec<raw::rknn_output>, ctx: raw::rknn_context) -> Self {
        Self { outputs, ctx }
    }

    /// 返回一个对底层 `rknn_output` 结构体切片的引用。
    ///
    /// 这允许用户安全地只读访问所有输出张量的信息，
    /// 而不暴露 `Vec` 的可变性或所有权。
    pub fn all(&self) -> &[raw::rknn_output] {
        &self.outputs
    }

    /// 返回对底层 `Vec<rknn_output>` 的不可变引用。
    ///
    /// 这允许调用者访问 Vec 的所有只读方法，如 len(), iter(), get() 等。
    pub fn outputs(&self) -> &Vec<raw::rknn_output> {
        &self.outputs
    }

    /// 获取指定索引的输出张量。
    ///
    /// # Arguments
    /// * `index` - 输出张量的索引
    ///
    /// # Returns
    /// 如果索引有效，返回 `Some(&rknn_output)`，否则返回 `None`。
    pub fn get(&self, index: usize) -> Option<&raw::rknn_output> {
        self.outputs.get(index)
    }

    /// 返回输出张量的数量。
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// 检查是否为空。
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// 返回一个迭代器，遍历所有输出张量。
    pub fn iter(&self) -> std::slice::Iter<'_, raw::rknn_output> {
        self.outputs.iter()
    }
}

/// 实现 Deref trait，允许 RknnOutputs 自动解引用为 Vec<rknn_output>。
///
/// 这使得可以直接在 RknnOutputs 实例上调用 Vec 的所有只读方法，
/// 如 len(), iter(), get(), is_empty() 等，无需显式调用 getter 方法。
impl Deref for RknnOutputs {
    type Target = Vec<raw::rknn_output>;

    fn deref(&self) -> &Self::Target {
        &self.outputs
    }
}

impl Drop for RknnOutputs {
    fn drop(&mut self) {
        println!("Dropping RknnOutputs and calling rknn_outputs_release...");
        unsafe {
            raw::rknn_outputs_release(
                self.ctx,
                self.outputs.len() as u32,
                self.outputs.as_mut_ptr(),
            );
        }
    }
}
