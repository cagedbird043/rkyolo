// 允许 bindgen 生成的不规范代码
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// 包含由 build.rs 在编译时生成的原始、不安全的 FFI 绑定
// 这些绑定被放在一个名为 `raw` 的模块中，以便与我们的安全封装区分开
pub mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// ---------------- 安全封装层 ----------------

/// 一个安全的 Rust 封装，用于表示一个 RKNN 模型上下文。
///
/// 它遵循 RAII 原则：当 `RknnContext` 实例被创建时，会初始化一个 RKNN 上下文；
/// 当它离开作用域时，会通过 `Drop` trait 自动调用 `rknn_destroy` 来释放资源。
pub struct RknnContext {
    /// 存储由 `rknn_init` 返回的原始、不透明的 C 指针/句柄。
    /// 我们将其设为私有，以防止外部代码直接操作这个不安全的句柄。
    ctx: raw::rknn_context,
}

impl RknnContext {
    /// 创建一个新的 RKNN 上下文实例。
    ///
    /// 这个方法会加载一个 RKNN 模型并初始化上下文。它是底层 `rknn_init` 函数的安全封装。
    ///
    /// # Arguments
    ///
    /// * `model_data` - 包含 RKNN 模型数据的字节切片
    /// * `flag` - 初始化标志，控制模型加载行为（如是否使用 NPU 等）
    /// * `extend` - 可选的扩展配置，用于高级初始化选项
    ///
    /// # Returns
    ///
    /// 成功时返回 `Ok(RknnContext)`，包含已初始化的上下文。
    /// 失败时返回 `Err(error_code)`，其中 error_code 是 RKNN 库返回的错误码。
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::fs;
    ///
    /// // 从文件加载模型
    /// let model_data = fs::read("model.rknn").expect("Failed to read model file");
    ///
    /// // 创建上下文（使用默认标志）
    /// let ctx = RknnContext::new(&model_data, 0, None)?;
    /// ```
    ///
    /// # Safety
    ///
    /// 虽然这个方法本身是安全的，但它内部调用了 unsafe FFI 函数。
    /// 确保传入的 model_data 是有效的 RKNN 模型格式。
    pub fn new(
        model_data: &[u8],
        flag: u32,
        extend: Option<&mut raw::rknn_init_extend>,
    ) -> Result<Self, i32> {
        // 1. 声明一个未初始化的 rknn_context 变量
        //    rknn_init 函数期望一个指向 rknn_context 的指针，它会负责填充这个变量
        let mut ctx: raw::rknn_context = 0;

        // 2. 将 model_data 切片转换为 C API 需要的 void* 和 size
        let model_ptr = model_data.as_ptr() as *mut std::ffi::c_void;
        let model_size = model_data.len() as u32;

        // 3. 处理可选的 extend 参数，将其转换为 C API 需要的裸指针
        //    如果 extend 是 Some，获取其可变裸指针；如果是 None，使用 null_mut()
        let extend_ptr = match extend {
            Some(ext) => ext as *mut raw::rknn_init_extend,
            None => std::ptr::null_mut(),
        };

        // 4. 调用底层的 FFI 函数，必须在 unsafe 块中完成
        let ret = unsafe {
            raw::rknn_init(
                &mut ctx, // 传递 ctx 的可变指针
                model_ptr, model_size, flag, extend_ptr,
            )
        };

        // 5. 检查 rknn_init 的返回值
        //    RKNN_SUCC (通常是 0) 表示成功
        if ret == raw::RKNN_SUCC as i32 {
            // 成功：创建一个 RknnContext 实例并包裹在 Ok 中返回
            // 此时 ctx 已经被 rknn_init 成功初始化了
            Ok(RknnContext { ctx })
        } else {
            // 失败：将 C API 返回的错误码包裹在 Err 中返回
            Err(ret)
        }
    }

    /// 查询模型的输入和输出张量的数量。
    ///
    /// 这是一个对 `rknn_query` 使用 `RKNN_QUERY_IN_OUT_NUM` 命令的安全封装。
    ///
    /// # Returns
    ///
    /// 成功时返回 `Ok(rknn_input_output_num)`，一个包含 n_input 和 n_output 字段的结构体。
    /// 失败时返回 `Err(error_code)`，其中 error_code 是 RKNN 库返回的错误码。
    ///
    /// # Examples
    ///
    /// ```rust
    /// let ctx = RknnContext::new(&model_data, 0, None)?;
    /// let io_num = ctx.query_io_num()?;
    /// println!("输入张量数量: {}, 输出张量数量: {}", io_num.n_input, io_num.n_output);
    /// ```
    pub fn query_io_num(&self) -> Result<raw::rknn_input_output_num, i32> {
        // 1. 创建一个 rknn_input_output_num 结构体实例，用零值初始化
        //    使用 std::mem::zeroed() 在 unsafe 上下文中创建全零结构体
        let mut io_num: raw::rknn_input_output_num = unsafe { std::mem::zeroed() };

        // 2. 调用底层的 FFI 函数 rknn_query
        let ret = unsafe {
            raw::rknn_query(
                self.ctx,                                                 // 当前上下文句柄
                raw::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,               // 查询命令
                &mut io_num as *mut _ as *mut std::ffi::c_void,           // 将结构体指针转为 void*
                std::mem::size_of::<raw::rknn_input_output_num>() as u32, // 结构体的大小
            )
        };

        // 3. 检查返回值并返回结果
        if ret == raw::RKNN_SUCC as i32 {
            // 成功：返回填充了信息的 io_num 结构体
            Ok(io_num)
        } else {
            // 失败：返回错误码
            Err(ret)
        }
    }

    /// 查询所有输入张量的属性。
    ///
    /// # Returns
    /// 成功时返回 `Ok(Vec<rknn_tensor_attr>)`，失败时返回 `Err(error_code)`。
    pub fn query_input_attrs(&self) -> Result<Vec<raw::rknn_tensor_attr>, i32> {
        // 1. 先查询输入/输出张量的数量
        let io_num = self.query_io_num()?;
        let num_inputs = io_num.n_input as usize;

        // 2. 创建一个 Vec 用于存放结果，并预分配容量
        let mut input_attrs = Vec::with_capacity(num_inputs);

        // 3. 循环查询每个输入张量的属性
        for i in 0..num_inputs {
            let mut attr: raw::rknn_tensor_attr = unsafe { std::mem::zeroed() };
            // 关键：设置要查询的张量的索引
            attr.index = i as u32;

            let ret = unsafe {
                raw::rknn_query(
                    self.ctx,
                    raw::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
                    &mut attr as *mut _ as *mut std::ffi::c_void,
                    std::mem::size_of::<raw::rknn_tensor_attr>() as u32,
                )
            };

            if ret != raw::RKNN_SUCC as i32 {
                // 如果任何一次查询失败，立即返回错误
                return Err(ret);
            }
            input_attrs.push(attr);
        }

        Ok(input_attrs)
    }

    /// 查询所有输出张量的属性。
    ///
    /// # Returns
    /// 成功时返回 `Ok(Vec<rknn_tensor_attr>)`，失败时返回 `Err(error_code)`。
    pub fn query_output_attrs(&self) -> Result<Vec<raw::rknn_tensor_attr>, i32> {
        // 实现逻辑与 query_input_attrs 几乎完全相同，只是查询命令和循环次数不同
        let io_num = self.query_io_num()?;
        let num_outputs = io_num.n_output as usize;

        let mut output_attrs = Vec::with_capacity(num_outputs);

        for i in 0..num_outputs {
            let mut attr: raw::rknn_tensor_attr = unsafe { std::mem::zeroed() };
            attr.index = i as u32;

            let ret = unsafe {
                raw::rknn_query(
                    self.ctx,
                    raw::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR, // 使用查询输出的命令
                    &mut attr as *mut _ as *mut std::ffi::c_void,
                    std::mem::size_of::<raw::rknn_tensor_attr>() as u32,
                )
            };

            if ret != raw::RKNN_SUCC as i32 {
                return Err(ret);
            }
            output_attrs.push(attr);
        }

        Ok(output_attrs)
    }
}

/// 为 RknnContext 实现 Drop trait，以确保资源自动释放。
///
/// 当 `RknnContext` 实例离开作用域时，会自动调用 `rknn_destroy` 释放底层资源。
/// 这遵循 Rust 的 RAII (Resource Acquisition Is Initialization) 原则。
impl Drop for RknnContext {
    /// 当 `RknnContext` 实例离开作用域时，此方法会被自动调用。
    ///
    /// 这个方法会调用底层的 `rknn_destroy` 函数来释放 RKNN 上下文资源。
    fn drop(&mut self) {
        // 在这里调用底层的 C API 来销毁 RKNN 上下文
        // 这是一个 FFI 调用，因此必须在 unsafe 块中进行
        // 打印调试信息来确认 drop 确实被调用了
        println!("Dropping RknnContext and calling rknn_destroy...");
        unsafe {
            raw::rknn_destroy(self.ctx);
        }
    }
}

// ================================================================================
// 线程安全性注释
// ================================================================================
//
// RknnContext 默认不实现 Send 或 Sync trait，这是因为：
// 1. 它包含的 raw::rknn_context 可能不是线程安全的
// 2. 底层 C 库的线程安全性需要单独验证
// 3. 这是一个安全的默认设置，防止意外的并发访问
//
// 如果确认底层库是线程安全的，可以手动实现这些 trait
