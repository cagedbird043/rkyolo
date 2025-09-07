use crate::io::RknnOutputs;
use crate::raw;

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
    ///
    /// 成功时返回 `Ok(Vec<rknn_tensor_attr>)`，包含所有输入张量的属性信息。
    /// 失败时返回 `Err(error_code)`，其中 error_code 是 RKNN 库返回的错误码。
    ///
    /// # Examples
    ///
    /// ```rust
    /// let ctx = RknnContext::new(&model_data, 0, None)?;
    /// let input_attrs = ctx.query_input_attrs()?;
    /// for attr in &input_attrs {
    ///     println!("输入张量 {}: {:?}", attr.index, attr.dims);
    /// }
    /// ```
    pub fn query_input_attrs(&self) -> Result<Vec<raw::rknn_tensor_attr>, i32> {
        let io_num = self.query_io_num()?;
        self.query_tensor_attrs(
            io_num.n_input as usize,
            raw::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
        )
    }

    /// 查询所有输出张量的属性。
    ///
    /// # Returns
    ///
    /// 成功时返回 `Ok(Vec<rknn_tensor_attr>)`，包含所有输出张量的属性信息。
    /// 失败时返回 `Err(error_code)`，其中 error_code 是 RKNN 库返回的错误码。
    ///
    /// # Examples
    ///
    /// ```rust
    /// let ctx = RknnContext::new(&model_data, 0, None)?;
    /// let output_attrs = ctx.query_output_attrs()?;
    /// for attr in &output_attrs {
    ///     println!("输出张量 {}: {:?}", attr.index, attr.dims);
    /// }
    /// ```
    pub fn query_output_attrs(&self) -> Result<Vec<raw::rknn_tensor_attr>, i32> {
        let io_num = self.query_io_num()?;
        self.query_tensor_attrs(
            io_num.n_output as usize,
            raw::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
        )
    }

    /// 为模型设置单个输入张量的数据（标准、非零拷贝方式）。
    ///
    /// 这个方法是对 `rknn_inputs_set` 的一个简化封装，专门用于只有一个输入的模型。
    /// 数据将被拷贝到 NPU 的内部缓冲区。
    ///
    /// # Arguments
    /// * `index` - 输入张量的索引，通常为 0。
    /// * `input_type` - 输入数据的张量类型，例如 `raw::rknn_tensor_type_RKNN_TENSOR_UINT8`。
    /// * `input_format` - 输入数据的张量格式，例如 `raw::_rknn_tensor_format_RKNN_TENSOR_NHWC`。
    /// * `data` - 包含输入数据的字节切片。
    ///
    /// # Returns
    /// 成功时返回 `Ok(())`，失败时返回 `Err(error_code)`。
    pub fn set_input(
        &mut self,
        index: u32,
        input_type: raw::rknn_tensor_type,
        input_format: raw::rknn_tensor_format,
        data: &[u8],
    ) -> Result<(), i32> {
        // 1. 构建一个 rknn_input 结构体实例
        let rknn_input = raw::rknn_input {
            index,
            buf: data.as_ptr() as *mut std::ffi::c_void,
            size: data.len() as u32,
            pass_through: 0, // 0 表示需要驱动进行类型和格式转换
            type_: input_type,
            fmt: input_format,
        };

        // 2. 将单个结构体放入一个数组中，因为 C API 需要一个数组指针
        let inputs = [rknn_input];

        // 3. 调用底层的 FFI 函数
        let ret = unsafe {
            raw::rknn_inputs_set(
                self.ctx,
                inputs.len() as u32, // n_inputs, 这里是 1
                inputs.as_ptr() as *mut raw::rknn_input,
            )
        };

        // 4. 检查结果
        if ret == raw::RKNN_SUCC as i32 {
            Ok(())
        } else {
            Err(ret)
        }
    }

    /// 获取模型的输出结果。
    ///
    /// 此方法调用 `rknn_outputs_get` 来获取由 NPU 计算出的所有输出张量。
    ///
    /// # Returns
    /// 成功时返回 `Ok(RknnOutputs)`，这是一个管理输出缓冲区的安全封装。
    /// 失败时返回 `Err(error_code)`。
    pub fn get_outputs(&mut self) -> Result<RknnOutputs, i32> {
        // 1. 查询输出张量的数量，以确定需要准备多大的 outputs 数组
        let io_num = self.query_io_num()?;
        let num_outputs = io_num.n_output as usize;

        // 2. 创建一个 Vec<rknn_output>。这里需要特别注意：
        //    我们需要一个由未初始化内存构成的 Vec，因为 C API 会负责填充它。
        let mut outputs: Vec<raw::rknn_output> = Vec::with_capacity(num_outputs);
        unsafe {
            outputs.set_len(num_outputs);
        }

        // 我们需要告诉 get 函数，我们希望它为我们分配内存
        for (i, output) in outputs.iter_mut().enumerate() {
            output.index = i as u32; // <-- 修正：明确指定要获取哪个输出

            output.is_prealloc = 0; // 0 表示 false
                                    // 如果模型是量化模型，我们通常希望得到浮点数结果用于后处理
            output.want_float = 0; // 1 表示 true
        }

        // 3. 调用底层的 FFI 函数
        let ret = unsafe {
            raw::rknn_outputs_get(
                self.ctx,
                num_outputs as u32,
                outputs.as_mut_ptr(),
                std::ptr::null_mut(), // extend 参数未使用
            )
        };

        // 4. 检查结果
        if ret == raw::RKNN_SUCC as i32 {
            // 成功：创建一个 RknnOutputs 实例来管理这些输出
            Ok(RknnOutputs::new(outputs, self.ctx))
        } else {
            // 失败：返回错误码
            Err(ret)
        }
    }

    pub fn run(&mut self) -> Result<(), i32> {
        let ret = unsafe { raw::rknn_run(self.ctx, std::ptr::null_mut()) };

        if ret == raw::RKNN_SUCC as i32 {
            Ok(())
        } else {
            Err(ret)
        }
    }

    /// 通用的张量属性查询方法（私有）。
    ///
    /// # Arguments
    ///
    /// * `count` - 要查询的张量数量
    /// * `query_cmd` - RKNN 查询命令（输入或输出）
    ///
    /// # Returns
    ///
    /// 成功时返回 `Ok(Vec<rknn_tensor_attr>)`，失败时返回 `Err(error_code)`。
    fn query_tensor_attrs(
        &self,
        count: usize,
        query_cmd: raw::_rknn_query_cmd,
    ) -> Result<Vec<raw::rknn_tensor_attr>, i32> {
        // 1. 创建一个 Vec 用于存放结果，并预分配容量
        let mut attrs = Vec::with_capacity(count);

        // 2. 循环查询每个张量的属性
        for i in 0..count {
            let mut attr: raw::rknn_tensor_attr = unsafe { std::mem::zeroed() };
            // 关键：设置要查询的张量的索引
            attr.index = i as u32;

            let ret = unsafe {
                raw::rknn_query(
                    self.ctx,
                    query_cmd,
                    &mut attr as *mut _ as *mut std::ffi::c_void,
                    std::mem::size_of::<raw::rknn_tensor_attr>() as u32,
                )
            };

            if ret != raw::RKNN_SUCC as i32 {
                // 如果任何一次查询失败，立即返回错误
                return Err(ret);
            }
            attrs.push(attr);
        }

        Ok(attrs)
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
