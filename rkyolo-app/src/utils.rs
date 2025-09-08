pub use log;
use log::{info, warn};
use rkyolo_core::{
    Detection, Lang, RknnContext, RknnError, get_type_string, rknn_ffi, rknn_ffi::RknnTensorMem,
    rknn_ffi::raw::rknn_tensor_attr,
};
use std::collections::HashMap;

pub enum ExecutionMode {
    ZeroCopy {
        input_mem: RknnTensorMem,
        input_attr: rknn_ffi::raw::rknn_tensor_attr,
    },
    Standard,
}

/// 尝试设置零拷贝执行模式。
/// 如果成功，返回配置好的 ZeroCopy 模式。
/// 如果失败，打印警告并返回 Standard 模式作为降级。
pub fn try_setup_zero_copy(ctx: &mut RknnContext, lang: &Lang) -> Result<ExecutionMode, RknnError> {
    // 1. 查询原生属性
    let native_input_attrs = match ctx.query_native_input_attrs() {
        Ok(attrs) => attrs,
        Err(e) => {
            match lang {
                Lang::En => warn!(
                    "Failed to query native input attributes (code: {}). Falling back to standard mode.",
                    e
                ),
                Lang::Zh => warn!("查询原生输入属性失败 (错误码: {})。将降级到标准模式。", e),
            }
            return Ok(ExecutionMode::Standard);
        }
    };
    // 我们只关心第一个输入
    let input_attr = if let Some(attr) = native_input_attrs.get(0) {
        attr.clone() // Clone attr so we can own it in ExecutionMode
    } else {
        match lang {
            Lang::En => warn!("No native input attributes found. Falling back to standard mode."),
            Lang::Zh => warn!("未找到原生输入属性。将降级到标准模式。"),
        }
        return Ok(ExecutionMode::Standard);
    };

    // 2. 创建 DMA 内存
    let input_mem = match ctx.create_mem(input_attr.size_with_stride) {
        Ok(mem) => mem,
        Err(e) => {
            match lang {
                Lang::En => warn!(
                    "Failed to create DMA memory (code: {}). Falling back to standard mode.",
                    e
                ),
                Lang::Zh => warn!("创建 DMA 内存失败 (错误码: {})。将降级到标准模式。", e),
            }
            return Ok(ExecutionMode::Standard);
        }
    };

    // 3. 绑定内存
    if let Err(e) = ctx.set_io_mem(&input_mem, &input_attr) {
        match lang {
            Lang::En => warn!(
                "Failed to set IO memory (code: {}). Falling back to standard mode.",
                e
            ),
            Lang::Zh => warn!("设置 IO 内存失败 (错误码: {})。将降级到标准模式。", e),
        }
        return Ok(ExecutionMode::Standard);
    }

    // 如果所有步骤都成功
    match lang {
        Lang::En => info!("Successfully initialized Zero-Copy mode."),
        Lang::Zh => info!("零拷贝模式初始化成功。"),
    }
    Ok(ExecutionMode::ZeroCopy {
        input_mem,
        input_attr,
    })
}

/// 【新增】辅助函数：记录检测结果摘要
pub fn log_detection_summary(detections: &[Detection], labels: &[String], lang: &Lang) {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for det in detections {
        *counts.entry(det.class_id).or_insert(0) += 1;
    }

    match lang {
        Lang::En => {
            info!(
                "Post-processing complete. Found {} final objects (after NMS).",
                detections.len()
            );
            if !detections.is_empty() {
                info!("Detection summary:");
                for (class_id, count) in &counts {
                    let label = labels
                        .get(*class_id as usize)
                        .map_or("unknown", |s| s.as_str());
                    info!("  - Class '{}': {} objects", label, count);
                }
            }
        }
        Lang::Zh => {
            info!(
                "后处理完成，共找到 {} 个最终目标（NMS后）。",
                detections.len()
            );
            if !detections.is_empty() {
                info!("检测结果摘要:");
                for (class_id, count) in &counts {
                    let label = labels
                        .get(*class_id as usize)
                        .map_or("未知类别", |s| s.as_str());
                    info!("  - 类别 '{}': {} 个目标", label, count);
                }
            }
        }
    }
}

/// 【新增】辅助函数：记录张量属性
pub fn log_tensor_attrs(attrs: &[rknn_tensor_attr], name_en: &str, name_zh: &str, lang: &Lang) {
    let (name, _dims_str, format_str, type_str, qnt_str) = match lang {
        Lang::En => (name_en, "Dims", "Format", "Type", "Quantization"),
        Lang::Zh => (name_zh, "维度", "格式", "类型", "量化"),
    };
    info!("- {}: {}", name, attrs.len());
    for (i, attr) in attrs.iter().enumerate() {
        let dims: Vec<String> = attr.dims[..attr.n_dims as usize]
            .iter()
            .map(|d| d.to_string())
            .collect();
        let type_name = get_type_string(attr.type_);
        info!(
            "  - [{}]: {} [{}], {}: {}, {}: {}, {}: ZP={}, Scale={:.4}",
            i,
            attr.name.iter().map(|&c| c as char).collect::<String>(),
            dims.join("x"),
            format_str,
            if attr.fmt == rknn_ffi::raw::_rknn_tensor_format_RKNN_TENSOR_NHWC {
                "NHWC"
            } else {
                "NCHW"
            },
            type_str,
            type_name,
            qnt_str,
            attr.zp,
            attr.scale
        );
    }
}
