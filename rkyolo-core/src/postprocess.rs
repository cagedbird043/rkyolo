//! YOLO 模型后处理模块 (INT8 版本)

use crate::{Lang, LetterboxInfo}; // <--- 引入 Lang
use log::{debug, info}; // <--- 引入 log 宏
use rknn_ffi::raw::rknn_tensor_attr;

// --- 数据结构 ---
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub class_id: i32,
}

/// 用于临时存储已识别的张量信息
#[derive(Debug)] // <--- 添加 Debug trait
enum TensorInfo {
    Box { h: u32, w: u32, index: usize },
    Score { h: u32, w: u32, index: usize },
}

// --- 辅助函数 ---
#[inline]
fn dequantize(q_val: i8, zero_point: i32, scale: f32) -> f32 {
    ((q_val as i32 - zero_point) as f32) * scale
}

/// 计算两个边界框的交并比 (IoU)
fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    let union_area = box1_area + box2_area - intersection_area;

    if union_area > 0.0 {
        intersection_area / union_area
    } else {
        0.0
    }
}

// --- 核心解码逻辑 ---

/// 解码单个模型输出分支。
fn decode_branch(
    box_data: &[i8],
    score_data: &[i8],
    box_attr: &rknn_tensor_attr,
    score_attr: &rknn_tensor_attr,
    stride: u32,
    conf_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    let grid_h = box_attr.dims[2] as usize;
    let grid_w = box_attr.dims[3] as usize;
    let grid_len = grid_h * grid_w;

    let class_num = score_attr.dims[1] as usize;

    let dfl_len = box_attr.dims[1] as usize / 4;

    for y in 0..grid_h {
        for x in 0..grid_w {
            let grid_cell_offset = y * grid_w + x;

            // 这个循环现在可以正确地遍历所有80个类别了
            for c in 0..class_num {
                let score_offset = c * grid_len + grid_cell_offset;
                let score_q = score_data[score_offset];
                let score = dequantize(score_q, score_attr.zp, score_attr.scale);

                if score > conf_threshold {
                    let mut box_dist = [0.0f32; 4];
                    for i in 0..4 {
                        let mut acc_sum = 0.0;
                        let mut exp_sum = 0.0;
                        let mut exp_values = [0.0f32; 16];

                        for j in 0..dfl_len {
                            let offset = (i * dfl_len + j) * grid_len + grid_cell_offset;
                            let box_q = box_data[offset];
                            let box_f = dequantize(box_q, box_attr.zp, box_attr.scale);
                            exp_values[j] = box_f.exp();
                            exp_sum += exp_values[j];
                        }

                        for j in 0..dfl_len {
                            acc_sum += exp_values[j] / exp_sum * (j as f32);
                        }
                        box_dist[i] = acc_sum;
                    }

                    let x_center_grid = x as f32 + 0.5;
                    let y_center_grid = y as f32 + 0.5;

                    let left = x_center_grid - box_dist[0];
                    let top = y_center_grid - box_dist[1];
                    let right = x_center_grid + box_dist[2];
                    let bottom = y_center_grid + box_dist[3];

                    let x1 = left * stride as f32;
                    let y1 = top * stride as f32;
                    let x2 = right * stride as f32;
                    let y2 = bottom * stride as f32;

                    detections.push(Detection {
                        bbox: BoundingBox { x1, y1, x2, y2 },
                        confidence: score,
                        class_id: c as i32,
                    });
                }
            }
        }
    }
    detections
}

/// YOLO 模型后处理的主函数 (INT8 版本)
pub fn post_process_i8(
    outputs_data: &[&[u8]],
    output_attrs: &[rknn_tensor_attr],
    conf_threshold: f32,
    nms_threshold: f32,
    letterbox: LetterboxInfo,
    lang: Lang, // <--- 新增 lang 参数
) -> Vec<Detection> {
    let model_in_h = 640;
    debug!(
        "Starting post-processing for {} outputs.",
        output_attrs.len()
    );

    // 1. 动态识别与分类：将所有输出张量识别为 Box 或 Score 类型
    let mut identified_tensors = Vec::new();
    for (i, attr) in output_attrs.iter().enumerate() {
        if attr.n_dims == 4 {
            // 我们只处理标准的4维输出
            let c = attr.dims[1];
            let h = attr.dims[2];
            let w = attr.dims[3];

            // 使用一个启发式规则来区分box和score张量
            // box张量的通道数通常是固定的（例如64），而score张量的通道数等于类别数
            if c == 64 {
                // 这个值通常与DFL的长度有关，对于yolo11是64
                identified_tensors.push(TensorInfo::Box { h, w, index: i });
            } else {
                identified_tensors.push(TensorInfo::Score { h, w, index: i });
            }
        }
    }
    debug!("Identified tensors: {:?}", identified_tensors);

    let mut all_detections = Vec::new();

    // 2. 动态配对与解码：遍历所有识别出的Box张量，并为它们寻找匹配的Score张量
    for tensor_info in &identified_tensors {
        if let &TensorInfo::Box {
            h,
            w,
            index: box_idx,
        } = tensor_info
        {
            // 尝试寻找一个具有相同 H 和 W 维度的 Score 张量
            let paired_score_info = identified_tensors.iter().find(|&t| {
                if let &TensorInfo::Score { h: sh, w: sw, .. } = t {
                    sh == h && sw == w
                } else {
                    false
                }
            });

            // 3. 如果找到了配对，则进行解码
            if let Some(&TensorInfo::Score {
                index: score_idx, ..
            }) = paired_score_info
            {
                let box_attr = &output_attrs[box_idx];
                let score_attr = &output_attrs[score_idx];
                debug!(
                    "Decoding branch: box_idx={}, score_idx={}, grid=({}x{})",
                    box_idx, score_idx, h, w
                );

                let box_data_u8 = outputs_data[box_idx];
                let score_data_u8 = outputs_data[score_idx];

                let box_data: &[i8] = unsafe {
                    std::slice::from_raw_parts(box_data_u8.as_ptr() as *const i8, box_data_u8.len())
                };
                let score_data: &[i8] = unsafe {
                    std::slice::from_raw_parts(
                        score_data_u8.as_ptr() as *const i8,
                        score_data_u8.len(),
                    )
                };

                let stride = model_in_h / box_attr.dims[2];

                let branch_detections = decode_branch(
                    box_data,
                    score_data,
                    box_attr,
                    score_attr,
                    stride,
                    conf_threshold,
                );
                all_detections.extend(branch_detections);
            }
        }
    }
    debug!("Found {} raw detections before NMS.", all_detections.len());

    // NMS 和坐标变换逻辑保持不变
    all_detections.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut nms_detections = Vec::new();
    while !all_detections.is_empty() {
        let best_det = all_detections.remove(0);
        nms_detections.push(best_det.clone());
        all_detections.retain(|det| {
            if det.class_id == best_det.class_id {
                calculate_iou(&best_det.bbox, &det.bbox) < nms_threshold
            } else {
                true
            }
        });
    }

    let mut corrected_detections = Vec::with_capacity(nms_detections.len());
    for det in nms_detections {
        let bbox = det.bbox;
        let corrected_x1 = (bbox.x1 - letterbox.pad_x as f32) / letterbox.scale;
        let corrected_y1 = (bbox.y1 - letterbox.pad_y as f32) / letterbox.scale;
        let corrected_x2 = (bbox.x2 - letterbox.pad_x as f32) / letterbox.scale;
        let corrected_y2 = (bbox.y2 - letterbox.pad_y as f32) / letterbox.scale;
        corrected_detections.push(Detection {
            bbox: BoundingBox {
                x1: corrected_x1,
                y1: corrected_y1,
                x2: corrected_x2,
                y2: corrected_y2,
            },
            ..det
        });
    }

    match lang {
        Lang::En => info!(
            "Post-processing complete. Found {} final objects (after NMS).",
            corrected_detections.len()
        ),
        Lang::Zh => info!(
            "后处理完成，共找到 {} 个最终目标（NMS后）。",
            corrected_detections.len()
        ),
    }

    corrected_detections
}
