//! YOLO 模型后处理模块 (INT8 版本)

use crate::LetterboxInfo;
use rknn_ffi::raw::rknn_tensor_attr;
use std::collections::HashMap;

// --- 常量定义 ---
const OBJ_CLASS_NUM: usize = 80; // COCO 数据集有80个类别
const PROP_BOX_SIZE: usize = OBJ_CLASS_NUM + 4; // 每个检测 proposta 的大小 (cx, cy, w, h, cls_scores...)

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

    // 【终极修正】: 类别数直接由score张量的通道数决定！
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
) -> Vec<Detection> {
    let model_in_h = 640;
    let output_per_branch = output_attrs.len() / 3;
    let mut all_detections = Vec::new();

    for i in 0..3 {
        let box_idx = i * output_per_branch;
        let score_idx = i * output_per_branch + 1;

        let box_attr = &output_attrs[box_idx];
        let score_attr = &output_attrs[score_idx];

        let box_data_u8 = outputs_data[box_idx];
        let score_data_u8 = outputs_data[score_idx];

        let box_data: &[i8] = unsafe {
            std::slice::from_raw_parts(box_data_u8.as_ptr() as *const i8, box_data_u8.len())
        };
        let score_data: &[i8] = unsafe {
            std::slice::from_raw_parts(score_data_u8.as_ptr() as *const i8, score_data_u8.len())
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

    // --- 【NMS 逻辑重构】 ---
    // 1. 按置信度对所有候选框进行一次全局降序排序
    all_detections.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut nms_detections = Vec::new();
    while !all_detections.is_empty() {
        // 2. 取出当前置信度最高的框
        let best_det = all_detections.remove(0);
        nms_detections.push(best_det.clone());

        // 3. 过滤掉与 best_det 重叠度高且类别相同的框
        all_detections.retain(|det| {
            // 只有当类别相同时，才计算IoU进行抑制
            if det.class_id == best_det.class_id {
                calculate_iou(&best_det.bbox, &det.bbox) < nms_threshold
            } else {
                true // 不同类别，不抑制
            }
        });
    }

    // --- 对经过NMS的结果进行坐标变换 ---
    let mut corrected_detections = Vec::with_capacity(nms_detections.len());
    for det in nms_detections {
        // 注意：这里我们遍历的是 nms_detections
        let bbox = det.bbox;

        // 应用Letterbox的逆运算
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

    println!(
        "后处理完成，共找到 {} 个最终目标（NMS后）",
        corrected_detections.len()
    );
    corrected_detections
}
