//! YOLO 模型后处理模块 (INT8 版本)

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

    // 遍历每一个网格单元
    for y in 0..grid_h {
        for x in 0..grid_w {
            let offset = y * grid_w + x;

            // 找到分数最高的类别
            let mut max_score = 0.0;
            let mut max_class_id = -1;

            for c in 0..OBJ_CLASS_NUM {
                let score_q = score_data[c * grid_len + offset];
                let score = dequantize(score_q, score_attr.zp, score_attr.scale);
                if score > max_score {
                    max_score = score;
                    max_class_id = c as i32;
                }
            }

            // 如果最高分大于阈值，则解码边界框
            if max_score > conf_threshold {
                let box_offset = (y * grid_w + x) * 4;
                let x1_q = box_data[box_offset + 0];
                let y1_q = box_data[box_offset + 1];
                let x2_q = box_data[box_offset + 2];
                let y2_q = box_data[box_offset + 3];

                let x1_f = dequantize(x1_q, box_attr.zp, box_attr.scale);
                let y1_f = dequantize(y1_q, box_attr.zp, box_attr.scale);
                let x2_f = dequantize(x2_q, box_attr.zp, box_attr.scale);
                let y2_f = dequantize(y2_q, box_attr.zp, box_attr.scale);

                // (cx, cy, w, h) -> (x1, y1, x2, y2)
                let center_x = (x as f32 - x1_f + 0.5) * stride as f32;
                let center_y = (y as f32 - y1_f + 0.5) * stride as f32;
                let width = (x as f32 + x2_f + 0.5) * stride as f32;
                let height = (y as f32 + y2_f + 0.5) * stride as f32;

                detections.push(Detection {
                    bbox: BoundingBox {
                        x1: center_x,
                        y1: center_y,
                        x2: width,
                        y2: height,
                    },
                    confidence: max_score,
                    class_id: max_class_id,
                });
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

    // --- 按类别进行非极大值抑制 (NMS) ---
    let mut final_detections = Vec::new();
    let mut detections_by_class: HashMap<i32, Vec<Detection>> = HashMap::new();

    // 1. 按 class_id 分组
    for det in all_detections {
        detections_by_class
            .entry(det.class_id)
            .or_default()
            .push(det);
    }

    // 2. 对每个类别独立进行 NMS
    for (_class_id, mut detections) in detections_by_class {
        // 按置信度降序排序
        detections.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut nms_detections = Vec::new();
        while !detections.is_empty() {
            let best_det = detections.remove(0);
            nms_detections.push(best_det.clone());

            detections.retain(|det| calculate_iou(&best_det.bbox, &det.bbox) < nms_threshold);
        }
        final_detections.extend(nms_detections);
    }

    println!(
        "后处理完成，共找到 {} 个最终目标（NMS后）",
        final_detections.len()
    );
    final_detections
}
