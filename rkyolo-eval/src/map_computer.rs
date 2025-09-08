use crate::{Bbox as PredBbox, GroundTruthBox, Prediction};
use log::{debug, trace};
use std::collections::HashMap;

/// 用于内部计算的绝对坐标边界框
#[derive(Debug, Clone, Copy)]
struct AbsBbox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

/// 用于存储单个预测框在匹配过程中的状态
#[derive(Debug, Clone)]
struct PrResult {
    is_tp: bool, // 是真阳性(True Positive)吗?
}

/// 计算两个绝对坐标边界框的交并比 (IoU)
fn calculate_iou(box1: &AbsBbox, box2: &AbsBbox) -> f32 {
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

/// 【我们的"公共函数"】根据 P-R 点对计算曲线下面积 (AP)
/// 使用 PASCAL VOC 2010+ 的所有点插值法
fn calculate_area_under_curve(pr_points: &[(f32, f32)]) -> f32 {
    if pr_points.is_empty() {
        return 0.0;
    }

    // 将 (recall, precision) 分离，并预留一个点
    let recalls: Vec<f32> = pr_points.iter().map(|p| p.0).collect();
    let mut precisions: Vec<f32> = pr_points.iter().map(|p| p.1).collect();

    // 1. 平滑 Precision 曲线，使其单调递减
    for i in (0..precisions.len() - 1).rev() {
        precisions[i] = precisions[i].max(precisions[i + 1]);
    }

    // 2. 在 recall 变化的点上计算面积
    let mut ap = 0.0;
    // 显式添加一个 (r=0, p=p_at_first_recall) 的点，简化计算
    ap += recalls[0] * precisions[0];
    for i in 1..pr_points.len() {
        ap += (recalls[i] - recalls[i - 1]) * precisions[i];
    }
    ap
}

/// 计算单个类别的平均精度 (AP)
fn calculate_ap_for_class(
    class_id: u32,
    predictions: &HashMap<String, Vec<Prediction>>,
    ground_truths: &HashMap<String, Vec<GroundTruthBox>>,
    image_dimensions: &HashMap<String, (u32, u32)>,
    iou_threshold: f32,
) -> f32 {
    // 1. 收集该类别的所有预测和真实框
    let mut class_preds: Vec<(String, PredBbox, f32)> = Vec::new();
    let mut total_gt_count = 0;

    for (image_name, preds) in predictions {
        for pred in preds {
            if pred.class_id == class_id {
                class_preds.push((image_name.clone(), pred.bbox, pred.confidence));
            }
        }
    }

    for gts in ground_truths.values() {
        for gt in gts {
            if gt.class_id == class_id {
                total_gt_count += 1;
            }
        }
    }

    if total_gt_count == 0 {
        return if class_preds.is_empty() { 1.0 } else { 0.0 }; // 没有GT时，若也无预测则完美，否则AP为0
    }
    if class_preds.is_empty() {
        return 0.0; // 有GT但无预测，AP为0
    }

    // 2. 按置信度从高到低排序预测
    class_preds.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // 3. 匹配 TP 和 FP
    let mut pr_results = Vec::with_capacity(class_preds.len());
    let mut matched_gt: HashMap<String, Vec<bool>> = HashMap::new();

    for (image_name, pred_bbox, confidence) in &class_preds {
        trace!("\n[TRACE] ===================================================================");
        trace!(
            "[TRACE] Processing Prediction on '{}': conf={:.4}, bbox=({:.1}, {:.1}, {:.1}, {:.1})",
            image_name, confidence, pred_bbox.x1, pred_bbox.y1, pred_bbox.x2, pred_bbox.y2
        );

        let gt_boxes = ground_truths
            .get(image_name)
            .map_or([].as_slice(), |v| v.as_slice());
        let (img_w, img_h) = image_dimensions.get(image_name).unwrap();

        let pred_abs_bbox = AbsBbox {
            x1: pred_bbox.x1,
            y1: pred_bbox.y1,
            x2: pred_bbox.x2,
            y2: pred_bbox.y2,
        };
        let mut best_iou = -1.0;
        let mut best_gt_idx = -1;

        let matched_flags = matched_gt
            .entry(image_name.clone())
            .or_insert_with(|| vec![false; gt_boxes.len()]);

        let class_gt_count = gt_boxes.iter().filter(|gt| gt.class_id == class_id).count();
        if class_gt_count == 0 {
            trace!(
                "[TRACE] Image '{}' has no GT boxes for this class.",
                image_name
            );
        } else {
            trace!(
                "[TRACE] Image '{}' has {} GT boxes for this class.",
                image_name, class_gt_count
            );
        }

        for (i, gt) in gt_boxes.iter().enumerate() {
            if gt.class_id != class_id {
                continue; // 只与同类别的 GT 比较
            }

            // 反归一化 GT box
            let w = gt.width * (*img_w as f32);
            let h = gt.height * (*img_h as f32);
            let x1 = gt.x_center * (*img_w as f32) - w / 2.0;
            let y1 = gt.y_center * (*img_h as f32) - h / 2.0;
            let gt_abs_bbox = AbsBbox {
                x1,
                y1,
                x2: x1 + w,
                y2: y1 + h,
            };

            let iou = calculate_iou(&pred_abs_bbox, &gt_abs_bbox);

            trace!(
                "[TRACE]   -> vs GT #{}: bbox=({:.1}, {:.1}, {:.1}, {:.1}), IoU={:.4}, matched_before={}",
                i,
                gt_abs_bbox.x1,
                gt_abs_bbox.y1,
                gt_abs_bbox.x2,
                gt_abs_bbox.y2,
                iou,
                matched_flags[i]
            );

            if iou > best_iou && !matched_flags[i] {
                // 只有尚未匹配的 GT 才能成为最佳匹配
                best_iou = iou;
                best_gt_idx = i as i32;
            }
        }

        if best_iou > iou_threshold {
            trace!(
                "[TRACE] ==> Match FOUND with GT #{}. IoU {:.4} > threshold {:.2}. Assigning TP.",
                best_gt_idx, best_iou, iou_threshold
            );
            matched_flags[best_gt_idx as usize] = true;
            pr_results.push(PrResult { is_tp: true });
        } else {
            trace!(
                "[TRACE] ==> Match NOT found. Best IoU was {:.4}. Assigning FP.",
                best_iou
            );
            pr_results.push(PrResult { is_tp: false });
        }
    }

    // 4. 计算 P-R 曲线点
    let mut cumulative_tp = 0;
    let mut cumulative_fp = 0;
    let mut pr_points = Vec::with_capacity(pr_results.len());

    for result in &pr_results {
        if result.is_tp {
            cumulative_tp += 1;
        } else {
            cumulative_fp += 1;
        }
        let precision = cumulative_tp as f32 / (cumulative_tp + cumulative_fp) as f32;
        let recall = cumulative_tp as f32 / total_gt_count as f32;
        pr_points.push((recall, precision));
    }

    // 5. 计算 AP
    calculate_area_under_curve(&pr_points)
}

/// 公共的计算函数，这是我们评估工具的入口点
pub fn calculate_map(
    predictions: &HashMap<String, Vec<Prediction>>,
    ground_truths: &HashMap<String, Vec<GroundTruthBox>>,
    image_dimensions: &HashMap<String, (u32, u32)>,
    num_classes: u32,
    iou_threshold: f32,
) -> f32 {
    println!("\nStarting mAP calculation for {} classes...", num_classes);
    let mut ap_sum = 0.0;
    let mut class_count = 0;

    for class_id in 0..num_classes {
        let ap = calculate_ap_for_class(
            class_id,
            predictions,
            ground_truths,
            image_dimensions,
            iou_threshold,
        );
        // 你可以在这里打印每个类别的 AP
        // println!("  - AP for class {}: {:.4}", class_id, ap);
        ap_sum += ap;
        class_count += 1;
    }

    if class_count > 0 {
        ap_sum / class_count as f32
    } else {
        0.0
    }
}

/// 【新增】计算 R² (决定系数) 以评估模型的计数准确性
///
/// # Arguments
/// * `predictions` - 模型的预测结果
/// * `ground_truths` - 真实标签
/// * `confidence_threshold` - 用于计数的置信度阈值
///
/// # Returns
/// * `Some(f32)` 包含 R² 分数，如果数据不足则返回 `None`
pub fn calculate_r_squared(
    predictions: &HashMap<String, Vec<Prediction>>,
    ground_truths: &HashMap<String, Vec<GroundTruthBox>>,
    confidence_threshold: f32,
    outlier_threshold: Option<u32>, // 筛选离群值的阈值
) -> Option<f32> {
    // 1. 为每张图片生成 (真实数量, 预测数量) 的数据对
    let mut counts: Vec<(f32, f32)> = Vec::new();
    let mut outliers: Vec<(String, u32, u32, u32)> = Vec::new();

    // --- 【插入日志 - 表头】 ---
    debug!(
        "\n[DEBUG] R² Calculation Data (GT vs. Pred Counts @ conf>{:.2}):",
        confidence_threshold
    );
    debug!(
        "[DEBUG] {:<40} | {:<10} | {:<10}",
        "Image Name", "GT Count", "Pred Count"
    );
    debug!("[DEBUG] ------------------------------------------|------------|-------------");
    // --- 【结束插入】 ---

    // 遍历所有有真实标签的图片，以确保我们评估的是同一个集合
    for image_name in ground_truths.keys() {
        let actual_count = ground_truths.get(image_name).unwrap().len() as f32;

        // 计算超过阈值的预测数量
        let predicted_count = predictions.get(image_name).map_or(0.0, |preds| {
            preds
                .iter()
                .filter(|p| p.confidence >= confidence_threshold)
                .count() as f32
        });
        // --- 【插入日志 - 表格行】 ---
        debug!(
            "[DEBUG] {:<40} | {:<10} | {:<10}",
            image_name, actual_count, predicted_count
        );
        // --- 【结束插入】 ---

        counts.push((actual_count, predicted_count));
        // 【新增】检查并记录离群值
        if let Some(threshold) = outlier_threshold {
            let diff = (actual_count - predicted_count).abs() as u32;
            if diff > threshold {
                outliers.push((
                    image_name.clone(),
                    actual_count as u32,
                    predicted_count as u32,
                    diff,
                ));
            }
        }
    }

    if counts.len() < 2 {
        // R² 至少需要2个数据点才有意义
        return None;
    }

    // 2. 计算真实数量的平均值
    let total_actual_sum: f32 = counts.iter().map(|(y, _)| *y).sum();
    let mean_actual = total_actual_sum / counts.len() as f32;

    // 3. 计算 SSR (残差平方和) 和 SST (总平方和)
    let mut ssr = 0.0;
    let mut sst = 0.0;

    for (y, y_hat) in &counts {
        ssr += (*y - *y_hat).powi(2);
        sst += (*y - mean_actual).powi(2);
    }

    // 4. 处理 SST 为 0 的特殊情况 (所有图片的真实物体数都一样)
    if sst == 0.0 {
        // 如果 SSR 也为 0，说明预测完美，R²=1。否则该指标无意义，返回0。
        return Some(if ssr == 0.0 { 1.0 } else { 0.0 });
    }

    if let Some(_) = outlier_threshold {
        if !outliers.is_empty() {
            // 按差异值从大到小排序
            outliers.sort_unstable_by_key(|k| std::cmp::Reverse(k.3));

            debug!(
                "\n--- Outlier Report (Count Difference > {}) ---",
                outlier_threshold.unwrap()
            );
            debug!(
                "{:<40} | {:<10} | {:<10} | {:<10}",
                "Image Name", "GT Count", "Pred Count", "Difference"
            );
            debug!(
                "------------------------------------------|------------|------------|------------"
            );
            for (name, gt, pred, diff) in outliers {
                debug!("{:<40} | {:<10} | {:<10} | {:<10}", name, gt, pred, diff);
            }
        } else {
            debug!("\n--- Outlier Report: No outliers found with the given threshold. ---");
        }
    }

    // 5. 计算最终的 R²
    Some(1.0 - (ssr / sst))
}
