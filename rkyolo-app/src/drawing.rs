//! 绘图与标签管理模块

use crate::postprocess::Detection;
use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// 加载类别标签文件。
pub fn load_labels(path: &Path) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    reader.lines().collect()
}

/// 在图像上绘制检测结果。
pub fn draw_results(image: &mut RgbImage, detections: &[Detection], labels: &[String]) {
    let font_data: &[u8] = include_bytes!("../../NotoSans-Regular.ttf");
    let font = FontRef::try_from_slice(font_data).unwrap();

    let font_size = PxScale::from(20.0);
    let color_blue = Rgb([0, 0, 255u8]);
    let color_red = Rgb([255, 0, 0u8]);

    for det in detections {
        let bbox = &det.bbox;

        // 确保边界框坐标有效
        let x1 = bbox.x1.max(0.0) as i32;
        let y1 = bbox.y1.max(0.0) as i32;
        let x2 = bbox.x2.max(bbox.x1) as i32;
        let y2 = bbox.y2.max(bbox.y1) as i32;

        let width = (x2 - x1) as u32;
        let height = (y2 - y1) as u32;

        if width > 0 && height > 0 {
            let rect = Rect::at(x1, y1).of_size(width, height);
            draw_hollow_rect_mut(image, rect, color_blue);

            let label = if (det.class_id as usize) < labels.len() {
                &labels[det.class_id as usize]
            } else {
                "unknown"
            };
            let text = format!("{} {:.2}", label, det.confidence);

            let (_text_w, text_h) = text_size(font_size, &font, &text);

            let text_y = if y1 as u32 > text_h {
                y1 as u32 - text_h
            } else {
                0
            };

            draw_text_mut(image, color_red, x1, text_y as i32, font_size, &font, &text);
        }
    }
}
