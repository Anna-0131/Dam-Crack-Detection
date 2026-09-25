# -*- coding: utf-8 -*-
"""成员B 工具函数:检测框绘制、裂缝长度估算。"""
import math
from PIL import Image, ImageDraw, ImageFont

# 长度估算换算系数:毫米/像素
# 假设:手机 12MP 照片、拍摄距离约 1 米。演示用估算值,真实工程需现场标定。
PIXEL_TO_MM = 0.5


def bbox_diagonal_px(bbox_xyxy):
    """框对角线像素长度。裂缝框通常沿裂缝走向,对角线更接近真实裂缝长度。"""
    x1, y1, x2, y2 = bbox_xyxy
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def estimate_length_mm(bbox_xyxy):
    """裂缝长度估算(毫米)= 对角线像素数 × 换算系数。"""
    return bbox_diagonal_px(bbox_xyxy) * PIXEL_TO_MM


def draw_boxes(image, detections):
    """在图片上画检测框,返回新 PIL Image(不修改原图)。"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    line_width = max(2, min(img.size) // 200)
    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)
        label = f"{d['class']} {d['conf']:.2f}"
        text_w = draw.textlength(label, font=font)
        top = max(0, y1 - 16)  # 标签放框上方,越界时收进框内
        draw.rectangle([x1, top, x1 + text_w + 4, top + 14], fill=(255, 0, 0))
        draw.text((x1 + 2, top + 1), label, fill=(255, 255, 255), font=font)
    return img
