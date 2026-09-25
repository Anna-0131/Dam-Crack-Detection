# -*- coding: utf-8 -*-
"""缺陷检测模块 —— 成员A 负责实现真实版本(YOLO 推理)。

当前为 MOCK 版:按图片尺寸自动生成两条横向裂缝框,保证任何图片都能看到红框。
成员A 完成真实实现后,整个替换本文件,保持函数签名不变即可。
"""
from PIL import Image


def detect(image_path: str) -> list:
    """输入图片路径,返回检测结果列表;无缺陷返回 []。
    格式:[{"class": "crack", "conf": 0.87, "bbox_xyxy": [x1, y1, x2, y2]}, ...]
    """
    # ===== MOCK 数据:两条横向裂缝,位置按图片宽高的比例放置 =====
    # 裂缝1:高度 35%~40% 处,宽度 20%~80%
    # 裂缝2:高度 65%~70% 处,宽度 20%~80%
    # ★ 等选定正式测试图后,可把坐标改成裂缝实际位置(Windows 画图可看像素坐标)
    w, h = Image.open(image_path).size
    return [
        {"class": "crack", "conf": 0.92,
         "bbox_xyxy": [int(w * 0.2), int(h * 0.35), int(w * 0.8), int(h * 0.40)]},
        {"class": "crack", "conf": 0.81,
         "bbox_xyxy": [int(w * 0.2), int(h * 0.65), int(w * 0.8), int(h * 0.70)]},
    ]
