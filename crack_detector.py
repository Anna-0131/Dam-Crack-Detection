"""DamCrack 裂缝/剥落检测模块 —— 供 LLM agent 直接 import 调用。

用法：
    from crack_detector import detect

    result = detect("path/to/image.jpg")

    result = {
        "detections": [
            {"class": "crack",    "class_id": 0, "confidence": 0.87, "bbox_xyxy": [x1, y1, x2, y2]},
            {"class": "spalling", "class_id": 1, "confidence": 0.65, "bbox_xyxy": [x1, y1, x2, y2]},
        ],
        "summary": {"crack": 1, "spalling": 1},          # 每类数量，供 LLM 快速读
        "annotated_image": <numpy.ndarray, RGB, HxWx3>,  # 画好框的图，给界面展示
        "report": "共检出 2 处损伤：...",                  # 确定性文本摘要，供 LLM 扩写
    }

说明：
- 输入支持：文件路径(str/Path)、PIL.Image、numpy.ndarray(BGR)
- 类别固定两类：crack(裂缝)、spalling(剥落)
- bbox_xyxy 为像素坐标 [左上x, 左上y, 右下x, 右下y]
"""

import math
import os

# 必须在 import torch/ultralytics 之前设置，否则规避 OpenMP 冲突不生效
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from collections import Counter
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ==================== 配置 ====================
# 模型路径：默认用 DamCrack 2 类模型。训练完成后 best.pt 会被更新，这里不用改。
# 跨机器协作时，成员 B 可以用环境变量覆盖成她自己机器上的路径：
#     set CRACK_MODEL_PATH=D:\xxx\best.pt        (Windows cmd)
#     $env:CRACK_MODEL_PATH="D:\xxx\best.pt"     (PowerShell)
MODEL_PATH = os.environ.get(
    'CRACK_MODEL_PATH',
    r'D:\.Download\crack-detection\runs\detect\damcrack_2cls\weights\best.pt',
)
DEFAULT_CONF = 0.5    # 置信度阈值，低于此值的框不返回（0.5 演示更干净，避免一条裂缝切碎成很多框）
# ==============================================

_model = None  # 模块级缓存，只加载一次模型


def _get_model() -> YOLO:
    """懒加载模型，避免每次调用都重新加载（37MB，加载要 1~2 秒）。"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'模型不存在: {MODEL_PATH}')
        _model = YOLO(MODEL_PATH)
    return _model


def _to_bgr(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """把各种输入统一转成 numpy BGR 数组（OpenCV 约定）。"""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f'无法读取图片: {image}')
        return img
    if isinstance(image, np.ndarray):
        return image  # 约定为 BGR
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    raise TypeError(f'不支持的输入类型: {type(image)}，仅支持路径 / PIL.Image / numpy.ndarray')


def _build_quantification(detections: list) -> dict:
    """汇总量化指标：数量、长度、面积。"""
    cracks = [d for d in detections if d['class'] == 'crack']
    spalling = [d for d in detections if d['class'] == 'spalling']
    crack_lens = [d['length_px'] for d in cracks]
    max_len = max(crack_lens) if crack_lens else 0.0
    avg_len = sum(crack_lens) / len(crack_lens) if crack_lens else 0.0
    max_spall_area = max(d['area_px'] for d in spalling) if spalling else 0.0

    return {
        'crack_count': len(cracks),
        'spalling_count': len(spalling),
        'total_defects': len(detections),
        'max_crack_length_px': round(max_len, 1),
        'avg_crack_length_px': round(avg_len, 1),
        'max_spalling_area_px': round(max_spall_area, 1),
    }


def _build_report(detections: list) -> str:
    """由检测结果拼出一段确定性文本摘要，供 LLM 扩写成正式报告。"""
    if not detections:
        return '未检测到裂缝（crack）或剥落（spalling）损伤。'
    lines = [f'共检出 {len(detections)} 处损伤：']
    for d in sorted(detections, key=lambda x: -x['confidence']):
        x1, y1, x2, y2 = d['bbox_xyxy']
        # 裂缝报长度，剥落报面积
        size = f"长度 {d['length_px']:.0f}px" if d['class'] == 'crack' else f"面积 {d['area_px']:.0f}px²"
        lines.append(f"- {d['class']}（置信度 {d['confidence']:.2f}，{size}，位置 [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]）")
    return '\n'.join(lines)


def detect(
    image: Union[str, Path, Image.Image, np.ndarray],
    conf: float = DEFAULT_CONF,
    pixels_per_cm: float = None,
) -> dict:
    """对一张图片做裂缝/剥落检测，返回结构化结果、量化指标、标注图和文本摘要。

    Args:
        image: 图片（文件路径 / PIL.Image / numpy BGR 数组）
        conf: 置信度阈值，默认 0.5
        pixels_per_cm: 可选。图片的像素-厘米换算比例（每厘米多少像素），
            提供后会在每个检测框额外输出 length_cm / width_mm。

    Returns:
        dict: {"detections", "summary", "quantification", "annotated_image", "report"}
    """
    bgr = _to_bgr(image)
    model = _get_model()

    results = model(bgr, conf=conf, verbose=False)[0]

    # 1. 结构化检测结果 + 每框的量化尺寸
    names = model.names  # {0: 'crack', 1: 'spalling'}
    detections = []
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            w, h = x2 - x1, y2 - y1
            length_px = round(math.hypot(w, h), 1)   # 对角线 ≈ 裂缝长度
            width_px = round(min(w, h), 1)           # 短边 ≈ 裂缝宽度
            d = {
                'class': names[cls_id],
                'class_id': cls_id,
                'confidence': round(float(box.conf[0]), 4),
                'bbox_xyxy': [x1, y1, x2, y2],
                'length_px': length_px,
                'width_px': width_px,
                'area_px': round(w * h, 1),
            }
            if pixels_per_cm:  # 提供换算比例时，额外输出真实尺寸
                d['length_cm'] = round(length_px / pixels_per_cm, 2)
                d['width_mm'] = round(width_px / pixels_per_cm * 10, 2)
            detections.append(d)

    # 2. 每类数量汇总
    summary = dict(Counter(d['class'] for d in detections))

    # 3. 量化汇总（数量、长度、严重程度）
    quantification = _build_quantification(detections)

    # 4. 标注图（plot 返回 BGR，转成 RGB 方便 Gradio/网页展示）
    annotated_rgb = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    # 5. 文本摘要
    report = _build_report(detections)

    return {
        'detections': detections,
        'summary': summary,
        'quantification': quantification,
        'annotated_image': annotated_rgb,
        'report': report,
    }


if __name__ == '__main__':
    # 本地自测：跑一张 DamCrack 验证图，打印结构化结果
    import glob

    test_imgs = sorted(glob.glob(r'D:\.Download\crack-detection\DamCrack_YOLO\images\val\*.jpg'))
    if not test_imgs:
        print('没找到测试图')
    else:
        r = detect(test_imgs[0])
        print('detections:', r['detections'])
        print('summary:', r['summary'])
        print('report:\n' + r['report'])
        print('annotated_image shape:', r['annotated_image'].shape)
