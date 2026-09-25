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

import os
from collections import Counter
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==================== 配置 ====================
# 模型路径：默认用 DamCrack 2 类模型。训练完成后 best.pt 会被更新，这里不用改。
# 跨机器协作时，成员 B 可以用环境变量覆盖成她自己机器上的路径：
#     set CRACK_MODEL_PATH=D:\xxx\best.pt        (Windows cmd)
#     $env:CRACK_MODEL_PATH="D:\xxx\best.pt"     (PowerShell)
MODEL_PATH = os.environ.get(
    'CRACK_MODEL_PATH',
    r'D:\.Download\crack-detection\runs\detect\damcrack_2cls\weights\best.pt',
)
DEFAULT_CONF = 0.25   # 置信度阈值，低于此值的框不返回
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


def _build_report(detections: list) -> str:
    """由检测结果拼出一段确定性文本摘要，供 LLM 扩写成正式报告。"""
    if not detections:
        return '未检测到裂缝（crack）或剥落（spalling）损伤。'
    lines = [f'共检出 {len(detections)} 处损伤：']
    for d in sorted(detections, key=lambda x: -x['confidence']):
        x1, y1, x2, y2 = d['bbox_xyxy']
        lines.append(f"- {d['class']}（置信度 {d['confidence']:.2f}，位置 [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]）")
    return '\n'.join(lines)


def detect(
    image: Union[str, Path, Image.Image, np.ndarray],
    conf: float = DEFAULT_CONF,
) -> dict:
    """对一张图片做裂缝/剥落检测，一次返回结构化结果、标注图和文本摘要。

    Args:
        image: 图片（文件路径 / PIL.Image / numpy BGR 数组）
        conf: 置信度阈值，默认 0.25

    Returns:
        dict: {"detections", "summary", "annotated_image", "report"}
    """
    bgr = _to_bgr(image)
    model = _get_model()

    results = model(bgr, conf=conf, verbose=False)[0]

    # 1. 结构化检测结果
    names = model.names  # {0: 'crack', 1: 'spalling'}
    detections = []
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            detections.append({
                'class': names[cls_id],
                'class_id': cls_id,
                'confidence': round(float(box.conf[0]), 4),
                'bbox_xyxy': [round(v, 1) for v in box.xyxy[0].tolist()],
            })

    # 2. 每类数量汇总
    summary = dict(Counter(d['class'] for d in detections))

    # 3. 标注图（plot 返回 BGR，转成 RGB 方便 Gradio/网页展示）
    annotated_rgb = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    # 4. 文本摘要
    report = _build_report(detections)

    return {
        'detections': detections,
        'summary': summary,
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
