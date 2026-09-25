"""
auto_label.py - 使用训练好的模型自动标注新图片
"""
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# 配置路径
MODEL_PATH = r"D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\best.pt"
NEW_IMAGES_DIR = r"D:\.Download\crack-detection\new_dataset\image"  # 你的500张新图片
OUTPUT_DIR = r"D:\.Download\crack-detection\new_dataset\label_model"  # 输出目录

# 创建输出目录
output_images_dir = Path(OUTPUT_DIR) / "images"
output_labels_dir = Path(OUTPUT_DIR) / "labels"
output_images_dir.mkdir(parents=True, exist_ok=True)
output_labels_dir.mkdir(parents=True, exist_ok=True)

# 加载模型
print(f"正在加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 置信度阈值（可以调低一点，宁可多检出，稍后人工删除）
CONF_THRESHOLD = 0.15

# 获取所有图片
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(Path(NEW_IMAGES_DIR).glob(f'*{ext}'))
    image_files.extend(Path(NEW_IMAGES_DIR).glob(f'*{ext.upper()}'))

print(f"找到 {len(image_files)} 张图片，开始自动标注...")

# 自动标注
for i, img_path in enumerate(image_files, 1):
    print(f"\r处理进度: {i}/{len(image_files)}", end="")

    # 读取图片（用于获取尺寸和保存）
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"\n无法读取图片: {img_path}")
        continue

    h, w = img.shape[:2]

    # 模型预测
    results = model(str(img_path), conf=CONF_THRESHOLD, device='cpu')

    # 提取检测框
    boxes = results[0].boxes
    label_lines = []

    if boxes is not None:
        for box in boxes:
            # 获取归一化的坐标（YOLO格式）
            xyxy = box.xyxyn[0].tolist()  # [x1, y1, x2, y2] 归一化
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # 转换为 YOLO 格式: class x_center y_center width height
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]

            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存图片（复制到输出目录）
    output_img_path = output_images_dir / img_path.name
    cv2.imwrite(str(output_img_path), img)

    # 保存标注文件（即使没有检测到框，也保存空文件，表示这是无裂缝图片）
    output_label_path = output_labels_dir / f"{img_path.stem}.txt"
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(label_lines))

print(f"\n\n✅ 自动标注完成！")
print(f"   图片保存到: {output_images_dir}")
print(f"   标注保存到: {output_labels_dir}")
print(f"\n现在可以用 LabelImg 打开这个目录进行人工修正了！")