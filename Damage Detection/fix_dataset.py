import os
import shutil
import random
from pathlib import Path

# ---------- 请根据你的实际情况修改这里 ----------
# 你的原始数据集根目录，注意路径中的斜杠
dataset_root = Path(r'D:\.Download\crack-detection\Damage Detection\Damage Detection')
# ----------------------------------------------

# 1. 定位图片和标注文件夹（YOLO会查找这两个名字）
images_source_dir = dataset_root / 'images'  # 原始图片目录
labels_source_dir = dataset_root / 'labels'  # 原始标注目录

# 如果上面两个目录不存在，尝试寻找可能的目录名（如 Images 大写）
if not images_source_dir.exists():
    for d in dataset_root.iterdir():
        if d.is_dir() and d.name.lower() == 'images':
            images_source_dir = d
            print(f"找到图片文件夹: {d}")
            break
if not labels_source_dir.exists():
    for d in dataset_root.iterdir():
        if d.is_dir() and d.name.lower() == 'labels':
            labels_source_dir = d
            print(f"找到标注文件夹: {d}")
            break

# 2. 检查两个文件夹是否存在
if not images_source_dir.exists():
    raise FileNotFoundError(f"错误：找不到图片文件夹，请确认 {dataset_root} 下有 images 或 Images 文件夹。")
if not labels_source_dir.exists():
    raise FileNotFoundError(f"错误：找不到标注文件夹，请确认 {dataset_root} 下有 labels 或 Labels 文件夹。")

# 3. 获取所有图片文件
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
image_files = [f for f in images_source_dir.iterdir() if f.suffix.lower() in image_extensions]
print(f"找到 {len(image_files)} 张图片")

# 确保图片和标注文件能对应上
valid_pairs = []
for img_path in image_files:
    label_path = labels_source_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        valid_pairs.append((img_path, label_path))
    else:
        print(f"警告：图片 {img_path.name} 没有对应的标注文件，已跳过。")

print(f"有 {len(valid_pairs)} 对有效的图片-标注文件")

if not valid_pairs:
    raise RuntimeError("没有找到任何有效的图片-标注文件对，请检查文件命名是否一致。")

# 4. 打乱顺序并划分训练集和验证集 (80% 训练, 20% 验证)
random.shuffle(valid_pairs)
split_idx = int(len(valid_pairs) * 0.8)
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

# 5. 创建符合 YOLO 标准的新目录结构
new_dataset_root = dataset_root.parent / 'Damage_Detection_YOLO'  # 在父目录下创建一个新文件夹
print(f"正在创建标准数据集于: {new_dataset_root}")

for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
    # 创建 images 和 labels 下的子目录
    img_dir = new_dataset_root / 'images' / split_name
    lbl_dir = new_dataset_root / 'labels' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, img_dir / img_path.name)
        shutil.copy(lbl_path, lbl_dir / lbl_path.name)

    print(f"  {split_name}: {len(pairs)} 张图片和标注")

# 6. 在数据集根目录下创建 data.yaml 配置文件
data_yaml_content = f"""
# 数据集路径（使用新创建的目录）
path: {new_dataset_root.resolve()}  # 数据集根目录的绝对路径
train: images/train  # 训练图片相对路径
val: images/val      # 验证图片相对路径

# 类别信息
nc: 2  # 类别数量
names: ['crack', 'spalling']  # 类别名称
"""

yaml_path = new_dataset_root / 'data.yaml'
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(data_yaml_content.strip())

print(f"\n✅ 数据集修复完成！")
print(f"新的数据集位置: {new_dataset_root}")
print(f"配置文件位置: {yaml_path}")
print("\n现在，请将你的训练脚本中的 data 参数修改为这个新的 data.yaml 路径。")