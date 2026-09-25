import shutil
import random
import sys
from pathlib import Path

# 让 Windows 控制台能正常输出中文，避免 GBK 编码报错
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------- 配置 ----------
INPUT_DIR = Path(r'd:\.Download\crack-detection\positive_dataset')   # 新数据集
OUTPUT_DIR = Path(r'd:\.Download\crack-detection\positive_dataset_YOLO')  # 划分后的标准 YOLO 目录
VAL_RATIO = 0.2        # 验证集比例
SEED = 42              # 固定随机种子，保证每次划分结果一致
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ------------------------

src_images = INPUT_DIR / 'image'
src_labels = INPUT_DIR / 'label'

# 1. 收集有效的 图片-标注 配对（跳过 classes.txt、跳过缺标注的图片）
valid_pairs = []
skipped_no_label = []

for img_path in sorted(src_images.iterdir()):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue
    label_path = src_labels / f"{img_path.stem}.txt"
    if label_path.exists():
        valid_pairs.append((img_path, label_path))
    else:
        skipped_no_label.append(img_path.name)

print(f"有效配对: {len(valid_pairs)} 张")
if skipped_no_label:
    print(f"[警告] 跳过（缺标注文件）: {skipped_no_label}")

# 2. 固定种子打乱，划分 train/val
random.seed(SEED)
random.shuffle(valid_pairs)
split_idx = int(len(valid_pairs) * (1 - VAL_RATIO))
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

# 3. 写入 YOLO 标准目录结构
for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
    img_dir = OUTPUT_DIR / 'images' / split_name
    lbl_dir = OUTPUT_DIR / 'labels' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dir / img_path.name)
        shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
    print(f"  {split_name}: {len(pairs)} 张")

# 4. 生成 data.yaml（单类别 crack）
yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
nc: 1
names:
- crack
"""
yaml_path = OUTPUT_DIR / 'data.yaml'
yaml_path.write_text(yaml_content.strip(), encoding='utf-8')

print(f"\n[完成] 划分完成！")
print(f"   输出目录: {OUTPUT_DIR}")
print(f"   配置文件: {yaml_path}")
