import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------- 配置 ----------
# DamCrack 原始目录（图片在 images/，标注在 labels/Yolo/ 子目录里）
SRC = Path(r'D:\.Download\crack-detection\Damage Detection_data')
IMG_DIR = SRC / 'images'
LBL_DIR = SRC / 'labels' / 'Yolo'

# 输出的标准 YOLO 数据集目录
OUT = Path(r'D:\.Download\crack-detection\DamCrack_YOLO')

VAL_RATIO = 0.2
SEED = 42
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ------------------------

# 1. 收集图片-标注对（按文件名 stem 匹配）
pairs = []
missing = 0
for img in sorted(IMG_DIR.iterdir()):
    if img.suffix.lower() not in IMAGE_EXTS:
        continue
    lbl = LBL_DIR / f'{img.stem}.txt'
    if lbl.exists():
        pairs.append((img, lbl))
    else:
        missing += 1

print(f'找到图片-标注对: {len(pairs)} 张')
if missing:
    print(f'[警告] 缺标注的图片: {missing} 张')

# 2. 打乱并划分 train/val
random.seed(SEED)
random.shuffle(pairs)
split_idx = int(len(pairs) * (1 - VAL_RATIO))
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]
print(f'train: {len(train_pairs)} 张 / val: {len(val_pairs)} 张')

# 3. 清空旧输出，避免残留
if OUT.exists():
    shutil.rmtree(OUT)

# 4. 复制文件到标准结构
for split_name, items in [('train', train_pairs), ('val', val_pairs)]:
    img_out = OUT / 'images' / split_name
    lbl_out = OUT / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in items:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(lbl_path, lbl_out / lbl_path.name)
    print(f'  {split_name}: {len(items)} 张 已复制')

# 5. 生成 data.yaml（2 类）
yaml_content = f"""path: {OUT.resolve()}
train: images/train
val: images/val
nc: 2
names:
- crack
- spalling
"""
(OUT / 'data.yaml').write_text(yaml_content.strip(), encoding='utf-8')

print(f'\n[完成] 标准数据集已生成: {OUT}')
print(f'  data.yaml: {OUT / "data.yaml"}')
print('  类别: 0=crack, 1=spalling')
