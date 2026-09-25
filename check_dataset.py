import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DATASET = Path(r'd:\.Download\crack-detection\combined_dataset_YOLO')
problems = []
stats = {}

# 1. 检查 data.yaml
yaml_path = DATASET / 'data.yaml'
print("=== 1. data.yaml ===")
print(yaml_path.read_text(encoding='utf-8'))
print()

# 2. 逐 split 检查
for split in ['train', 'val']:
    img_dir = DATASET / 'images' / split
    lbl_dir = DATASET / 'labels' / split
    imgs = sorted(img_dir.iterdir()) if img_dir.exists() else []
    lbls = sorted(lbl_dir.iterdir()) if lbl_dir.exists() else []

    n_img = len(imgs)
    n_lbl = len(lbls)
    print(f"=== 2. {split}: 图片 {n_img} / 标注 {n_lbl} ===")

    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}

    missing_lbl = img_stems - lbl_stems
    missing_img = lbl_stems - img_stems
    if missing_lbl:
        problems.append(f"{split}: 有图片缺标注 {sorted(missing_lbl)[:5]}")
    if missing_img:
        problems.append(f"{split}: 有标注缺图片 {sorted(missing_img)[:5]}")

    n_empty = 0
    bad_label_files = []
    bad_boxes = 0
    for lp in lbls:
        text = lp.read_text(encoding='utf-8').strip()
        if not text:
            n_empty += 1
            continue
        for ln in text.splitlines():
            parts = ln.split()
            if len(parts) != 5:
                bad_label_files.append((lp.name, ln))
                continue
            try:
                cls, cx, cy, w, h = [float(x) for x in parts]
            except ValueError:
                bad_label_files.append((lp.name, ln))
                continue
            if cls != 0:
                problems.append(f"{split}/{lp.name}: 类别不是 0 -> {cls}")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                bad_boxes += 1
            if w <= 0 or h <= 0:
                bad_boxes += 1
            # 允许框贴到图像边缘（裂缝常延伸到边缘，YOLO 会自动裁剪），只有明显越界才报错
            if cx - w / 2 < -0.005 or cx + w / 2 > 1.005 or cy - h / 2 < -0.005 or cy + h / 2 > 1.005:
                bad_boxes += 1

    stats[split] = {'total': n_img, 'positive(有框)': n_img - n_empty, 'negative(空标注)': n_empty}
    print(f"   正样本(有标注): {n_img - n_empty} / 负样本(空标注): {n_empty}")
    if bad_label_files:
        print(f"   [问题] 格式错误的标注行: {bad_label_files[:5]}")
        problems.append(f"{split}: {len(bad_label_files)} 行标注格式错误")
    if bad_boxes:
        print(f"   [问题] 坐标越界/非法框: {bad_boxes} 个")
        problems.append(f"{split}: {bad_boxes} 个非法框")
    if missing_lbl or missing_img:
        print(f"   [问题] 缺标注 {len(missing_lbl)} / 缺图片 {len(missing_img)}")
    else:
        print("   [OK] 图片-标注一一对应")

# 3. 检查 train/val 是否有重名（数据泄漏）
train_stems = {p.stem for p in (DATASET / 'images' / 'train').iterdir()}
val_stems = {p.stem for p in (DATASET / 'images' / 'val').iterdir()}
overlap = train_stems & val_stems
print(f"\n=== 3. train/val 重名检查: {len(overlap)} 个重叠 ===")
if overlap:
    problems.append(f"train/val 重名: {sorted(overlap)[:5]}")

# 4. 图片文件是否为空
empty_imgs = [p.name for split in ['train', 'val'] for p in (DATASET / 'images' / split).iterdir() if p.stat().st_size == 0]
print(f"=== 4. 空图片文件: {len(empty_imgs)} 个 ===")
if empty_imgs:
    problems.append(f"空图片: {empty_imgs[:5]}")

# 汇总
print("\n" + "=" * 50)
if problems:
    print("[发现问题]")
    for p in problems:
        print("  -", p)
else:
    print("[全部通过] 数据集完整，可以开始训练")
    print(f"  train: 共 {stats['train']['total']} 张（正 {stats['train']['positive(有框)']} / 负 {stats['train']['negative(空标注)']}）")
    print(f"  val:   共 {stats['val']['total']} 张（正 {stats['val']['positive(有框)']} / 负 {stats['val']['negative(空标注)']}）")
