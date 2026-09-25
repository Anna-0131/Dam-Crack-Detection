import os
from pathlib import Path
from ultralytics import YOLO

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ========== 配置区域 ==========
# 模型路径（使用你训练好的150轮模型）
MODEL_PATH = r'D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\best.pt'

# 输入文件夹（存放要预测的图片）
INPUT_DIR = r'D:\.Download\crack-detection\test'

# 输出文件夹（保存预测结果）
OUTPUT_DIR = r'D:\.Download\crack-detection\prediction_results'

# 置信度阈值（0.25-0.5之间调整）
CONF_THRESHOLD = 0.3

# ========== 开始预测 ==========
print("=" * 50)
print("批量裂缝检测脚本")
print("=" * 50)

# 1. 检查模型是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型不存在: {MODEL_PATH}")
    exit()

# 2. 检查输入文件夹是否存在
if not os.path.exists(INPUT_DIR):
    print(f"❌ 输入文件夹不存在: {INPUT_DIR}")
    exit()

# 3. 创建输出文件夹
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 4. 加载模型
print(f"\n📦 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✅ 模型加载成功")
print(f"   识别类别: {model.names}")

# 5. 获取所有图片文件
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
image_files = []

for ext in image_extensions:
    image_files.extend(Path(INPUT_DIR).glob(f'*{ext}'))
    image_files.extend(Path(INPUT_DIR).glob(f'*{ext.upper()}'))

image_files = sorted(set(image_files))  # 去重并排序
print(f"\n📷 找到 {len(image_files)} 张图片")

if len(image_files) == 0:
    print(f"❌ 在 {INPUT_DIR} 中没有找到图片文件")
    exit()

# 6. 批量预测
print(f"\n🔍 开始预测...")
print("-" * 50)

results_list = []
crack_stats = []

for i, img_path in enumerate(image_files, 1):
    print(f"\r处理进度: {i}/{len(image_files)}", end="")

    # 预测
    results = model(str(img_path), conf=CONF_THRESHOLD, save=False)

    # 获取检测结果
    num_cracks = len(results[0].boxes) if results[0].boxes is not None else 0

    # 保存结果图片
    output_path = Path(OUTPUT_DIR) / f"result_{img_path.name}"
    results[0].save(str(output_path))

    # 记录统计信息
    crack_stats.append({
        'filename': img_path.name,
        'num_cracks': num_cracks,
        'output_path': str(output_path)
    })

    results_list.append(results)

print(f"\n\n✅ 预测完成！")
print(f"   处理图片: {len(image_files)} 张")
print(f"   结果保存到: {OUTPUT_DIR}")

# 7. 生成统计报告
print("\n" + "=" * 50)
print("检测统计报告")
print("=" * 50)

total_cracks = sum(s['num_cracks'] for s in crack_stats)
images_with_cracks = sum(1 for s in crack_stats if s['num_cracks'] > 0)
images_without_cracks = len(crack_stats) - images_with_cracks

print(f"总裂缝数: {total_cracks}")
print(f"有裂缝的图片: {images_with_cracks} 张")
print(f"无裂缝的图片: {images_without_cracks} 张")

if images_with_cracks > 0:
    avg_cracks = total_cracks / images_with_cracks
    print(f"平均每张有裂缝的图片: {avg_cracks:.2f} 个裂缝")

# 8. 显示每张图片的详细信息
print("\n" + "-" * 50)
print("详细结果:")
print("-" * 50)

for stat in crack_stats:
    if stat['num_cracks'] > 0:
        print(f"  ✓ {stat['filename']}: {stat['num_cracks']} 个裂缝")
    else:
        print(f"  ✗ {stat['filename']}: 无裂缝")

# 9. 保存统计报告到文件
report_path = Path(OUTPUT_DIR) / 'detection_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write("裂缝检测报告\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"模型路径: {MODEL_PATH}\n")
    f.write(f"输入文件夹: {INPUT_DIR}\n")
    f.write(f"置信度阈值: {CONF_THRESHOLD}\n\n")
    f.write(f"总图片数: {len(image_files)}\n")
    f.write(f"总裂缝数: {total_cracks}\n")
    f.write(f"有裂缝的图片: {images_with_cracks}\n")
    f.write(f"无裂缝的图片: {images_without_cracks}\n\n")
    f.write("详细结果:\n")
    f.write("-" * 50 + "\n")
    for stat in crack_stats:
        f.write(f"{stat['filename']}: {stat['num_cracks']} 个裂缝\n")

print(f"\n📄 报告已保存到: {report_path}")

# 10. 显示示例结果（如果有检测到裂缝）
if images_with_cracks > 0:
    print("\n" + "=" * 50)
    print("检测示例（前3张有裂缝的图片）:")
    print("=" * 50)
    examples = [s for s in crack_stats if s['num_cracks'] > 0][:3]
    for ex in examples:
        print(f"  📷 {ex['filename']}: {ex['num_cracks']} 个裂缝")
        print(f"     💾 {ex['output_path']}")
else:
    print("\n⚠️ 警告: 没有检测到任何裂缝，建议降低置信度阈值")

print("\n✅ 所有任务完成！")