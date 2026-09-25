import os
from pathlib import Path

import torch
from ultralytics import YOLO

# 解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==================== 配置区域（按需修改）====================
# 训练好的模型（合并数据集微调后的新模型）
MODEL_PATH = r'D:\.Download\crack-detection\runs\detect\crack_finetune_combined\weights\best.pt'

# 输入：可以是文件夹（批量测试），也可以是单张图片路径
INPUT_PATH = r'D:\.Download\crack-detection\test'

# 输出：标注后的结果图保存到这里
OUTPUT_DIR = r'D:\.Download\crack-detection\prediction_results'

# 置信度阈值：低于这个值的检测框会被过滤掉（模型 mAP 不高，可适当调低）
CONF_THRESHOLD = 0.3

# 是否逐张弹出窗口预览结果（True=预览，按任意键看下一张；False=只保存不弹窗）
SHOW_RESULTS = False

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# ============================================================


def collect_images(input_path: Path) -> list[Path]:
    """输入是文件夹就收集里面所有图片，是单张图片就只返回这一张。"""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f'输入路径不存在: {input_path}')

    images = []
    for ext in IMAGE_EXTS:
        images.extend(input_path.glob(f'*{ext}'))
        images.extend(input_path.glob(f'*{ext.upper()}'))
    return sorted(set(images))


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'❌ 模型不存在: {MODEL_PATH}')
        return

    # 收集待测图片
    try:
        images = collect_images(Path(INPUT_PATH))
    except FileNotFoundError as e:
        print(f'❌ {e}')
        return

    if not images:
        print(f'❌ 在 {INPUT_PATH} 中没有找到图片')
        return

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 50)
    print(f'模型: {MODEL_PATH}')
    model = YOLO(MODEL_PATH)
    print(f'类别: {model.names}')
    print(f'推理设备: {device}')
    print(f'置信度阈值: {CONF_THRESHOLD}')
    print(f'待测图片: {len(images)} 张')
    print('=' * 50)

    # 创建输出目录
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 批量推理
    results_summary = []  # (文件名, 检测框数量)
    for i, img_path in enumerate(images, 1):
        results = model(img_path, conf=CONF_THRESHOLD, device=device, verbose=False)
        boxes = results[0].boxes
        num = len(boxes) if boxes is not None else 0

        # 保存标注结果
        save_path = out_dir / f'result_{img_path.name}'
        results[0].save(filename=str(save_path))

        results_summary.append((img_path.name, num))
        print(f'[{i}/{len(images)}] {img_path.name}: {num} 个检测框 -> {save_path.name}')

        if SHOW_RESULTS:
            import cv2
            annotated = results[0].plot()
            cv2.imshow('crack detection', annotated)
            cv2.waitKey(0)

    if SHOW_RESULTS:
        import cv2
        cv2.destroyAllWindows()

    # 汇总统计
    total = sum(n for _, n in results_summary)
    hit = sum(1 for _, n in results_summary if n > 0)
    print()
    print('=' * 50)
    print('检测汇总')
    print('=' * 50)
    print(f'总图片数: {len(images)}')
    print(f'检出目标的图片: {hit} 张')
    print(f'总检出框数: {total}')
    print(f'结果已保存到: {out_dir}')


if __name__ == '__main__':
    main()
