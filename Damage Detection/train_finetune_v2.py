import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# ============================================================
# 微调脚本 v2（修正版）
# 上次失败原因：没写 optimizer，默认 optimizer='auto' 把你写的
#   lr0=0.001 覆盖成了 AdamW lr=0.002，学习率过高导致灾难性遗忘。
# 本次关键修复：显式 optimizer='SGD' + freeze=10 冻结主干 + imgsz 回到 640。
#
# 注意：从【旧的强模型】重新开始，不要从上次退化后的模型继续。
# ============================================================

# 基础模型 = 150 轮训练出来的那个好模型（mAP50 ≈ 0.42）
model = YOLO(r'D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\best.pt')

model.train(
    # 合并后的数据集（1070 裂缝 + 2000 无裂缝）
    data=r'D:\.Download\crack-detection\combined_dataset_YOLO\data.yaml',

    epochs=50,          # 比上次多一点，给够收敛时间
    imgsz=640,          # 关键：回到 640，别降，小裂纹需要像素细节
    batch=2,            # CPU 内存有限，保持小批量
    device='cpu',
    workers=0,

    optimizer='SGD',    # 关键：显式指定，否则 auto 会覆盖你的 lr0
    lr0=0.001,          # 关键：默认(0.01)的 1/10；更保守可试 0.0005
    lrf=0.01,
    momentum=0.937,     # SGD 默认动量
    cos_lr=True,        # 余弦退火，微调更平稳

    freeze=10,          # 关键：冻结前 10 层主干，防止遗忘旧特征

    patience=15,        # 15 轮没提升就提前停止
    save_period=10,     # 每 10 轮保存一次
    plots=True,         # 生成训练曲线
    exist_ok=True,
    name='crack_finetune_v2'   # 新名字，和上次区分
)

# 提醒：这次验证集里混入了大量负样本，mAP50 数值会天然比纯裂缝验证集低，
# 目标不是追高 mAP 数字，而是"干净钢板上的误报变少"。跑完后重点看：
#   1. metrics/precision(B) 有没有比上次高
#   2. 在真实测试图上的漏检是否可接受
