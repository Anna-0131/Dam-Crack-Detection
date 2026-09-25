import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# ============================================================
# DamCrack 2 类（crack + spalling）迁移学习训练脚本
#
# 起点：之前训练了 150 轮的模型（DamCrack 裂缝单类）
# 原理：ultralytics 会用 data.yaml 里的 nc=2 重建检测头，
#       骨干网络和颈部权重从旧模型迁移过来，检测头重新初始化。
# ============================================================

# 旧模型（150 轮，DamCrack 裂缝）作为迁移学习起点
model = YOLO(r'D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\best.pt')

model.train(
    # DamCrack 标准数据集（train 3200 / val 800，2 类）
    data=r'D:\.Download\crack-detection\DamCrack_YOLO\data.yaml',

    epochs=100,          # 头部是全新的，需要足够的轮数学剥落类
    imgsz=640,           # 和数据集原始尺寸一致
    batch=2,             # CPU 内存有限
    device='cpu',
    workers=0,

    optimizer='SGD',     # 显式指定，别让 auto 覆盖学习率
    lr0=0.005,           # 头部全新，学习率适当给高一点让它快速学
    lrf=0.01,
    momentum=0.937,
    cos_lr=True,

    patience=20,         # 20 轮没提升就提前停
    save_period=10,      # 每 10 轮保存一次
    plots=True,          # 生成训练曲线
    exist_ok=True,
    name='damcrack_2cls' # 新名字，和旧的区分
)

# 提示：类别不平衡（crack 35515 框 vs spalling 4696 框，约 7.5:1），
# 跑完后重点看 spalling 类的 recall，若偏低可考虑调大 spalling 样本权重。
