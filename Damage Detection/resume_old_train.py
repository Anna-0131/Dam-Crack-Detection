import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# ============================================================
# 断点续训旧模型（方式 A）
# 作用：旧模型 crack_150_epochs 原本 epochs=150，但只跑到第 109 轮就中断了，
#       本脚本从 last.pt 接着跑完剩下的约 41 轮（109 → 150）。
#
# resume=True 会读取原训练目录里的 args.yaml，自动继承当时的全部配置
#   （数据、imgsz=640、batch、学习率调度等），并恢复优化器状态，
#   是无缝继续，不是从头开始。
# ============================================================

model = YOLO(r'D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\last.pt')

model.train(resume=True)

# 注意：
# 1. 续训完成后，新权重仍保存在 crack_150_epochs/weights/ 下（best.pt / last.pt 会被更新）。
# 2. 旧数据集只有裂缝图、没有负样本，所以这条路只能小幅提高裂缝召回率，
#    不会减少"干净钢板上误报"。
# 3. CPU 约 18 分钟/轮，补完剩余 41 轮大约需要 12 小时，可用 patience 提前停。
