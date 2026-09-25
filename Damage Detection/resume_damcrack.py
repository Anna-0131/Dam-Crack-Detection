import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# ============================================================
# 续训 DamCrack 2 类模型
# 作用：从上次中断的 last.pt 接着跑（上次停在 epoch 5，本次从 6 继续到 100）。
# resume=True 会读原训练目录的 args.yaml，自动恢复优化器状态和学习率调度，
# 并继续写入同一个 runs/detect/damcrack_2cls 目录，不会覆盖从头训练。
# ============================================================

model = YOLO(r'D:\.Download\crack-detection\runs\detect\damcrack_2cls\weights\last.pt')

model.train(resume=True)
