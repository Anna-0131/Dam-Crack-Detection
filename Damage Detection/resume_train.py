import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# 从中断时的最后检查点继续训练（不会从头开始）
# last.pt 记录了已完成到第 6 轮，resume=True 会从第 7 轮接着跑完剩下的 24 轮
model = YOLO(r'D:\.Download\crack-detection\runs\detect\crack_finetune_combined\weights\last.pt')

model.train(resume=True)
