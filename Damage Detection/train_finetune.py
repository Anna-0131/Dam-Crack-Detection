import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# 从现有最优模型继续训练（微调），而不是从头训练
model = YOLO(r'D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\best.pt')

model.train(
    # 合并后的数据集（1070 裂缝 + 2000 无裂缝）
    data=r'D:\.Download\crack-detection\combined_dataset_YOLO\data.yaml',
    epochs=30,        # 微调不需要太多轮
    imgsz=512,        # 从 640 降到 512，提速约 40%，精度损失很小
    batch=2,          # CPU 内存有限，保持小批量
    device='cpu',
    workers=0,

    lr0=0.001,        # 微调关键：学习率降到默认(0.01)的 1/10，避免破坏已学到的特征
    lrf=0.01,
    cos_lr=True,      # 余弦退火，微调更平稳
    # freeze=10,      # 可选：冻结前 10 层主干网络，小数据防过拟合（数据够多可不加）

    patience=10,      # 10 轮没提升就提前停止
    save_period=10,
    plots=True,       # 生成训练曲线
    exist_ok=True,
    name='crack_finetune_combined'  # 新模型名字，方便和旧的区分
)
