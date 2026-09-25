import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

model = YOLO(r'D:\.Download\crack-detection\Damage Detection\runs\detect\crack_50_epochs\weights\best.pt')

model.train(
    data=r'D:\.Download\crack-detection\Damage Detection\Damage_Detection_YOLO\data.yaml',
    epochs=150,
    imgsz=640,
    batch=2,
    device='cpu',
    workers=0,

    patience=20,  # 如果20轮没提升就自动停止，节省时间
    save_period=10,  # 每10轮保存一次模型
    plots=True,  # 自动生成训练曲线图
    exist_ok=True,  # 允许覆盖之前的训练结果
    name='crack_150_epochs'  # 给这次训练起个名字，方便区分
)