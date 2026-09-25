import cv2
import time
from ultralytics import YOLO
import torch
import os

# 解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    # --- 1. 配置区域（请修改这些路径）---
    # 替换为你的 RK3588 机器人 IP 地址
    ROBOT_IP = "192.168.0.100"  # 改成你机器人的实际IP
    RTSP_URL = f"rtsp://{ROBOT_IP}:8554/live"

    # 使用你训练好的模型路径
    MODEL_PATH = r"D:\.Download\crack-detection\Damage Detection\runs\detect\crack_50_epochs\weights\best.pt"

    # 检测置信度阈值（可以调低一点，让模型更敏感）
    CONF_THRESHOLD = 0.3  # 从0.5降到0.3，因为你的模型mAP还不高

    # 检测类别（0: crack, 1: spalling，如果想只检测裂缝就改成[0]）
    CLASSES = [0, 1]  # 同时检测裂缝和剥落

    # --- 2. 初始化 ---
    print(f"正在加载模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 检查是否在使用 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"推理设备: {device}")
    print(f"模型类别: {model.names}")

    # --- 3. 测试本地摄像头（如果没有RTSP流，可以用这个测试）---
    # 如果暂时没有机器人RTSP流，可以先用本地摄像头测试
    USE_LOCAL_CAM = True  # 改成 False 使用 RTSP
    if USE_LOCAL_CAM:
        print("使用本地摄像头测试...")
        cap = cv2.VideoCapture(0)  # 0 是默认摄像头
        if not cap.isOpened():
            print("本地摄像头打开失败，尝试RTSP...")
            cap = cv2.VideoCapture(RTSP_URL)
    else:
        print(f"正在连接 RTSP 流: {RTSP_URL}...")
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("错误：无法打开视频流，请检查：")
        print("1. 如果使用RTSP，请确认机器人IP和MediaMTX是否运行")
        print("2. 如果使用本地摄像头，请确认摄像头是否连接")
        return

    print("连接成功！正在接收实时画面...")

    # 用于计算 FPS
    prev_time = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("丢失流数据，正在尝试重连...")
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL)
                continue

            # 可选：缩小图片尺寸加速推理（如果需要）
            # frame = cv2.resize(frame, (640, 480))

            # --- 4. YOLO 推理 ---
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                device=device,
                classes=CLASSES,  # 只检测指定的类别
                verbose=False
            )

            # --- 5. 渲染结果 ---
            annotated_frame = results[0].plot()

            # 计算并显示 FPS
            curr_time = time.time()
            if prev_time > 0:
                fps = 1 / (curr_time - prev_time)
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示检测统计
            boxes = results[0].boxes
            if boxes is not None:
                num_detections = len(boxes)
                cv2.putText(annotated_frame, f"Detections: {num_detections}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            prev_time = curr_time

            # --- 6. 显示窗口 ---
            cv2.imshow("Dam Crack Detection - Real-time", annotated_frame)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    # --- 7. 清理资源 ---
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == "__main__":
    main()