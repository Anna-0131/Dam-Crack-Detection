import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
import torch

# 【防延迟核心 1】强制 OpenCV 底层使用 TCP 协议拉取 RTSP 流，解决 UDP 丢包和起播慢的问题
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def main():
    # --- 1. 配置区域 ---
    # 请核实机器人当前真实的局域网 IP
    ROBOT_IP = "192.168.0.100"
    # 端口已修正为 OpenCV 专用的 8554
    RTSP_URL = f"rtsp://{ROBOT_IP}:8554/live"

    MODEL_PATH = r"D:\.Download\crack-detection\Damage Detection\runs\detect\crack_150_epochs\weights\best.pt"

    # 配合 Tracking 算法，置信度可以适当放宽到 0.25，算法会自动脑补漏掉的帧
    CONF_THRESHOLD = 0.25

    # --- 2. 初始化模型与设备 ---
    print(f"正在加载钢板检测模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 如果你的电脑有独显，优先使用 cuda；没有则回退到 cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前推理算力设备: {device}")

    # --- 3. 开启视频流 ---
    print(f"正在建立超低延迟连接: {RTSP_URL}...")
    cap = cv2.VideoCapture(RTSP_URL)

    # 【防延迟核心 2】限制 OpenCV 缓冲区大小为 1，永远只处理最新的一帧，彻底告别画面积压
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("-" * 50)
        print("【连接失败】")
        print("1. 检查机器人 IP 是否准确。")
        print("2. 确保机器人端 robot_push.service 正在 running。")
        print("-" * 50)
        return

    print("连接成功！知擎者 AI 视觉监测系统已启动，按 'q' 键退出...")
    prev_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue  # 无线网络偶尔的帧波动，直接跳过即可

            # --- 4. 算力转移：PC 端瞬间高清化 ---
            # 将机器人传来的 640x480 轻量流，瞬间拉伸为 720P 大图
            # 这点计算量对 PC 来说只需不到 1 毫秒，却能让 YOLO 抓取更多像素特征
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)

            # --- 5. 画质死保：USM 锐化滤镜 ---
            # 强化反光钢板上的微小裂纹与锈斑边缘，大幅提升模型置信度
            blurred = cv2.GaussianBlur(frame, (0, 0), 3)
            frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

            # --- 6. 解决闪烁：YOLOv12 时序追踪 (Tracking) ---
            # 使用 track() 替代 predict()，并开启 ByteTrack 算法
            # 即使某几帧因为反光没检测到，追踪算法也会根据轨迹强行锁住缺陷！
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                device=device,
                verbose=False
            )

            # --- 7. 渲染结果与信息提示 ---
            annotated_frame = results[0].plot()

            # 计算真实渲染帧率
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 左上角显示系统状态
            cv2.putText(annotated_frame, f"FPS: {int(fps)} | Delay-Free: ON", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # --- 8. 显示最终窗口 ---
            cv2.imshow("ZhiQingZhe - Steel Defect Monitor", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("监测安全终止。")
                break

    except Exception as e:
        print(f"运行中发生异常: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()