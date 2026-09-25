# -*- coding: utf-8 -*-
"""视频巡检 —— 成员B。抽帧 → 逐帧检测 → 输出标注视频 + 每帧检测结果。"""
import os
import tempfile

import cv2

SAMPLE_INTERVAL_SEC = 1  # 每 1 秒抽 1 帧做检测(CPU 机器,演示节奏够用)


def process_video(video_path, detect_fn):
    """对视频抽帧检测,返回 (标注视频路径, 采样结果列表)。
    采样结果:[{"time_sec": 1.0, "detections": [...]}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件:{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every = max(1, int(fps * SAMPLE_INTERVAL_SEC))

    out_path = os.path.join(tempfile.gettempdir(), "dam_agent_video_out.mp4")
    writer = None
    for fourcc in ("avc1", "mp4v"):  # avc1(H.264)浏览器兼容最好,失败退回 mp4v
        w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                            fps, (width, height))
        if w.isOpened():
            writer = w
            break
    if writer is None:
        cap.release()
        raise RuntimeError("无法创建视频输出文件(编码器不可用)")

    samples = []
    last_dets = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:  # 采样帧:落盘 → 检测
            tmp_img = os.path.join(tempfile.gettempdir(), "dam_agent_frame.jpg")
            cv2.imwrite(tmp_img, frame)
            last_dets = detect_fn(tmp_img)
            samples.append({"time_sec": round(frame_idx / fps, 1),
                            "detections": last_dets})
        for d in last_dets:  # 中间帧沿用最近一次采样结果画框
            x1, y1, x2, y2 = d["bbox_xyxy"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{d['class']} {d['conf']:.2f}",
                        (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    if not samples:
        raise RuntimeError("视频里没有读到任何帧")
    return out_path, samples
