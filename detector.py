# -*- coding: utf-8 -*-
"""缺陷检测模块 —— 成员A 实现真实版本。

实现方式:把图片 POST 给组长的 YOLO HTTP 检测服务(detect_server.py),
服务返回 {detections, summary, report, annotated_image_base64},本模块把
服务返回结果转换成成员B 契约要求的格式。

对外只暴露 detect(image_path) -> list[dict],元素形如:
    {"class": "crack", "conf": 0.87, "bbox_xyxy": [x1, y1, x2, y2]}
无缺陷返回 []。

组长的服务返回每个框含 class / class_id / confidence / bbox_xyxy 四个字段,
其中契约只认 class / conf / bbox_xyxy,因此这里做两处转换:
    1. confidence -> conf(重命名)
    2. 去掉 class_id
另外把 bbox 坐标转成 int:成员B 的视频巡检用 cv2.rectangle 画框,
而 cv2 不接受浮点坐标,服务返回的却是 round(x,1) 的浮点数,此处统一取整。
"""
import os

import requests
from dotenv import load_dotenv

# 读取项目根目录 .env(服务地址等配置放这里,不要写死在代码里)
load_dotenv()

# 组长检测服务地址。优先取环境变量 DETECT_SERVER_URL(可在 .env 里配置)。
# 默认值仅为兜底,联调时请在 .env 写入实际可用的地址。
SERVER_URL = os.environ.get("DETECT_SERVER_URL", "http://10.22.28.46:8000")

# 请求超时(秒)。服务端首次请求含模型加载,给宽松一点;联调若嫌慢可调小。
TIMEOUT = 60


def detect(image_path: str) -> list:
    """上传图片到组长检测服务,返回契约格式的检测结果列表;无缺陷返回 []。

    Args:
        image_path: 图片文件路径(支持中文路径)

    Returns:
        [{"class": str, "conf": float, "bbox_xyxy": [int, int, int, int]}, ...]

    Raises:
        请求失败或服务返回异常时抛异常(不吞异常,由界面层兜底)。
    """
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{SERVER_URL}/detect",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=TIMEOUT,
        )
    resp.raise_for_status()  # 网络/服务出错直接抛,交给界面兜底
    data = resp.json()

    result = []
    for det in data.get("detections", []):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        result.append({
            "class": det["class"],
            "conf": float(det["confidence"]),
            "bbox_xyxy": [int(round(x1)), int(round(y1)),
                          int(round(x2)), int(round(y2))],
        })
    return result


def annotate(image_path: str) -> dict:
    """可选:返回组长服务原始结果(含标注图 base64 与文本摘要),备查用。

    成员B 的界面自己用 detect() 的坐标画框,不依赖本函数;
    保留它是为了将来需要组长原生标注图时有现成入口。
    """
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{SERVER_URL}/detect",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=TIMEOUT,
        )
    resp.raise_for_status()
    return resp.json()
