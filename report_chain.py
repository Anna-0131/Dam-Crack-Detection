# -*- coding: utf-8 -*-
"""巡检报告生成链 —— 成员A 负责实现真实版本(调 LLM)。

当前为模板版:不调 LLM,按模板生成 Markdown 报告,输出格式与真版一致。
成员A 完成后替换本文件,保持函数签名不变。
"""


def generate_report(detections: list) -> str:
    if not detections:
        return (
            "## 巡检报告\n\n"
            "**检测结果:未发现明显缺陷。**\n\n"
            "结论:坝面外观正常,建议按原计划开展下次巡检。"
        )
    rows = ["| 序号 | 缺陷类型 | 置信度 | 位置(像素坐标) |", "|---|---|---|---|"]
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        rows.append(f"| {i} | {d['class']} | {d['conf']:.2f} | ({x1},{y1})~({x2},{y2}) |")
    return (
        "## 巡检报告\n\n"
        f"本次共检测到 **{len(detections)}** 处疑似缺陷。\n\n"
        + "\n".join(rows)
        + "\n\n> 注:当前为模板报告,成员A 接入 LLM 后自动生成风险评级与整改建议。"
    )


def generate_video_report(samples):
    """视频巡检报告 —— 成员A 后续接 LLM 版本。
    samples: [{"time_sec": 1.0, "detections": [...]}, ...]
    """
    total = len(samples)
    hit = sum(1 for s in samples if s["detections"])
    if hit == 0:
        return ("## 视频巡检报告\n\n"
                f"共抽检 **{total}** 帧(每 1 秒 1 帧),**未发现明显缺陷**。\n\n"
                "结论:视频覆盖坝面区域外观正常,建议按原计划开展下次巡检。")
    rows = ["| 时间(秒) | 检出数量 | 最高置信度 |", "|---|---|---|"]
    for s in samples:
        if not s["detections"]:
            continue
        best = max(d["conf"] for d in s["detections"])
        rows.append(f"| {s['time_sec']} | {len(s['detections'])} | {best:.2f} |")
    return ("## 视频巡检报告\n\n"
            f"共抽检 **{total}** 帧,其中 **{hit}** 帧检出缺陷。\n\n"
            + "\n".join(rows)
            + "\n\n> 注:当前为模板报告,成员A 接入 LLM 后自动生成风险评级与整改建议。")
