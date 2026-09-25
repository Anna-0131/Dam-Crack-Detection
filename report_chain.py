# -*- coding: utf-8 -*-
"""巡检报告生成链 —— 成员A 实现真实版本(调 LLM)。

对外暴露两个函数(签名与契约 INTERFACE.md 一致):
    generate_report(detections) -> str          # 图片报告
    generate_video_report(samples) -> str        # 视频报告

实现方式:检测结果交给 DeepSeek(经共享的 llm.chat)生成 Markdown 报告。
降级策略:LLM 调用失败时不抛异常、不白屏,自动退回规则模板报告
(与成员B 的 qa_chain.answer_fallback 风格一致)。
"""
from datetime import date

from llm import chat

# 报告日期(调用时取当天)
DATE_STR = date.today().isoformat()

# 类别中文名,供模板报告显示
CLASS_NAME_CN = {"crack": "裂缝", "spalling": "剥落"}


# ---------------------------------------------------------------------------
# 规则模板(LLM 不可用时的兜底,保证界面始终有报告可显示)
# ---------------------------------------------------------------------------
def _template_report(detections: list) -> str:
    if not detections:
        return (
            "## 巡检报告\n\n"
            "**检测结果:未发现明显缺陷。**\n\n"
            "结论:坝面外观正常,建议按原计划开展下次巡检。"
        )
    rows = ["| 序号 | 缺陷类型 | 置信度 | 位置(像素坐标) |", "|---|---|---|---|"]
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        name = CLASS_NAME_CN.get(d["class"], d["class"])
        rows.append(f"| {i} | {name}({d['class']}) | {d['conf']:.2f} | ({x1},{y1})~({x2},{y2}) |")
    return (
        "## 巡检报告\n\n"
        f"本次共检测到 **{len(detections)}** 处疑似缺陷。\n\n"
        + "\n".join(rows)
        + "\n\n> 注:当前为离线模板报告(LLM 暂时不可用),未包含风险评级与整改建议,请稍后重试或人工判读。"
    )


def _template_video_report(samples: list) -> str:
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
            + "\n\n> 注:当前为离线模板报告(LLM 暂时不可用),未包含风险评级与整改建议。")


# ---------------------------------------------------------------------------
# 喂给 LLM 的文本构造
# ---------------------------------------------------------------------------
def _detection_text(detections: list) -> str:
    if not detections:
        return "本次巡检未检测到明显缺陷。"
    lines = []
    for i, d in enumerate(detections, 1):
        name = CLASS_NAME_CN.get(d["class"], d["class"])
        x1, y1, x2, y2 = d["bbox_xyxy"]
        lines.append(
            f"{i}. 类别:{name}({d['class']});置信度:{d['conf']:.2f};"
            f"位置(像素):[{x1},{y1},{x2},{y2}];像素宽:{x2 - x1};像素高:{y2 - y1}。"
        )
    return "\n".join(lines)


REPORT_PROMPT = """你是一名大坝混凝土结构巡检专家。请根据以下 YOLO 检测结果,生成一份标准化 Markdown 巡检报告。

今天是 {today}。请在报告开头写明报告日期,不要写"待填写"或臆造其他日期。

要求:
- 用 Markdown 格式,标题层级清晰、序号连续(不要出现两个"三")
- 包含以下部分:巡检概况、缺陷列表、风险等级(低/中/高)、整改建议、是否需要人工复核
- 风险等级要给出判断依据;置信度偏低或检测框重叠时要指出不确定性
- 不要编造检测结果里没有的信息;像素尺寸不等于实际物理尺寸,不要臆造真实长度
- 只输出报告正文,不要多余的前言或解释

检测结果:
{detection_text}
"""

VIDEO_PROMPT = """你是一名大坝混凝土结构巡检专家。以下是一次视频巡检的抽帧检测结果(每 {interval} 秒抽 1 帧),巡检日期 {today}。

要求:
- 用 Markdown 格式,标题层级清晰、序号连续
- 包含:抽检概况(抽检帧数、检出帧数)、缺陷时间分布、风险等级(低/中/高)、整改建议
- 结合缺陷在时间上的分布判断是持续存在还是偶发
- 不要编造检测结果里没有的信息

抽帧检测结果:
{samples_text}
"""


# ---------------------------------------------------------------------------
# 对外函数
# ---------------------------------------------------------------------------
def generate_report(detections: list) -> str:
    """输入 detect() 的结果,返回 Markdown 巡检报告。LLM 失败时降级为模板报告。"""
    prompt = REPORT_PROMPT.format(today=DATE_STR, detection_text=_detection_text(detections))
    try:
        return chat([{"role": "user", "content": prompt}])
    except Exception as e:
        # 降级:不白屏,退回模板报告
        print(f"[report_chain] LLM 调用失败,降级为模板报告:{type(e).__name__}: {e}")
        return _template_report(detections)


def generate_video_report(samples: list) -> str:
    """输入抽帧检测结果 [{"time_sec": 1.0, "detections": [...]}, ...],返回 Markdown 视频报告。

    LLM 失败时降级为模板报告。
    """
    if not samples:
        return _template_video_report(samples)

    lines = []
    for s in samples:
        dets = s["detections"]
        if dets:
            desc = ";".join(
                f"{CLASS_NAME_CN.get(d['class'], d['class'])} conf={d['conf']:.2f} bbox={d['bbox_xyxy']}"
                for d in dets
            )
            lines.append(f"{s['time_sec']}s:检出 {len(dets)} 处 —— {desc}")
        else:
            lines.append(f"{s['time_sec']}s:未检出")
    samples_text = "\n".join(lines)

    prompt = VIDEO_PROMPT.format(interval=1, today=DATE_STR, samples_text=samples_text)
    try:
        return chat([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"[report_chain] 视频报告 LLM 调用失败,降级为模板报告:{type(e).__name__}: {e}")
        return _template_video_report(samples)
