# -*- coding: utf-8 -*-
"""问答链 —— 成员B。把检测结果 + 用户问题交给 DeepSeek,回答基于检测数据。"""
import json
from llm import chat
from utils import estimate_length_mm, PIXEL_TO_MM

SYSTEM_PROMPT = (
    "你是一名大坝混凝土结构巡检专家助理。用户会就本次巡检的检测结果提问。\n"
    "你会收到检测结果 JSON,其中 bbox_xyxy 为像素坐标 [x1, y1, x2, y2]。\n"
    "回答规则:\n"
    f"1. 裂缝长度估算 = 框对角线像素数 × 换算系数 {PIXEL_TO_MM} 毫米/像素"
    "(假设拍摄距离约 1 米)。回答时必须给出估算值,并说明这是基于假设的估算。\n"
    "2. 回答简洁专业;涉及风险时给出整改建议。\n"
    "3. 检测数据里没有的信息不要编造,不确定就说需要人工复核。\n"
)


def answer_question(question: str, detections: list, history: list = None) -> str:
    """基于检测结果回答。失败时抛异常,由调用方兜底。"""
    content = f"本次检测结果(JSON):\n{json.dumps(detections, ensure_ascii=False)}"
    if history:
        turns = [f"用户:{q}\n助手:{a}" for q, a in history[-4:]]  # 最近4轮
        content += "\n\n对话历史:\n" + "\n".join(turns)
    content += f"\n\n用户当前问题:{question}"
    return chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ])


def answer_fallback(question: str, detections: list) -> str:
    """DeepSeek 不可用时的规则兜底,保证界面不白屏。"""
    q = question or ""
    if not detections:
        return "本次检测未发现缺陷,无需整改建议。"
    if "长" in q or "多大" in q:
        parts = []
        for i, d in enumerate(detections, 1):
            mm = estimate_length_mm(d["bbox_xyxy"])
            parts.append(f"第{i}处裂缝长度约 {mm:.0f} 毫米(按拍摄距离约1米估算)")
        return "、".join(parts) + "。精确值需现场标定复核。"
    if "几" in q or "多少" in q or "数量" in q:
        return f"共检测到 {len(detections)} 处缺陷。"
    if "类" in q or "什么" in q:
        return "缺陷类型:" + "、".join(sorted({d["class"] for d in detections})) + "。"
    return "当前为离线兜底回答(LLM 不可用)。可尝试问:裂缝多长、几处缺陷、什么类型。"
