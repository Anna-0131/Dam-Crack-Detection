# -*- coding: utf-8 -*-
"""大坝缺陷巡检智能体 —— Web 界面(成员B)。
运行:在项目根目录执行 python web_app.py,浏览器打开 http://127.0.0.1:7860
"""
import os
import tempfile

import gradio as gr

from detector import detect
from report_chain import generate_report, generate_video_report
from qa_chain import answer_question, answer_fallback
from utils import draw_boxes
from video_inspect import process_video

QA_MAX_DETS = 20  # 视频抽帧结果很多,问答只取前 20 条,避免 prompt 过长


def run_inspection(image):
    """图片巡检:检测 → 画框 → 报告。返回(标注图, 报告, 检测结果状态)"""
    if image is None:
        return None, "## 巡检报告\n\n请先上传图片。", None
    tmp_path = os.path.join(tempfile.gettempdir(), "dam_agent_upload.jpg")
    image.save(tmp_path)
    detections = detect(tmp_path)
    annotated = draw_boxes(image, detections)
    report = generate_report(detections)
    return annotated, report, detections


def run_video_inspection(video_path):
    """视频巡检:抽帧检测 → 输出标注视频 + 报告。"""
    if not video_path:
        return None, "## 巡检报告\n\n请先上传视频。", None
    out_video, samples = process_video(video_path, detect)
    report = generate_video_report(samples)
    flat = [d for s in samples for d in s["detections"]][:QA_MAX_DETS]
    return out_video, report, flat


def chat_respond(question, history, detections):
    """问答回调。detections 为 None 表示还没巡检;[] 表示巡检过但无缺陷。"""
    history = history or []
    question = (question or "").strip()
    if not question:
        return history, ""
    if detections is None:
        answer = "请先上传图片或视频并点击巡检按钮,再针对巡检结果提问。"
    else:
        try:
            answer = answer_question(question, detections, history)
        except Exception:
            answer = answer_fallback(question, detections)  # 离线兜底,不白屏
    history.append((question, answer))
    return history, ""


with gr.Blocks(title="大坝缺陷巡检智能体") as demo:
    gr.Markdown("# 大坝缺陷巡检智能体")
    detections_state = gr.State(None)  # 图片/视频共用,存最近一次检测结果

    with gr.Tab("图片巡检"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="上传巡检照片")
                inspect_btn = gr.Button("开始巡检", variant="primary")
            with gr.Column():
                output_img = gr.Image(type="pil", label="检测结果")
                report_md = gr.Markdown("等待巡检…")
        inspect_btn.click(run_inspection, inputs=[input_img],
                          outputs=[output_img, report_md, detections_state])

    with gr.Tab("视频巡检"):
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="上传巡检视频")
                video_btn = gr.Button("开始视频巡检", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="检测结果(已标注)")
                video_report_md = gr.Markdown("等待巡检…")
        video_btn.click(run_video_inspection, inputs=[input_video],
                        outputs=[output_video, video_report_md, detections_state])

    gr.Markdown("### 针对最近一次巡检结果问答")
    chatbot = gr.Chatbot(label="巡检问答", height=300)
    with gr.Row():
        question_box = gr.Textbox(label="问题",
                                  placeholder="例如:裂缝大概多长?整改建议是什么?")
        ask_btn = gr.Button("提问")
    ask_btn.click(chat_respond, inputs=[question_box, chatbot, detections_state],
                  outputs=[chatbot, question_box])
    question_box.submit(chat_respond, inputs=[question_box, chatbot, detections_state],
                        outputs=[chatbot, question_box])

demo.launch()
