"""大坝缺陷巡检智能体 - Gradio 前端(成员B)

Monochrome 编辑杂志风:黑白配色、衬线大标题、等宽小标签、
方角、粗黑线、黑白反转悬停。仅样式变化,全部功能保持不变。
"""

import os
import tempfile

import gradio as gr
import requests
from PIL import Image

from detector import SERVER_URL, detect

try:
    from detector import get_heatmap
except ImportError:
    get_heatmap = None

from report_chain import generate_report, generate_video_report
from qa_chain import answer_question, answer_fallback
from utils import draw_boxes
from video_inspect import process_video

# 问答上下文最多带多少条检测结果
QA_MAX_DETS = 20

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ========== 全局:强制黑白浅色主题(覆盖系统暗色模式) ========== */
body {
  background: #FFFFFF !important;
  color: #000000 !important;
  --body-background-fill: transparent !important;
  --body-text-color: #000000 !important;
  --body-text-color-subdued: #525252 !important;
  --color-accent: #000000 !important;
  --color-accent-soft: #F5F5F5 !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #FFFFFF !important;
  --background-fill-tertiary: #F5F5F5 !important;
  --border-color-primary: #000000 !important;
  --border-color-accent: #000000 !important;
  --block-background-fill: #FFFFFF !important;
  --block-border-color: #000000 !important;
  --block-label-text-color: #000000 !important;
  --block-title-text-color: #000000 !important;
  --block-info-text-color: #525252 !important;
  --block-radius: 0px !important;
  --panel-background-fill: #FFFFFF !important;
  --panel-border-color: #000000 !important;
  --panel-border-width: 2px !important;
  --button-primary-background-fill: #000000 !important;
  --button-primary-text-color: #FFFFFF !important;
  --button-primary-border-color: #000000 !important;
  --button-primary-background-fill-hover: #FFFFFF !important;
  --button-primary-text-color-hover: #000000 !important;
  --button-secondary-background-fill: #FFFFFF !important;
  --button-secondary-text-color: #000000 !important;
  --button-secondary-border-color: #000000 !important;
  --button-secondary-background-fill-hover: #000000 !important;
  --button-secondary-text-color-hover: #FFFFFF !important;
  --button-border-width: 2px !important;
  --button-radius: 0px !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: #000000 !important;
  --input-text-color: #000000 !important;
  --input-placeholder-color: #525252 !important;
  --input-radius: 0px !important;
  --link-text-color: #000000 !important;
  --table-odd-background-fill: #FFFFFF !important;
  --table-even-background-fill: #F5F5F5 !important;
}

/* ========== 纸张质感:细横线 + 噪点 ========== */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image: repeating-linear-gradient(0deg, transparent, transparent 1px, #000 1px, #000 2px);
  background-size: 100% 4px;
  opacity: 0.015;
}
body::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  opacity: 0.02;
}
.gradio-container {
  position: relative;
  z-index: 1;
  max-width: 1200px;
  background: transparent;
}
footer { display: none !important; }

/* ========== 字体:衬线标题 + 等宽标签 ========== */
.prose, .markdown {
  color: #000000 !important;
  font-family: Georgia, "Times New Roman", "Songti SC", SimSun, serif;
}
.prose h1, .prose h2, .prose h3, .prose h4,
.markdown h1, .markdown h2, .markdown h3, .markdown h4 {
  font-family: "Playfair Display", Georgia, "STZhongsong", "Songti SC", SimSun, serif !important;
  color: #000000 !important;
  font-weight: 700;
}
.prose h3, .markdown h3 {
  border-left: 4px solid #000000;
  padding-left: 0.75rem;
  margin-top: 1.5rem;
}
.prose code, .markdown code {
  font-family: "JetBrains Mono", Consolas, monospace;
  background: #F5F5F5;
  border-radius: 0;
}
.hint { font-style: italic; color: #525252 !important; }

/* ========== 标签:等宽小字 ========== */
.gradio-container label, .gradio-container label span {
  font-family: "JetBrains Mono", Consolas, monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #000000 !important;
  font-weight: 600;
}

/* ========== 按钮:方角 + 2px 黑边 + 悬停黑白反转 ========== */
.gradio-container button {
  border-radius: 0 !important;
  border: 2px solid #000000 !important;
  font-family: "JetBrains Mono", Consolas, monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  transition: background 0.12s ease, color 0.12s ease;
}
.gradio-container button.primary {
  background: #000000 !important;
  color: #FFFFFF !important;
}
.gradio-container button.primary:hover {
  background: #FFFFFF !important;
  color: #000000 !important;
}
.gradio-container button.secondary {
  background: #FFFFFF !important;
  color: #000000 !important;
}
.gradio-container button.secondary:hover {
  background: #000000 !important;
  color: #FFFFFF !important;
}

/* ========== 卡片:白底 2px 黑边 方角,悬停印刷硬投影 ========== */
.card {
  background: #FFFFFF !important;
  border: 2px solid #000000 !important;
  border-radius: 0 !important;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.card:hover {
  transform: translate(-2px, -2px);
  box-shadow: 6px 6px 0 #000000;
}

/* ========== 图片预览限高 + 小图标按钮细化 ========== */
#upload_img img, #output_img img, #heatmap_img img {
  max-height: 320px !important;
  object-fit: contain !important;
}
#upload_img button, #output_img button, #heatmap_img button {
  border-width: 1px !important;
  font-size: 0.7rem !important;
}
.image-container .icon-buttons button {
  border: 1px solid #000000 !important;
  background: rgba(255, 255, 255, 0.9) !important;
  color: #000000 !important;
  padding: 2px 6px !important;
}

/* ========== 问答气泡:去掉 Gradio 默认黄色底,统一白底 ========== */
#chatbot .user-row, #chatbot .user-row * {
  background-color: #FFFFFF !important;
  background-image: none !important;
}

/* ========== 标签页:方角按钮,选中项黑白反转 ========== */
.tab-nav {
  background: #FFFFFF !important;
  border-bottom: 4px solid #000000 !important;
}
.tab-nav button {
  border: 2px solid #000000 !important;
  border-radius: 0 !important;
  margin-right: 6px;
  background: #FFFFFF !important;
  color: #000000 !important;
  font-family: "JetBrains Mono", Consolas, monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.tab-nav button.selected, .tab-nav button:hover {
  background: #000000 !important;
  color: #FFFFFF !important;
}

/* ========== 下载组件:强制白底黑字(防暗色模式黑条) ========== */
#report_file, #video_report_file { background: #FFFFFF !important; }
#report_file *, #video_report_file * {
  color: #000000 !important;
  background-color: #FFFFFF !important;
  background-image: none !important;
}

/* ========== 滚动条 ========== */
* { scrollbar-color: #000000 #F5F5F5; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #F5F5F5; }
::-webkit-scrollbar-thumb { background: #000000; border: 2px solid #F5F5F5; }

/* ========== 入场动画 ========== */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate { animation: fadeUp 0.5s ease-out both; }
"""

HERO_HTML = """
<div style="text-align:center;padding:3.5rem 1rem 3rem;border-bottom:4px solid #000;background:#FFF;">
  <div style="display:inline-flex;align-items:center;gap:.6rem;font-family:Consolas,monospace;font-size:.72rem;letter-spacing:.15em;color:#000;margin-bottom:2rem;">
    <span style="width:8px;height:8px;background:#000;display:inline-block;"></span>
    DAM INSPECTION · VOL.1 · 基建巡检
  </div>
  <div style="font-family:Georgia,'STZhongsong','Songti SC',SimSun,serif;font-weight:900;font-size:clamp(2.4rem,6vw,4.2rem);letter-spacing:-.02em;line-height:1.15;color:#000;">
    大坝缺陷巡检智能体
  </div>
  <div style="font-family:Consolas,monospace;font-size:.72rem;letter-spacing:.4em;color:#525252;margin-top:1rem;">
    DAM CRACK DETECTION AGENT
  </div>
  <div style="display:flex;align-items:center;gap:1rem;max-width:30rem;margin:1.8rem auto;">
    <span style="flex:1;height:2px;background:#000;"></span>
    <span style="width:12px;height:12px;border:2px solid #000;"></span>
    <span style="flex:1;height:2px;background:#000;"></span>
  </div>
  <div style="font-family:Georgia,'Songti SC',SimSun,serif;font-style:italic;font-size:1.02rem;color:#000;">
    上传坝面图像 → 自动识别缺陷 → 生成巡检报告 → 智能问答
  </div>
  <div style="font-family:Consolas,monospace;font-size:.68rem;letter-spacing:.15em;color:#525252;margin-top:2rem;">
    EST. 2026 · 检测即报告 · PRINTED IN BLACK &amp; WHITE
  </div>
</div>
"""

SERVER_ONLINE = False


def _check_server():
    """启动时探测一次检测服务,决定统计卡状态"""
    global SERVER_ONLINE
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        SERVER_ONLINE = r.status_code == 200
    except Exception:
        SERVER_ONLINE = False


def _stats_html(count, detections, online):
    """黑色反转统计带:检测次数 / 检出缺陷 / 服务状态"""
    n = len(detections) if detections else 0
    if online:
        status = '<span style="display:inline-block;width:10px;height:10px;background:#16a34a;margin-right:8px;"></span>在线'
    else:
        status = '<span style="display:inline-block;width:10px;height:10px;background:#dc2626;margin-right:8px;"></span>离线'
    return f"""
    <div class="animate" style="display:grid;grid-template-columns:repeat(3,1fr);background:#000;color:#FFF;border-top:4px solid #000;border-bottom:4px solid #000;margin-top:1.5rem;">
      <div style="padding:1.8rem 1rem;text-align:center;border-right:1px solid rgba(255,255,255,.25);color:#FFF;">
        <div style="font-family:Georgia,serif;font-weight:900;font-size:2.6rem;line-height:1;color:#FFF;">{count}</div>
        <div style="font-family:Consolas,monospace;font-size:.68rem;letter-spacing:.18em;margin-top:.6rem;opacity:.7;color:#FFF;">检测次数 · RUNS</div>
      </div>
      <div style="padding:1.8rem 1rem;text-align:center;border-right:1px solid rgba(255,255,255,.25);color:#FFF;">
        <div style="font-family:Georgia,serif;font-weight:900;font-size:2.6rem;line-height:1;color:#FFF;">{n}</div>
        <div style="font-family:Consolas,monospace;font-size:.68rem;letter-spacing:.18em;margin-top:.6rem;opacity:.7;color:#FFF;">检出缺陷 · DEFECTS</div>
      </div>
      <div style="padding:1.8rem 1rem;text-align:center;color:#FFF;">
        <div style="font-family:Georgia,serif;font-weight:700;font-size:1.4rem;line-height:1;display:flex;align-items:center;justify-content:center;color:#FFF;">{status}</div>
        <div style="font-family:Consolas,monospace;font-size:.68rem;letter-spacing:.18em;margin-top:.6rem;opacity:.7;color:#FFF;">检测服务 · SERVER</div>
      </div>
    </div>
    """


def _save_report(report, filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def run_inspection(image_path, count):
    """图片巡检:检测 → 画框 → 热力图 → 报告"""
    count = count + 1
    if not image_path:
        return None, None, "**请先上传一张坝面图像。**", None, None, count, _stats_html(count, [], SERVER_ONLINE)
    try:
        img = Image.open(image_path)
    except Exception as e:
        return None, None, f"**图像读取失败:** {e}", None, None, count, _stats_html(count, [], SERVER_ONLINE)
    try:
        detections = detect(image_path)
    except Exception as e:
        annotated = draw_boxes(img, [])
        return annotated, None, f"**检测服务调用失败:** {e}", [], None, count, _stats_html(count, [], SERVER_ONLINE)
    if detections is None:
        detections = []
    annotated = draw_boxes(img, detections)
    heatmap = None
    if get_heatmap is not None:
        try:
            heatmap = get_heatmap(image_path)
        except Exception:
            heatmap = None
    try:
        report = generate_report(detections)
    except Exception as e:
        report = f"**报告生成失败:** {e}"
    report_file = _save_report(report, "dam_report.md")
    return annotated, heatmap, report, detections, report_file, count, _stats_html(count, detections, SERVER_ONLINE)


def run_video_inspection(video_path, count):
    """视频巡检:抽帧检测 → 标注视频 + 汇总报告"""
    count = count + 1
    if not video_path:
        return None, "**请先上传一段巡检视频。**", None, None, count, _stats_html(count, [], SERVER_ONLINE)
    try:
        video_out_path, samples = process_video(video_path, detect)
    except Exception as e:
        return None, f"**视频处理失败:** {e}", None, None, count, _stats_html(count, [], SERVER_ONLINE)
    if not samples:
        return None, "**未从视频中抽到有效帧。**", None, None, count, _stats_html(count, [], SERVER_ONLINE)
    try:
        report = generate_video_report(samples)
    except Exception as e:
        report = f"**报告生成失败:** {e}"
    report_file = _save_report(report, "dam_video_report.md")
    return video_out_path, report, report_file, None, count, _stats_html(count, [], SERVER_ONLINE)


def chat_respond(question, history, detections):
    """基于最近一次图片巡检结果问答;服务不可用时降级为规则回答"""
    history = history or []
    if not question or not str(question).strip():
        return history, ""
    if detections is None:
        reply = "请先完成一次图片巡检,再向我提问。"
    else:
        dets = detections[:QA_MAX_DETS]
        try:
            reply = answer_question(question, dets, history)
        except Exception:
            try:
                reply = answer_fallback(question, dets)
            except Exception:
                reply = "问答服务暂时不可用,请稍后重试。"
    return history + [(question, reply)], ""


_check_server()

with gr.Blocks(css=CUSTOM_CSS, title="大坝缺陷巡检智能体") as demo:
    gr.HTML(HERO_HTML)

    detections_state = gr.State(None)
    inspect_count_state = gr.State(0)
    stats_html = gr.HTML(_stats_html(0, [], SERVER_ONLINE))

    with gr.Tab("图片巡检"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["card"]):
                input_img = gr.Image(type="filepath", label="上传坝面图像", elem_id="upload_img")
                inspect_btn = gr.Button("开始巡检", variant="primary")
                gr.Markdown("支持 jpg / png,单张检测约 2~5 秒,支持中文文件名。", elem_classes=["hint"])
            with gr.Column(scale=1, elem_classes=["card"]):
                output_img = gr.Image(type="pil", label="检测结果(红框标注)", elem_id="output_img")
            with gr.Column(scale=1, elem_classes=["card"]):
                heatmap_img = gr.Image(type="pil", label="缺陷密度热力图", elem_id="heatmap_img")
        with gr.Column(elem_classes=["card"]):
            report_md = gr.Markdown("上传图像并点击「开始巡检」后,巡检报告将显示在这里。")
            report_file = gr.File(label="下载巡检报告", interactive=False, elem_id="report_file")

    with gr.Tab("视频巡检"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["card"]):
                video_in = gr.Video(label="上传巡检视频")
                video_btn = gr.Button("开始视频巡检", variant="primary")
                gr.Markdown("视频按秒抽帧检测,处理需要一些时间,请耐心等待。", elem_classes=["hint"])
            with gr.Column(scale=2, elem_classes=["card"]):
                video_out = gr.Video(label="标注巡检视频(红框)")
        with gr.Column(elem_classes=["card"]):
            video_report_md = gr.Markdown("上传视频并点击「开始视频巡检」后,视频巡检报告将显示在这里。")
            video_report_file = gr.File(label="下载视频巡检报告", interactive=False, elem_id="video_report_file")

    with gr.Tab("检测问答"):
        with gr.Column(elem_classes=["card"]):
            gr.Markdown("### 针对最近一次巡检结果问答")
            chatbot = gr.Chatbot(label="巡检问答", height=400, render_markdown=False, bubble_full_width=True, elem_id="chatbot")
            question_box = gr.Textbox(label="输入问题", placeholder="例如:裂缝大概多长?最严重的缺陷在哪?整改建议?")
            ask_btn = gr.Button("提问", variant="primary")

    inspect_btn.click(run_inspection, inputs=[input_img, inspect_count_state],
                      outputs=[output_img, heatmap_img, report_md, detections_state, report_file, inspect_count_state, stats_html])
    video_btn.click(run_video_inspection, inputs=[video_in, inspect_count_state],
                    outputs=[video_out, video_report_md, video_report_file, detections_state, inspect_count_state, stats_html])
    ask_btn.click(chat_respond, inputs=[question_box, chatbot, detections_state], outputs=[chatbot, question_box])
    question_box.submit(chat_respond, inputs=[question_box, chatbot, detections_state], outputs=[chatbot, question_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

