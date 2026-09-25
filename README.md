# 大坝缺陷巡检智能体

上传巡检照片/视频 → YOLO 缺陷检测 → LLM 生成巡检报告 → 基于检测结果问答。

## 运行

1. 复制 .env.example 为 .env,填入 DEEPSEEK_API_KEY
2. pip install -r requirements.txt
3. python web_app.py,浏览器打开 http://127.0.0.1:7860

## 文件分工

- 成员A:detector.py、report_chain.py
- 成员B:web_app.py、qa_chain.py、utils.py、video_inspect.py
- 共享:llm.py、INTERFACE.md

## 仓库其他内容

- 组长维护:检测服务(detect_server.py / detect_client.py)、数据集工具(build_damcrack_dataset.py 等)
