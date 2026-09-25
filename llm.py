# -*- coding: utf-8 -*-
"""DeepSeek 封装(共享)。成员A的 report_chain 和成员B的 qa_chain 都调用 chat()。"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取项目根目录的 .env(必须在项目根目录运行)

_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"


def chat(messages, temperature=0.3):
    """messages: [{"role": "system|user|assistant", "content": str}, ...]
    返回回复文本。失败时抛异常,由调用方兜底。"""
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content
