"""DamCrack 检测服务客户端示例（给成员 B 用）。

指定服务地址有三种方式，优先级从高到低：
1. 调用时传参：detect('图.jpg', server_url='http://...:8000')
2. 环境变量：set DETECT_SERVER_URL=http://...:8000   (PowerShell)
3. 默认值（文件里 SERVER_URL 常量，已填好）

用法：
    python detect_client.py 图片.jpg                 # 用默认地址
    python detect_client.py 图片.jpg http://IP:8000  # 临时指定地址
"""

import base64
import io
import os

import requests

# 默认服务地址（可用环境变量 DETECT_SERVER_URL 覆盖）
SERVER_URL = os.environ.get('DETECT_SERVER_URL', 'http://10.22.28.46:8000')


def detect(image_path: str, conf: float = 0.25, server_url: str = None) -> dict:
    """上传一张图片到检测服务，返回 {detections, summary, report, annotated_image_base64}。"""
    url = server_url or SERVER_URL
    with open(image_path, 'rb') as f:
        resp = requests.post(
            f'{url}/detect',
            files={'file': (os.path.basename(image_path), f, 'image/jpeg')},
            params={'conf': conf},
            timeout=60,
        )
    resp.raise_for_status()   # 网络/服务出错会抛异常
    return resp.json()


def annotated_to_image(result: dict):
    """把返回里的 base64 标注图解码成 PIL.Image，方便界面展示。"""
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(result['annotated_image_base64'])))


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('用法: python detect_client.py 图片路径 [服务地址]')
        print(f'当前默认服务地址: {SERVER_URL}')
        sys.exit(1)

    url = sys.argv[2] if len(sys.argv) > 2 else SERVER_URL
    r = detect(sys.argv[1], server_url=url)
    print('report:\n' + r['report'])
    print('detections:', r['detections'])
    print('summary:', r['summary'])

    img = annotated_to_image(r)
    img.save('annotated_result.jpg')
    print('标注图已保存到 annotated_result.jpg')
