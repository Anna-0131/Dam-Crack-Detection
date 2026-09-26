"""DamCrack 检测 HTTP 服务（FastAPI）。

启动（在项目根目录、temp_env 环境下）：
    temp_env\\Scripts\\python.exe -m uvicorn detect_server:app --host 0.0.0.0 --port 8000

成员 B 访问：
    POST http://<你的局域网IP>:8000/detect   （multipart 上传图片文件）
    GET  http://<你的局域网IP>:8000/health    （连通性检查）

返回 JSON：
    {
        "detections": [{"class", "class_id", "confidence", "bbox_xyxy"}, ...],
        "summary": {"crack": n, "spalling": m},
        "report": "文本摘要",
        "annotated_image_base64": "<JPEG base64>"
    }
"""

import base64
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from crack_detector import _get_model, detect

app = FastAPI(title='DamCrack 检测服务', version='1.0')

# import 时（即启动时）就加载模型，避免第一个请求慢
_get_model()
print('检测服务就绪，模型已加载。')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/detect')
async def detect_image(file: UploadFile = File(...), conf: float = 0.5):
    """接收一张图片，返回检测结果（结构化 + 标注图 base64 + 文本摘要）。"""
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='无法解码图片，请上传 jpg/png/bmp 等格式')

    try:
        result = detect(img, conf=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'检测失败: {e}')

    # 标注图是 RGB numpy，cv2.imencode 需要 BGR，转一下再压成 JPEG base64
    annotated_bgr = cv2.cvtColor(result['annotated_image'], cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_b64 = base64.b64encode(buf).decode('ascii') if ok else ''

    # 热力图同样转 base64
    heatmap_bgr = cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR)
    ok2, buf2 = cv2.imencode('.jpg', heatmap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    heatmap_b64 = base64.b64encode(buf2).decode('ascii') if ok2 else ''

    return {
        'detections': result['detections'],
        'summary': result['summary'],
        'quantification': result['quantification'],
        'report': result['report'],
        'annotated_image_base64': annotated_b64,
        'heatmap_base64': heatmap_b64,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
