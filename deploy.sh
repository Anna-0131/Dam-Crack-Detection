#!/bin/bash
# DamCrack 检测服务一键部署脚本（上传到服务器后，在服务器上运行一次即可）
set -e
cd "$(dirname "$0")"

echo "=== 1/4 更新包管理器 ==="
sudo apt-get update -y

echo "=== 2/4 安装 Python、pip 和 OpenCV 需要的系统图形库 ==="
sudo apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

echo "=== 3/4 创建虚拟环境并安装依赖（torch CPU 版） ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics fastapi uvicorn python-multipart opencv-python-headless pillow numpy

echo "=== 4/4 启动服务（后台常驻） ==="
export CRACK_MODEL_PATH="$(pwd)/best.pt"
nohup venv/bin/python -m uvicorn detect_server:app --host 0.0.0.0 --port 8000 > damcrack.log 2>&1 &
sleep 5

echo ""
echo "=== 部署完成 ==="
echo "日志文件: $(pwd)/damcrack.log"
echo "本地自检结果:"
curl -s http://127.0.0.1:8000/health || echo "(服务可能还在启动，稍等几秒再看日志)"
echo ""
