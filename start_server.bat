@echo off
title DamCrack 检测服务
cd /d "D:\.Download\crack-detection"
echo ============================================
echo   DamCrack 检测服务正在启动...
echo   成员 B 请连接: http://10.22.28.46:8000
echo   关闭本窗口 = 停止服务
echo ============================================
temp_env\Scripts\python.exe -m uvicorn detect_server:app --host 0.0.0.0 --port 8000
pause
