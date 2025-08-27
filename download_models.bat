@echo off
REM AI模型下载脚本 (国内镜像优化版)

set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%.venv

echo ===============================================
echo  AI TourMap Generator - Model Downloader
echo  (国内镜像优化版)
echo ===============================================
echo.

REM 设置国内镜像源
echo [INFO] 配置国内镜像源...
set HF_ENDPOINT=https://hf-mirror.com
echo [INFO] 使用镜像: %HF_ENDPOINT%

REM 检查并激活虚拟环境
if exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [INFO] 激活虚拟环境: %VENV_PATH%
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [WARNING] 虚拟环境未找到，使用系统Python
)

echo.
echo [INFO] 检查网络连接...
ping -n 1 hf-mirror.com >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] 镜像站点可访问
) else (
    echo [WARNING] 镜像站点无法访问，尝试其他方案
    set HF_ENDPOINT=https://huggingface.co
)

echo.
echo [INFO] 检查Python依赖...
python -c "import torch, diffusers, transformers" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 缺少必要依赖，请先运行: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ===============================================
echo  开始下载AI模型 (预计需要5-15分钟)
echo ===============================================
echo.

REM 运行下载脚本
python download_models.py

REM 检查下载结果
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo ^|                                              ^|
    echo ^|    *** 模型下载成功完成! ***                  ^|
    echo ^|                                              ^|
    echo ^|    现在可以使用以下命令启动服务:              ^|
    echo ^|    - start_offline.bat  (离线模式)           ^|
    echo ^|    - start_gpu.bat      (GPU加速)            ^|
    echo ^|    - start.bat          (标准模式)           ^|
    echo ^|                                              ^|
    echo ================================================
    echo.
    echo [INFO] 模型文件已保存到 model_cache 目录
    echo [INFO] 可以在无网络环境下运行应用
) else (
    echo.
    echo ************************************************
    echo *                                              *
    echo *        模型下载失败!                        *
    echo *                                              *
    echo *  故障排除建议:                               *
    echo *  1. 检查网络连接                             *
    echo *  2. 尝试使用代理或VPN                        *
    echo *  3. 手动设置镜像: python setup_mirror.py     *
    echo *  4. 重新运行此脚本 (支持断点续传)             *
    echo *                                              *
    echo ************************************************
    echo.
)

pause