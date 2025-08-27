@echo off
REM 模型检查和预下载脚本

echo ===============================================
echo  AI TourMap Generator - Model Check & Download
echo ===============================================
echo.

set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%.venv
set CACHE_DIR=%SCRIPT_DIR%model_cache

REM 激活虚拟环境
if exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [INFO] 激活虚拟环境...
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [ERROR] 虚拟环境不存在，请先运行 install_gpu.bat
    pause
    exit /b 1
)

echo [INFO] 检查模型缓存状态...
if exist "%CACHE_DIR%" (
    echo [OK] 模型缓存目录存在: %CACHE_DIR%
    
    REM 检查关键模型文件
    set MODEL_COMPLETE=true
    
    if not exist "%CACHE_DIR%\models--lllyasviel--sd-controlnet-canny" (
        echo [MISSING] ControlNet Canny 模型未找到
        set MODEL_COMPLETE=false
    ) else (
        echo [OK] ControlNet Canny 模型已缓存
    )
    
    if not exist "%CACHE_DIR%\models--runwayml--stable-diffusion-v1-5" (
        echo [MISSING] Stable Diffusion v1.5 模型未找到
        set MODEL_COMPLETE=false
    ) else (
        echo [OK] Stable Diffusion v1.5 模型已缓存
    )
    
    if not exist "%CACHE_DIR%\models--h94--IP-Adapter" (
        echo [MISSING] IP-Adapter 模型未找到
        set MODEL_COMPLETE=false
    ) else (
        echo [OK] IP-Adapter 模型已缓存
    )
    
    if "%MODEL_COMPLETE%"=="true" (
        echo [SUCCESS] 所有模型都已缓存，可以离线运行！
        echo [INFO] 使用 start_gpu.bat 或 start_offline.bat 启动服务
        pause
        exit /b 0
    )
) else (
    echo [INFO] 模型缓存目录不存在，需要首次下载
    set MODEL_COMPLETE=false
)

if "%MODEL_COMPLETE%"=="false" (
    echo.
    echo [WARNING] 检测到模型文件不完整或缺失
    echo [INFO] 需要下载AI模型才能正常运行
    echo.
    echo [CHOICE] 现在下载模型吗？这可能需要5-15分钟
    choice /c YN /n /m "Press Y to download, N to exit: "
    
    if errorlevel 2 (
        echo [INFO] 取消下载。请手动运行 download_models.bat
        pause
        exit /b 1
    )
    
    echo.
    echo [INFO] 开始下载模型...
    
    REM 清除代理设置，使用官方源
    set http_proxy=
    set https_proxy=
    set all_proxy=
    set HF_ENDPOINT=https://huggingface.co
    
    echo [INFO] 使用官方源下载: %HF_ENDPOINT%
    
    REM 运行下载脚本
    python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
exec(open('download_models.py').read())
    " 2>&1
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo [SUCCESS] 模型下载完成！
        echo [INFO] 现在可以使用 start_gpu.bat 启动服务
    ) else (
        echo.
        echo [ERROR] 模型下载失败
        echo [INFO] 请检查网络连接或手动运行 download_models.bat
        echo [INFO] 也可以尝试使用镜像源下载
    )
)

pause