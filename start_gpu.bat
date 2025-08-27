@echo off
REM GPU optimized startup script for AI Hand-Drawn Map Generator

set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%.venv

echo ===============================================
echo  AI TourMap Generator - GPU Accelerated
echo ===============================================
echo.

REM Clear proxy settings and setup offline mode
echo [INFO] Configuring network settings for optimal performance...
set http_proxy=
set https_proxy=
set all_proxy=
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

REM Set offline mode only if explicitly needed
if exist "%SCRIPT_DIR%model_cache" (
    echo [INFO] Model cache detected, checking network...
    ping -n 1 huggingface.co >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [INFO] Network available, using online mode for latest models...
        set HF_HUB_OFFLINE=0
        set TRANSFORMERS_OFFLINE=0
    ) else (
        echo [INFO] Network unavailable, enabling offline mode...
        set HF_HUB_OFFLINE=1
        set TRANSFORMERS_OFFLINE=1
    )
    set HF_ENDPOINT=https://huggingface.co
) else (
    echo [INFO] No model cache found, will download from official source...
    set HF_HUB_OFFLINE=0
    set TRANSFORMERS_OFFLINE=0
    set HF_ENDPOINT=https://huggingface.co
)

REM Check if virtual environment exists and activate it
if exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment from: %VENV_PATH%
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment not found at %VENV_PATH%.
    echo [WARNING] Attempting to run with current environment...
)

echo.
echo [INFO] Checking GPU availability...
python -c "import torch; print('[INFO] Checking GPU environment...')" 2>nul
python -c "import torch; print('[OK] GPU Available:', torch.cuda.is_available())" 2>nul
python -c "import torch; print('[INFO] GPU Count:', torch.cuda.device_count())" 2>nul
python -c "import torch; print('[INFO] GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" 2>nul
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] GPU detected and ready for acceleration!
    echo [INFO] Proceeding with GPU mode...
    set USE_GPU_MODE=true
    goto :start_server
) else (
    echo [WARNING] No GPU detected! Available options:
    echo [WARNING] 1. Install NVIDIA drivers and CUDA toolkit
    echo [WARNING] 2. Install GPU version of PyTorch
    echo [WARNING] 3. Use CPU version instead: start.bat
    echo.
    echo [CHOICE] Continue with CPU fallback mode? (Y/N)
    choice /c YN /n /m "Press Y for CPU mode, N to exit: "
    if errorlevel 2 (
        echo [INFO] Exiting... Please use start.bat for CPU mode.
        pause
        exit /b 1
    )
    echo [INFO] Continuing in CPU fallback mode...
    set USE_GPU_MODE=false
)

:start_server

echo [INFO] Checking network connectivity...
ping -n 1 8.8.8.8 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Network connectivity is available
) else (
    echo [WARNING] Network connectivity issues detected
)

echo.
echo ===============================================
echo  Starting AI Map Generator...
echo ===============================================
echo.

REM 根据GPU检测结果选择启动方式
if "%USE_GPU_MODE%"=="true" (
    echo [INFO] Starting with GPU acceleration...
    uvicorn main_gpu:app --host 127.0.0.1 --port 8003
) else (
    echo [INFO] Starting with CPU fallback mode...
    uvicorn main:app --host 127.0.0.1 --port 8002
)

REM Check if the server started successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo ^|                                              ^|
    echo ^|    *** GPU SERVER STARTED SUCCESSFULLY! *** ^|
    echo ^|                                              ^|
    echo ^|    URL: http://127.0.0.1:8003                ^|
    echo ^|    GPU-Accelerated Map Generator Ready!      ^|
    echo ^|    Press Ctrl+C to stop the server          ^|
    echo ^|                                              ^|
    echo ================================================
    echo.
    echo +----------------------------------------------+
    echo ^|       GPU-Powered AI TourMap Generator       ^|
    echo ^|     Transform maps with lightning speed!    ^|
    echo ^|         Check /api/gpu_status for info      ^|
    echo +----------------------------------------------+
    echo.
) else (
    echo.
    echo ************************************************
    echo *                                              *
    echo *        GPU SERVER FAILED TO START!         *
    echo *                                              *
    echo *  GPU Troubleshooting:                        *
    echo *  1. Install CUDA-compatible PyTorch          *
    echo *  2. Check NVIDIA drivers                      *
    echo *  3. Verify GPU memory availability            *
    echo *  4. Run: pip install torch torchvision       *
    echo *     torchaudio --index-url                    *
    echo *     https://download.pytorch.org/whl/cu118    *
    echo *                                              *
    echo ************************************************
    echo.
)

pause