@echo off
setlocal
echo ==============================================
echo Semantic NLP Platform - Startup Script
echo ==============================================

:: 1. Environment Setup
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set USE_UV=1
    echo [INFO] uv found. Fast path activated.
    echo [INFO] Syncing environment...
    if not exist pyproject.toml (
        uv init
    )
    uv add -r requirements.txt
    uv sync
    uv lock
    if %ERRORLEVEL% neq 0 goto :error
) else (
    set USE_UV=0
    echo [INFO] uv not found. Falling back to python venv and pip.
    if not exist .venv (
        echo [INFO] Creating virtual environment...
        python -m venv .venv
        if %ERRORLEVEL% neq 0 goto :error
    )
    call .venv\Scripts\activate.bat
    if %ERRORLEVEL% neq 0 goto :error
    
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 goto :error
)

:: 3. Frontend Build
echo [INFO] Building Frontend UI...
cd frontend
call npm install
if %ERRORLEVEL% neq 0 goto :error_frontend
call npm run build
if %ERRORLEVEL% neq 0 goto :error_frontend
cd ..

:: 4. Launch Application
echo [INFO] Launching Application API...
echo [INFO] The website will open in your browser shortly...

:: Open browser after a short delay in the background
start /b cmd /c "ping 127.0.0.1 -n 5 > nul & start http://127.0.0.1:8000"

:: Start FastAPI app
if %USE_UV% equ 1 (
    uv run python -m src.api.server
    if %ERRORLEVEL% neq 0 goto :error
) else (
    python -m src.api.server
    if %ERRORLEVEL% neq 0 goto :error
)

goto :EOF

:error_frontend
cd ..
:error
echo.
echo ==============================================
echo [ERROR] A critical error occurred during startup.
echo Please review the logs above to see what failed.
echo ==============================================
pause
exit /b 1
