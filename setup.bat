@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -m pip install -r requirements.txt
) else (
    python -m pip install -r requirements.txt
)

if errorlevel 1 (
    echo Dependency installation failed. Check the error above.
    pause
    exit /b 1
)

echo Dependencies installed successfully.
pause
