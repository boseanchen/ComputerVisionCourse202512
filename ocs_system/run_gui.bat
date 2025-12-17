@echo off
REM OCS System - GUI Version Launcher
REM 啟動圖形介面版本

echo ========================================
echo  OCS 硬幣辨識系統 - GUI 版本
echo ========================================
echo.

REM 檢查 Python 環境
echo [1] 檢查 Python 環境...
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\vision\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" --version
    echo.
    echo [2] 檢查依賴套件...
    "%PYTHON_EXE%" -m pip install --quiet customtkinter opencv-python pillow numpy
    if %ERRORLEVEL% NEQ 0 (
        echo 警告: 部分套件安裝失敗，嘗試繼續...
    )
    echo.
    echo [3] 啟動圖形介面...
    echo.
    "%PYTHON_EXE%" main_gui.py
    goto :end_check
)

echo Warning: Specific Python path not found. Falling back to system python.
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 錯誤: 找不到 Python
    echo 請確認已安裝 Python 3.8+ 或檢查 conda 環境路徑
    pause
    exit /b 1
)

python --version
echo.

echo [2] 檢查依賴套件...
python -m pip install --quiet customtkinter opencv-python pillow numpy
if %ERRORLEVEL% NEQ 0 (
    echo 警告: 部分套件安裝失敗，嘗試繼續...
)
echo.

echo [3] 啟動圖形介面...
echo.
python main_gui.py

:end_check

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo 程式執行失敗
    echo ========================================
    pause
)
