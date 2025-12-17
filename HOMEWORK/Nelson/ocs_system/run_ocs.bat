@echo off
REM OCS System - Quick Test Script
REM 測試硬幣辨識系統

echo ========================================
echo OCS 硬幣辨識系統 - 快速測試
echo ========================================
echo.

REM 檢查 Python 環境
echo [1] 檢查 Python 環境...
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\vision\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" --version
    echo.
    echo [2] 安裝依賴套件...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo.
    echo [3] 執行硬幣辨識程式...
    "%PYTHON_EXE%" main.py
    goto :end
)

echo Warning: Specific Python path not found. Falling back to system python.
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version
    echo.
    echo [2] 安裝依賴套件...
    python -m pip install -r requirements.txt
    echo.
    echo [3] 執行硬幣辨識程式...
    python main.py
    goto :end
)

echo.
echo 錯誤: 找不到 Python
echo 請確認已安裝 Python 3.8+ 或檢查 conda 環境路徑
echo.

:end

pause
