@echo off
REM OCS System - Test Recognition Script
REM 測試硬幣辨識準確度

echo ========================================
echo  OCS 硬幣辨識測試
echo ========================================
echo.

REM 檢查 Python 環境
echo [1] 檢查 Python 環境...
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\vision\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" --version
    echo.
    echo [2] 執行測試...
    "%PYTHON_EXE%" test_recognition.py
    goto :end
)

echo Warning: Specific Python path not found. Falling back to system python.
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version
    echo.
    echo [2] 執行測試...
    python test_recognition.py
    goto :end
)

echo.
echo ========================================
echo 錯誤: 找不到 Python
echo ========================================
echo.
echo 請確認已安裝 Python 3.8 或更高版本
echo 或檢查 conda 環境路徑: %PYTHON_EXE%
echo.

:end
pause
