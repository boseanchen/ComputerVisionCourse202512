@echo off
REM 診斷腳本 - 找出最佳檢測參數

echo ========================================
echo  OCS 參數診斷工具
echo ========================================
echo.

REM 檢查 Python
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\vision\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" diagnose_params.py
    goto :end
)

echo Warning: Specific Python path not found. Falling back to system python.
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python diagnose_params.py
    goto :end
)

echo 錯誤: 找不到 Python
echo.

:end
pause
