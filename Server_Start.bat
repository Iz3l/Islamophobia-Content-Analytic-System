@echo off
title Islamophobia Detection AI Server
echo Starting Islamophobia Detection AI Server...
echo.

REM Set environment variables
set KERAS_BACKEND=torch
set PYTHONIOENCODING=utf-8

REM First navigate to XAMPP htdocs
cd /d C:\Xampp\htdocs

REM Check if folder exists
if not exist "IslamophobiaDetectionV3" (
    echo ERROR: Folder 'IslamophobiaDetectionV3' not found!
    echo Available folders in htdocs:
    dir /B
    pause
    exit /b 1
)

REM Navigate to your project
cd IslamophobiaDetectionV3
echo Current directory: %CD%
echo.

REM Detect Python
set PYTHON_BIN=python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python command not in system PATH. Searching common directories...
    if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe" (
        set PYTHON_BIN="%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    ) else if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
        set PYTHON_BIN="%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
    ) else (
        for /d %%d in ("%USERPROFILE%\AppData\Local\Programs\Python\Python*") do (
            if exist "%%d\python.exe" (
                set PYTHON_BIN="%%d\python.exe"
            )
        )
    )
)

REM Verify Python executable works
%PYTHON_BIN% --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo 1. Install Python from https://www.python.org/
    echo 2. Check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check packages
echo Checking required packages...
%PYTHON_BIN% -m pip list | findstr flask >nul || echo Installing Flask... && %PYTHON_BIN% -m pip install flask flask-cors
%PYTHON_BIN% -m pip list | findstr torch >nul || echo Installing PyTorch... && %PYTHON_BIN% -m pip install torch
%PYTHON_BIN% -m pip list | findstr pandas >nul || echo Installing pandas... && %PYTHON_BIN% -m pip install pandas numpy
%PYTHON_BIN% -m pip list | findstr keras >nul || echo Installing Keras... && %PYTHON_BIN% -m pip install keras

echo.
echo ========================================
echo Starting AI Model Server...
echo Server URL: http://localhost:5001
echo Press Ctrl+C to stop
echo ========================================
echo.

REM Start the server
%PYTHON_BIN% TextApp.py

echo.
echo Server stopped.
pause