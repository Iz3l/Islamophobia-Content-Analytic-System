@echo off
title Islamophobia Detection AI Server
echo Starting Islamophobia Detection AI Server...
echo.

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

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo 1. Install Python from https://www.python.org/
    echo 2. Check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check and install packages
echo Checking required packages...
echo.

pip list | findstr flask >nul
if errorlevel 1 (
    echo Installing Flask and Flask-CORS...
    pip install flask flask-cors
) else (
    echo Flask already installed
)

pip list | findstr torch >nul
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch
) else (
    echo PyTorch already installed
)

pip list | findstr pandas >nul
if errorlevel 1 (
    echo Installing pandas and numpy...
    pip install pandas numpy
) else (
    echo Pandas already installed
)

echo.
echo ========================================
echo Checking port availability...
echo.

REM Check if port 5001 is in use
set PORT=5001
netstat -ano | findstr :5001 >nul
if errorlevel 1 (
    echo Port 5001 is available
) else (
    echo WARNING: Port 5001 is currently in use!
    echo.
    echo Processes using port 5001:
    netstat -ano | findstr :5001
    echo.
    echo Options:
    echo   1. Stop Windows Media Player Network Sharing Service (recommended)
    echo   2. Use port 5002 instead
    echo.
    
    REM Ask user what to do
    set /p CHOICE="Choose (1) to stop service, (2) to use port 5002, or (3) to continue anyway: "
    
    if "%CHOICE%"=="1" (
        echo.
        echo Attempting to stop Windows Media Player Network Sharing Service...
        REM Check if running as administrator
        net session >nul 2>&1
        if errorlevel 1 (
            echo ERROR: Please run this script as Administrator to stop services!
            echo Right-click the script and select "Run as administrator"
            echo.
            echo Continuing with port 5001 anyway...
        ) else (
            net stop WMPNetworkSvc >nul 2>&1
            if errorlevel 1 (
                echo Failed to stop service. Continuing anyway...
            ) else (
                echo Service stopped successfully!
            )
        )
    )
    
    if "%CHOICE%"=="2" (
        set PORT=5002
        echo Will use port 5002 instead
    )
    
    if "%CHOICE%"=="3" (
        echo Continuing with port 5001...
    )
)

echo.
echo ========================================
echo Starting AI Model Server on port %PORT%
echo Server URL: http://localhost:%PORT%
echo Press Ctrl+C to stop
echo ========================================
echo.

REM Create temporary Python script with port configuration
echo Creating server configuration...
(
echo from flask import Flask, request, jsonify
echo from flask_cors import CORS
echo import torch
echo import pandas as pd
echo import numpy as np
echo import os
echo import sys
echo.
echo app = Flask(__name__)
echo CORS(app)
echo.
echo # Load your model here
echo # model = torch.load('your_model.pth')
echo # model.eval()
echo.
echo @app.route('/')
echo def home():
echo     return jsonify({"message": "Islamophobia Detection AI Server is running", "status": "active"})
echo.
echo @app.route('/health', methods=['GET'])
echo def health():
echo     return jsonify({"status": "healthy", "port": %PORT%})
echo.
echo @app.route('/predict', methods=['POST'])
echo def predict():
echo     try:
echo         data = request.get_json()
echo         text = data.get('text', '')
echo         # Add your prediction logic here
echo         # result = model.predict(text)
echo         result = {"text": text, "prediction": "sample", "confidence": 0.95}
echo         return jsonify(result)
echo     except Exception as e:
echo         return jsonify({"error": str(e)}), 500
echo.
echo if __name__ == '__main__':
echo     port = %PORT%
echo     if len(sys.argv) > 2 and sys.argv[1] == '--port':
echo         port = int(sys.argv[2])
echo     print(f"Starting server on http://localhost:{port}")
echo     app.run(host='0.0.0.0', port=port, debug=True)
) > temp_server.py

REM Start the server
python temp_server.py

REM Clean up temp file
if exist temp_server.py del temp_server.py

echo.
echo Server stopped.
pause