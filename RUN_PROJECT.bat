@echo off
REM ======================================================================
REM   IU Learning Classroom - Quick Start
REM   Easy setup for running on any PC
REM ======================================================================

REM ========== CONFIGURATION ==========
REM Change these settings if needed:
set HOST=0.0.0.0
set PORT=8000
set AUTO_OPEN_BROWSER=yes
REM ===================================

echo.
echo ========================================
echo   IU Learning Classroom
echo   Voice-Powered Education Platform
echo ========================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"
echo [*] Working directory: %CD%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [*] Python found: 
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    echo [*] Virtual environment found - starting server...
    echo.
    goto :START_SERVER
)

REM If venv doesn't exist, create it
echo [WARNING] Virtual environment not found!
echo [*] Creating virtual environment...
echo.
python -m venv venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [*] Installing dependencies...
echo This may take a few minutes...
echo.
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\pip.exe install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Setup complete!
echo.

:START_SERVER

REM Kill any existing Python processes on the port
echo [*] Checking for existing processes on port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo [*] Stopping process %%a...
    taskkill /F /PID %%a >nul 2>&1
)
echo.

REM Display startup info
echo ========================================
echo [*] Starting server...
echo [*] Host: %HOST%
echo [*] Port: %PORT%
echo.
if "%HOST%"=="0.0.0.0" (
    echo [*] Access URLs:
    echo     Local:    http://127.0.0.1:%PORT%
    echo     Network:  http://YOUR_IP_ADDRESS:%PORT%
    echo.
    echo [TIP] To access from other devices on your network,
    echo       find your IP address with: ipconfig
    echo       Look for "IPv4 Address" under your network adapter
) else (
    echo [*] Server URL: http://%HOST%:%PORT%
)
echo.
echo ========================================
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

REM Auto-open browser
if /i "%AUTO_OPEN_BROWSER%"=="yes" (
    timeout /t 3 /nobreak >nul
    start http://127.0.0.1:%PORT%
)

REM Start the server
venv\Scripts\python.exe -m uvicorn app.main:app --host %HOST% --port %PORT% --reload

REM If server exits, pause so user can see error messages
echo.
echo [*] Server stopped
pause
