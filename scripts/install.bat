@echo off
REM ========================================
REM  MBP LLC DriveOrganizerPro Installer
REM  Maximum Business Performance
REM  Powered by Pork(TM)
REM ========================================

echo.
echo ========================================
echo  MBP LLC DriveOrganizerPro
echo  Level 9999 Edition Installer
echo ========================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or later from python.org
    pause
    exit /b 1
)

echo Installing DriveOrganizerPro...
echo.

REM Upgrade pip
python -m pip install --upgrade pip

REM Install package
python -m pip install -e .

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo You can now run DriveOrganizerPro with:
echo   driveorganizerpro-gui
echo.
pause
