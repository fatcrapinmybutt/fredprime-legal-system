@echo off
REM Build standalone executable with PyInstaller

echo ========================================
echo  Building DriveOrganizerPro Executable
echo ========================================
echo.

REM Check for PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Build executable
pyinstaller --onefile --windowed ^
  --name="DriveOrganizerPro-MBP-LLC" ^
  --add-data "src/drive_organizer_pro/config;drive_organizer_pro/config" ^
  --add-data "assets;assets" ^
  --noconsole ^
  src/drive_organizer_pro/gui/main_window.py

echo.
echo ========================================
echo  Build Complete!
echo ========================================
echo.
echo Executable location: dist\DriveOrganizerPro-MBP-LLC.exe
echo.
pause
