@echo off
REM Run test suite

echo Running DriveOrganizerPro test suite...
echo.

pytest tests/ -v --cov=src/drive_organizer_pro --cov-report=term-missing

pause
