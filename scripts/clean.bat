@echo off
REM Clean build artifacts

echo Cleaning build artifacts...

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
if exist __pycache__ rmdir /s /q __pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .coverage del /q .coverage

echo Clean complete!
pause
