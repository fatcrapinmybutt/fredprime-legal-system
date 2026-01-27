@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHON=python
"%PYTHON%" "%SCRIPT_DIR%OUT\code\mcl_plane_suite.py" --in "%SCRIPT_DIR%IN" --out "%SCRIPT_DIR%OUT" --mode converge --adversarial true --web false
endlocal
