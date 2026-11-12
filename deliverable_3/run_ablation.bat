@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON=py"
)
if not defined PYTHON (
  where python >nul 2>nul
  if not errorlevel 1 set "PYTHON=python"
)
if not defined PYTHON (
  echo Python was not found.
  echo Run setup_env.bat once, or install Python 3.11+ and add it to PATH.
  pause
  exit /b 1
)

echo Running weather/calendar ablation (this takes a few minutes)...
echo.
"%PYTHON%" -m src.ablation
echo.
echo Results written to models\ablation.json
pause
