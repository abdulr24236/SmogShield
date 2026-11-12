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

echo Generating model artifacts (one-time)...
echo.
"%PYTHON%" -m src.train_pipeline
echo.
echo Done. Now double-click run_app.bat to start the app.
pause
