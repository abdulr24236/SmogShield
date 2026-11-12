@echo off
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo Python was not found.
  echo Install Python 3.11 or newer, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating project-local virtual environment .venv ...
  "%PY%" -m venv .venv
  if errorlevel 1 (
    echo Could not create .venv.
    pause
    exit /b 1
  )
)

echo Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo.
echo Done. Next: double-click setup_models.bat, then run_app.bat.
pause
