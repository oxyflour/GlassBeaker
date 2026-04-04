@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"

where python >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=python"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
  ) else (
    echo Could not find python.exe or py.exe.
    exit /b 9009
  )
)

pushd "%SCRIPT_DIR%"
call %PYTHON_CMD% "%SCRIPT_DIR%run_render.py" ^
  --engine-root "C:\Program Files\Epic Games\UE_5.7" ^
  --project-root "%PROJECT_ROOT%" ^
  --project-name "HeadlessObjRender" ^
  --obj "C:\Users\oxyfl\Downloads\basic-phone.obj" ^
  --output-dir "%SCRIPT_DIR%Shots" ^
  --width 1920 ^
  --height 1080
set "RC=%ERRORLEVEL%"
popd

exit /b %RC%
