@echo off
REM Setup script for TD3 attempts (Windows) to create GPU-enabled conda env

setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "ENV_NAME=inverted-pendulum-gpu"

if /i "%1" NEQ "create" (
    echo Usage: setup.bat create
    goto :eof
)

where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda not found. Please install Miniconda/Anaconda and add it to PATH.
    goto :eof
)

call conda --version || goto :error

echo Removing any existing environment "%ENV_NAME%" (safe to ignore errors)...
call conda env remove -n "%ENV_NAME%" -y >nul 2>&1

echo Creating environment "%ENV_NAME%" from environment.yml ...
call conda env create -n "%ENV_NAME%" -f environment.yml || goto :error

echo.
echo [OK] Environment created successfully!
echo Activate with:
echo   conda activate %ENV_NAME%
echo Verify CUDA with:
echo   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
goto :end

:error
echo [ERROR] Failed to create environment. See conda output above.

:end
popd >nul
endlocal

