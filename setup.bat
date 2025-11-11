@echo off
REM Setup script for Inverted Pendulum Competition (Windows)
REM This script creates a conda environment and installs all dependencies

setlocal enabledelayedexpansion

set ENV_NAME=inverted-pendulum-env

echo =========================================
echo Inverted Pendulum Competition Setup
echo =========================================
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first:
    echo   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo   - Anaconda: https://www.anaconda.com/products/distribution
    exit /b 1
)

echo [OK] Conda found
conda --version
echo.

REM Check if environment already exists
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Environment '%ENV_NAME%' already exists.
    set /p REPLY="Do you want to remove it and create a fresh one? (y/N): "
    if /i "!REPLY!"=="y" (
        echo Removing existing environment...
        conda env remove -n "%ENV_NAME%" -y
    ) else (
        echo Updating existing environment...
        conda env update -n "%ENV_NAME%" -f environment.yml --prune
        echo.
        echo [OK] Environment updated successfully!
        echo.
        echo To activate the environment, run:
        echo   conda activate %ENV_NAME%
        exit /b 0
    )
)

REM Create the environment
echo Creating conda environment '%ENV_NAME%' from environment.yml...
conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create environment
    exit /b 1
)

echo.
echo [OK] Environment created successfully!
echo.
echo =========================================
echo Next Steps:
echo =========================================
echo.
echo 1. Activate the environment:
echo    conda activate %ENV_NAME%
echo.
echo 2. Verify installation:
echo    python -c "import mujoco; import numpy; import scipy; import glfw; print('All packages imported successfully!')"
echo.
echo 3. Run the simulation:
echo    python Run_PendulumEnv.py
echo.
echo To deactivate the environment later:
echo    conda deactivate
echo.

endlocal

