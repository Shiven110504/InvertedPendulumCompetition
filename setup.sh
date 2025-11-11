#!/bin/bash

# Setup script for Inverted Pendulum Competition
# This script creates a conda environment and installs all dependencies

set -e  # Exit on any error

ENV_NAME="inverted-pendulum-env"

echo "========================================="
echo "Inverted Pendulum Competition Setup"
echo "========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "  - Anaconda: https://www.anaconda.com/products/distribution"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Updating existing environment..."
        conda env update -n "${ENV_NAME}" -f environment.yml --prune
        echo ""
        echo "✅ Environment updated successfully!"
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create the environment
echo "Creating conda environment '${ENV_NAME}' from environment.yml..."
conda env create -f environment.yml

echo ""
echo "✅ Environment created successfully!"
echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo ""
echo "1. Activate the environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Verify installation:"
echo "   python -c \"import mujoco; import numpy; import scipy; import glfw; print('✅ All packages imported successfully!')\""
echo ""
echo "3. Run the simulation:"
echo "   python Run_PendulumEnv.py"
echo ""
echo "To deactivate the environment later:"
echo "   conda deactivate"
echo ""

