#!/bin/bash
# ============================================================
# CNU Mapper · conda environment setup (one-time)
# CNU Mapper · conda 环境一次性配置
# ============================================================
# Usage / 用法: bash setup.sh
# ============================================================

set -e
cd "$(dirname "$0")"

ENV_NAME="cnu_mapper"

echo "=================================================="
echo "  CNU Mapper - conda environment setup"
echo "=================================================="

# 1. Check conda / 检查 conda
if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "conda available: $(conda --version)"

# 2. Create env if missing / 若环境不存在则创建
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists; skipping creation."
else
    echo "Creating conda env '$ENV_NAME' (Python 3.11)..."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

# 3. Install deps in the env (no need to activate) / 在新环境里装依赖
echo "Installing dependencies..."
conda run -n "$ENV_NAME" --no-capture-output pip install --quiet -r requirements.txt

echo ""
echo "=================================================="
echo "Environment ready."
echo "=================================================="

if [ ! -f ".env" ]; then
    echo ""
    echo "No .env file found. Create one with your API key:"
    echo ""
    echo "   echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' > .env"
    echo ""
fi

echo "To run:"
echo ""
echo "   conda activate $ENV_NAME"
echo "   python llm_classifier.py"
echo ""
echo "To exit the env:"
echo ""
echo "   conda deactivate"
echo ""
