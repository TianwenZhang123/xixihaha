#!/bin/bash
# ==============================================================================
# P-Flow AutoDL Setup Script
# One-click dependency installation for AutoDL 4090 environment.
#
# Prerequisites:
#   - AutoDL instance with CUDA (RTX 4090 recommended)
#   - Model already downloaded at /root/autodl-tmp/models/Wan2.1-T2V-1.3B
#   - Dataset at /root/autodl-tmp/data/moviegen_bench/
#
# Usage:
#   cd /root/autodl-tmp/videofake
#   chmod +x setup_autodl.sh
#   ./setup_autodl.sh
#
# After setup, set your DashScope API key:
#   export DASHSCOPE_API_KEY="your-key-here"
# ==============================================================================

set -e

echo "=============================================="
echo " P-Flow AutoDL Environment Setup"
echo "=============================================="

# Check Python version
echo "[1/5] Checking Python version..."
python3 --version

# Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "[3/5] Installing core dependencies..."
pip install -r requirements.txt

# Verify critical packages
echo "[4/5] Verifying installation..."
python3 -c "
import torch
import numpy as np
import yaml
from openai import OpenAI
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  numpy: {np.__version__}')
print(f'  PyYAML: {yaml.__version__}')
print('  openai: OK')
"

# Check model and data paths
echo "[5/5] Checking paths..."
MODEL_PATH="/root/autodl-tmp/models/Wan2.1-T2V-1.3B"
DATA_PATH="/root/autodl-tmp/data/moviegen_bench"

if [ -d "$MODEL_PATH" ]; then
    echo "  Model path OK: $MODEL_PATH"
else
    echo "  WARNING: Model not found at $MODEL_PATH"
fi

if [ -d "$DATA_PATH" ]; then
    echo "  Dataset path OK: $DATA_PATH"
    echo "  Videos found: $(ls $DATA_PATH/water_mark_out/*.mp4 2>/dev/null | wc -l)"
else
    echo "  WARNING: Dataset not found at $DATA_PATH"
fi

# Check API key
echo ""
echo "=============================================="
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "WARNING: DASHSCOPE_API_KEY not set!"
    echo ""
    echo "Before running P-Flow, set your API key:"
    echo "  export DASHSCOPE_API_KEY=\"your-key-here\""
else
    echo "DASHSCOPE_API_KEY is set. Ready to go!"
fi
echo "=============================================="

echo ""
echo "Setup complete! Run P-Flow with:"
echo ""
echo "  export DASHSCOPE_API_KEY=\"your-key-here\""
echo "  python scripts/run_pflow_paper.py \\"
echo "    --reference_video /root/autodl-tmp/data/moviegen_bench/water_mark_out/024.mp4 \\"
echo "    --prompt \"A cat wakes up its owner\" \\"
echo "    --output_dir /root/autodl-tmp/outputs/test_024 \\"
echo "    --seed 42"
echo ""
