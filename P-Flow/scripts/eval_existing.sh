#!/bin/bash
# Quick evaluation of existing experiment results
# Run on AutoDL: bash scripts/eval_existing.sh

cd /root/autodl-tmp/P-Flow

# Evaluate test_022 (3 iterations, alpha=0.001, 1.3B model)
echo "=========================================="
echo "Evaluating test_022 results..."
echo "=========================================="
python evaluation/eval_reproduction.py \
    --experiment_dir /root/autodl-tmp/outputs/test_022 \
    --device cuda \
    --num_frames 16

echo ""
echo "=========================================="
echo "Done! Check /root/autodl-tmp/outputs/test_022/reproduction_eval.json"
echo "=========================================="
