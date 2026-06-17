#!/bin/bash
# ================================================================
# Phase 1: α 消融实验 — 5个代表性case × 多组参数
# ================================================================
# Case 选择:
#   32 (animal)       → 当前效果好 (+0.14), 验证降α是否伤好case
#   73 (scene)        → 当前效果差 (-0.10), 验证降α能否修复
#   80 (scene+camera) → 当前效果差 (-0.10), 同上
#  111 (scene+camera) → 场景型但效果好 (+0.28), 例外验证
#   50 (unusual)      → 异常活动型 (-0.06), 不同类型验证
#
# 已有对照 (α=0.004 SVD+FI):
#   32: XC=0.8395 (Δ+0.1394)  |  73: XC=0.6509 (Δ-0.1025)
#   80: XC=0.6622 (Δ-0.1008)  | 111: XC=0.8555 (Δ+0.2834)
#   50: XC=0.7915 (Δ-0.0552)
# ================================================================

set -e

CASE_IDS="32 73 80 111 50"
CAPTION_DIR="/root/xixihaha/test-v200/test-v200/captions"

echo "=========================================="
echo "  Phase 1: α Ablation — 5 representative cases"
echo "  Cases: $CASE_IDS"
echo "=========================================="

# ────────────────────────────────────────────
# 实验1: α=0.001 纯SVD
# ────────────────────────────────────────────
echo ""
echo ">>> [1/6] α=0.001, SVD only"
python run.py \
    --data_dir data/videos \
    --caption_dir $CAPTION_DIR \
    --output_dir outputs/ablation_a001_SVD_5cases \
    --sample_ids $CASE_IDS \
    --inversion --svd --blend --alpha 0.001 \
    --seed 42 --verbose

# ────────────────────────────────────────────
# 实验2: α=0.001 SVD+FI (λ=0.05)
# ────────────────────────────────────────────
echo ""
echo ">>> [2/6] α=0.001, SVD+FI (λ=0.05)"
python run.py \
    --data_dir data/videos \
    --caption_dir $CAPTION_DIR \
    --output_dir outputs/ablation_a001_SVD_FI_5cases \
    --sample_ids $CASE_IDS \
    --inversion --svd --blend --alpha 0.001 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose

# ────────────────────────────────────────────
# 实验3: α=0.002 纯SVD
# ────────────────────────────────────────────
echo ""
echo ">>> [3/6] α=0.002, SVD only"
python run.py \
    --data_dir data/videos \
    --caption_dir $CAPTION_DIR \
    --output_dir outputs/ablation_a002_SVD_5cases \
    --sample_ids $CASE_IDS \
    --inversion --svd --blend --alpha 0.002 \
    --seed 42 --verbose

# ────────────────────────────────────────────
# 实验4: α=0.002 SVD+FI (λ=0.05)
# ────────────────────────────────────────────
echo ""
echo ">>> [4/6] α=0.002, SVD+FI (λ=0.05)"
python run.py \
    --data_dir data/videos \
    --caption_dir $CAPTION_DIR \
    --output_dir outputs/ablation_a002_SVD_FI_5cases \
    --sample_ids $CASE_IDS \
    --inversion --svd --blend --alpha 0.002 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose

# ────────────────────────────────────────────
# 实验5: α=0.001 SVD+FI 降λ=0.03
# ────────────────────────────────────────────
echo ""
echo ">>> [5/6] α=0.001, SVD+FI (λ=0.03)"
python run.py \
    --data_dir data/videos \
    --caption_dir $CAPTION_DIR \
    --output_dir outputs/ablation_a001_SVD_FI_lam003_5cases \
    --sample_ids $CASE_IDS \
    --inversion --svd --blend --alpha 0.001 \
    --feature_inject --fi_layers mid --fi_lambda 0.03 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose

# ────────────────────────────────────────────
# 评估全部实验
# ────────────────────────────────────────────
echo ""
echo ">>> [6/6] Evaluating all experiments..."

for dir in ablation_a001_SVD_5cases \
           ablation_a001_SVD_FI_5cases \
           ablation_a002_SVD_5cases \
           ablation_a002_SVD_FI_5cases \
           ablation_a001_SVD_FI_lam003_5cases; do
    echo ""
    echo "====== Evaluating: $dir ======"
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/$dir \
        --caption-dir $CAPTION_DIR \
        --output-dir outputs/$dir/eval_clip
done

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "  Results in outputs/ablation_*/eval_clip/"
echo "=========================================="
