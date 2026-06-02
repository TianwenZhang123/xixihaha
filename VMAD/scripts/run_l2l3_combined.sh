#!/bin/bash
# ================================================================
# VMAD L2+L3: es=0.1 + blend_alpha=0.001
# ================================================================
#
# 背景：
#   es=0.005 在 10 样本上几乎无增益（CLIP -0.0048）
#   加大 L2 强度 (es=0.1, 注入 ≈ 3.3) + 同时开 L3 噪声先验
#
# 对照：使用之前已跑的 ctrl 组（l2_direct_validation/ctrl, CLIP=0.8840）
#
# 预估时间: 10 sample × 2min × 1组 ≈ 20 分钟
# 重要: 使用固定 seed=42，与历史基线对齐
# ================================================================

set -e
export OMP_NUM_THREADS=4

# ─── 环境变量 ───
VMAD_DIR="/root/autodl-tmp/videofake/VMAD"
VIDEO_DIR="/root/autodl-tmp/data/video-200/water_mark_out"
CAPTION_DIR="/root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0"
ASSETS_DIR="/root/autodl-tmp/outputs/vmad_v4_10samples/assets"
OUTPUT="/root/autodl-tmp/outputs/l2l3_combined/l2_es0.1_l3_b0.001"
SAMPLE_IDS="7 17 21 31 32 33 34 43 46 47"

cd "$VMAD_DIR"

echo "================================================================"
echo "VMAD L2+L3: es=0.1 + blend_alpha=0.001 - $(date)"
echo "================================================================"
echo "Ctrl baseline (ref): CLIP=0.8840, XCLIP=0.7137"
echo ""

mkdir -p "$OUTPUT"
if [ ! -e "$OUTPUT/assets" ]; then
    ln -sfn "$ASSETS_DIR" "$OUTPUT/assets"
fi

python run_batch_extract.py \
    --video-dir "$VIDEO_DIR" \
    --caption-dir "$CAPTION_DIR" \
    --content SELF --apply-only --no-token_decode \
    --embed-strength 0.1 --blend-alpha 0.001 \
    --sample-ids $SAMPLE_IDS \
    --seed 42 --resume -v \
    --output-dir "$OUTPUT"

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT/eval" \
    --method-name "L2es0.1_L3b0.001"

echo ""
echo "================================================================"
echo "Done - $(date)"
echo "Results: $OUTPUT/eval/eval_summary.md"
echo "================================================================"
cat "$OUTPUT/eval/eval_summary.md"
