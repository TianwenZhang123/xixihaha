#!/bin/bash
# ================================================================
# VMAD Layer 3 (SVD Blend) 单独验证
# ================================================================
#
# 只开 Layer 3 噪声先验（η_motion blend），不开 Layer 2
# 验证修复后的 _get_blended_latents() (P-Flow style) 是否有效
#
# 对照：使用之前已跑的 ctrl 组（l2_direct_validation/ctrl, CLIP=0.8840）
#
# 预估时间: 10 sample × 2min ≈ 20 分钟
# 重要: 使用固定 seed=42，与历史基线对齐
# ================================================================

set -e
export OMP_NUM_THREADS=4

# ─── 环境变量 ───
VMAD_DIR="/root/autodl-tmp/videofake/VMAD"
VIDEO_DIR="/root/autodl-tmp/data/video-200/water_mark_out"
CAPTION_DIR="/root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0"
ASSETS_DIR="/root/autodl-tmp/outputs/vmad_v4_10samples/assets"
OUTPUT="/root/autodl-tmp/outputs/l2l3_combined/l3_only_b0.001"
SAMPLE_IDS="7 17 21 31 32 33 34 43 46 47"

cd "$VMAD_DIR"

echo "================================================================"
echo "VMAD Layer 3 Only: blend_alpha=0.001 - $(date)"
echo "================================================================"
echo "Ctrl baseline (ref): CLIP=0.8840, XCLIP=0.7137"
echo ""

mkdir -p "$OUTPUT"
if [ ! -e "$OUTPUT/assets" ]; then
    ln -sfn "$ASSETS_DIR" "$OUTPUT/assets"
fi

# --no-velocity: 不注入 Layer 2 (Δe)
# 不加 --no-blend: 开启 Layer 3 噪声先验
# --blend-alpha 0.001: 默认 blend 强度
python run_batch_extract.py \
    --video-dir "$VIDEO_DIR" \
    --caption-dir "$CAPTION_DIR" \
    --content SELF --apply-only --no-token_decode \
    --no-velocity \
    --blend-alpha 0.001 \
    --sample-ids $SAMPLE_IDS \
    --seed 42 --resume -v \
    --output-dir "$OUTPUT"

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT/eval" \
    --method-name "L3_blend0.001"

echo ""
echo "================================================================"
echo "Done - $(date)"
echo "Results: $OUTPUT/eval/eval_summary.md"
echo "================================================================"
cat "$OUTPUT/eval/eval_summary.md"
