#!/bin/bash
# ================================================================
# VMAD Layer 2 直接缩放验证实验
# ================================================================
#
# 背景：
#   fix_validation 轮次中 norm-adaptive scaling 全部失败（CLIP 暴跌）。
#   根因: norm_scale = ||e0||/||de|| ≈ 1448/36 = 40x 放大，
#   导致 es=0.01 的实际注入是旧方法 alpha=0.005 的 80 倍。
#
#   修复: 去掉 norm-adaptive scaling，回退到直接缩放：
#     hidden_states += embed_strength * delta_e
#   其中 embed_strength = 0.005 已在 sample #7 上验证 CLIP=0.9446
#
# 本轮实验：
#   Exp 0: 控制组（无 L2 无 L3），确认基线 CLIP≈0.8842
#   Exp 1: L2 embed_strength=0.005（★历史最优 CLIP）
#   Exp 2: L2 embed_strength=0.008（★历史最优 XCLIP）
#   Exp 3: L2 embed_strength=0.003（边界探索）
#   Exp 4: L2 embed_strength=0.01（对照）
#
# 预估时间: 10 sample × 2.7min × 5组 ≈ 2.3 小时
#
# 成功标准:
#   - Exp 1 CLIP ≥ 0.92, XCLIP ≥ 0.74 （多样本上复现 sample #7 的增益方向）
#   - 任一实验 CLIP > 0.8842 （超过控制组）
# ================================================================

set -e
export OMP_NUM_THREADS=4

# ─── 环境变量 ───
VMAD_DIR="/root/autodl-tmp/videofake/VMAD"
VIDEO_DIR="/root/autodl-tmp/data/video-200/water_mark_out"
CAPTION_DIR="/root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0"
ASSETS_DIR="/root/autodl-tmp/outputs/vmad_v4_10samples/assets"
OUTPUT_BASE="/root/autodl-tmp/outputs/l2_direct_validation"
SAMPLE_IDS="7 17 21 31 32 33 34 43 46 47"

# ─── 断点续跑 ───
START_FROM=${1:-0}
if [[ "$1" == "--from" ]]; then
    START_FROM=${2:-0}
fi

exp_done() {
    local exp_dir="$1"
    [ -f "$exp_dir/eval/eval_summary.md" ]
}

prepare_output() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    if [ ! -e "$out_dir/assets" ]; then
        ln -sfn "$ASSETS_DIR" "$out_dir/assets"
    fi
}

COMMON_ARGS="--video-dir $VIDEO_DIR \
    --caption-dir $CAPTION_DIR \
    --content SELF \
    --apply-only \
    --per-sample-seed \
    --no-token_decode \
    --no-blend \
    --sample-ids $SAMPLE_IDS \
    --seed 42 \
    --resume \
    -v"

cd "$VMAD_DIR"
mkdir -p logs

echo "================================================================"
echo "VMAD Layer 2 Direct Scaling Validation - $(date)"
echo "Fix: removed norm-adaptive (40x amplification)"
echo "Now: hidden_states += embed_strength * delta_e"
echo "================================================================"
echo ""

# ================================================================
# Exp 0: 控制组
# ================================================================
OUTPUT_0="$OUTPUT_BASE/exp0_ctrl"
if [[ $START_FROM -le 0 ]] && ! exp_done "$OUTPUT_0"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 0] Control: no L2, no L3 (expect CLIP≈0.8842)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
prepare_output "$OUTPUT_0"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_0" \
    --no-velocity

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_0/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_0/eval" \
    --method-name "ctrl_no_L2_no_L3"
else
echo "[Exp 0] SKIP (already done or --from > 0)"
fi
echo ""

# ================================================================
# Exp 1: embed_strength=0.005 (★历史 CLIP 最优)
# ================================================================
OUTPUT_1="$OUTPUT_BASE/exp1_l2_es0.005"
if [[ $START_FROM -le 1 ]] && ! exp_done "$OUTPUT_1"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 1] L2 direct: embed_strength=0.005 (★CLIP optimal on #7)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
prepare_output "$OUTPUT_1"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_1" \
    --embed-strength 0.005

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_1/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_1/eval" \
    --method-name "L2_direct_es0.005"
else
echo "[Exp 1] SKIP (already done or --from > 1)"
fi
echo ""

# ================================================================
# Exp 2: embed_strength=0.008 (★历史 XCLIP 最优)
# ================================================================
OUTPUT_2="$OUTPUT_BASE/exp2_l2_es0.008"
if [[ $START_FROM -le 2 ]] && ! exp_done "$OUTPUT_2"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 2] L2 direct: embed_strength=0.008 (★XCLIP optimal on #7)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
prepare_output "$OUTPUT_2"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_2" \
    --embed-strength 0.008

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_2/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_2/eval" \
    --method-name "L2_direct_es0.008"
else
echo "[Exp 2] SKIP (already done or --from > 2)"
fi
echo ""

# ================================================================
# Exp 3: embed_strength=0.003 (边界探索)
# ================================================================
OUTPUT_3="$OUTPUT_BASE/exp3_l2_es0.003"
if [[ $START_FROM -le 3 ]] && ! exp_done "$OUTPUT_3"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 3] L2 direct: embed_strength=0.003 (lower bound)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
prepare_output "$OUTPUT_3"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_3" \
    --embed-strength 0.003

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_3/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_3/eval" \
    --method-name "L2_direct_es0.003"
else
echo "[Exp 3] SKIP (already done or --from > 3)"
fi
echo ""

# ================================================================
# Exp 4: embed_strength=0.01 (对照)
# ================================================================
OUTPUT_4="$OUTPUT_BASE/exp4_l2_es0.01"
if [[ $START_FROM -le 4 ]] && ! exp_done "$OUTPUT_4"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 4] L2 direct: embed_strength=0.01 (upper reference)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
prepare_output "$OUTPUT_4"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_4" \
    --embed-strength 0.01

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_4/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_4/eval" \
    --method-name "L2_direct_es0.01"
else
echo "[Exp 4] SKIP (already done or --from > 4)"
fi
echo ""

# ================================================================
# 汇总结果
# ================================================================
echo ""
echo "================================================================"
echo "         L2 Direct Scaling Validation Results"
echo "================================================================"
echo ""
echo "| Experiment | embed_strength | CLIP | XCLIP | vs Ctrl |"
echo "|------------|----------------|------|-------|---------|"

CTRL_CLIP=""
for EXP_DIR in "$OUTPUT_BASE"/exp*/eval; do
    if [ -f "$EXP_DIR/eval_summary.md" ]; then
        CLIP=$(grep "orig_gen_clip" "$EXP_DIR/eval_summary.md" | grep -v "text" | head -1 | awk -F'|' '{gsub(/[* ]/,"",$3); print $3}')
        XCLIP=$(grep "orig_gen_xclip" "$EXP_DIR/eval_summary.md" | grep -v "text" | head -1 | awk -F'|' '{gsub(/[* ]/,"",$3); print $3}')
        METHOD=$(grep "Method:" "$EXP_DIR/eval_summary.md" | awk -F': ' '{print $2}' | tr -d ' ')
        EXP_NAME=$(basename $(dirname "$EXP_DIR"))

        if [[ "$EXP_NAME" == "exp0_ctrl" ]]; then
            CTRL_CLIP="$CLIP"
            echo "| $EXP_NAME | — | $CLIP | $XCLIP | baseline |"
        else
            echo "| $EXP_NAME | — | $CLIP | $XCLIP | $METHOD |"
        fi
    fi
done

echo ""
echo "Historical reference (sample #7 only, V4 caption + old alpha method):"
echo "  alpha=0.005 → CLIP=0.9446, XCLIP=0.7541"
echo "  alpha=0.008 → CLIP=0.9395, XCLIP=0.7581"
echo ""
echo "================================================================"
echo "Validation complete - $(date)"
echo "================================================================"
