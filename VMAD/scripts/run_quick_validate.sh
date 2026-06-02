#!/bin/bash
# ================================================================
# VMAD Layer 2 修复验证 — 快速单样本测试
# ================================================================
#
# 目的：用 sample #7 快速验证 norm-adaptive 修复是否正确
# 预期结果：
#   - ctrl 组（无 L2）：CLIP ≈ 0.9397, XCLIP ≈ 0.7260
#     （与之前 V4 caption 纯净基线一致）
#   - es=0.005 组：CLIP ≈ 0.9446, XCLIP ≈ 0.7541
#     （复现历史 alpha=0.005 的结果）
#
# 如果两个数字对上 → 修复正确 → 可以扩到 10 样本
# 如果 ctrl 不对 → pipeline 其他地方有改动
# 如果 es=0.005 不对 → 需要检查注入逻辑
#
# 预估时间：2 个视频生成 ≈ 5-6 分钟
# ================================================================

set -e
export OMP_NUM_THREADS=4

# ─── 路径配置 ───
VMAD_DIR="/root/autodl-tmp/videofake/VMAD"
VIDEO_DIR="/root/autodl-tmp/data/video-200/water_mark_out"
CAPTION_DIR="/root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0"
ASSETS_DIR="/root/autodl-tmp/outputs/vmad_v4_10samples/assets"
OUTPUT_BASE="/root/autodl-tmp/outputs/quick_validate"

# 只用 sample #7（历史验证锚点）
SAMPLE_ID="7"

cd "$VMAD_DIR"

echo "================================================================"
echo "VMAD Quick Validation — $(date)"
echo "Sample: #${SAMPLE_ID} only"
echo "Fix: hidden_states += embed_strength * delta_e (no norm-adaptive)"
echo "================================================================"
echo ""

# ─── 公共参数 ───
COMMON_ARGS="--video-dir $VIDEO_DIR \
    --caption-dir $CAPTION_DIR \
    --content SELF \
    --apply-only \
    --per-sample-seed \
    --no-token_decode \
    --no-blend \
    --sample-ids $SAMPLE_ID \
    --seed 42 \
    --resume \
    -v"

# ================================================================
# Test 1: 控制组 — 无 L2，验证基线不变
# ================================================================
OUTPUT_CTRL="$OUTPUT_BASE/ctrl"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Test 1] Control: V4 caption, no L2 (expect CLIP≈0.9397)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "$OUTPUT_CTRL"
if [ ! -e "$OUTPUT_CTRL/assets" ]; then
    ln -sfn "$ASSETS_DIR" "$OUTPUT_CTRL/assets"
fi

python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_CTRL" \
    --no-velocity

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_CTRL/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_CTRL/eval" \
    --method-name "ctrl_no_L2"

echo ""

# ================================================================
# Test 2: L2 embed_strength=0.005 — 验证修复后能复现历史最优
# ================================================================
OUTPUT_ES005="$OUTPUT_BASE/es0.005"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Test 2] L2 direct: embed_strength=0.005 (expect CLIP≈0.9446)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "$OUTPUT_ES005"
if [ ! -e "$OUTPUT_ES005/assets" ]; then
    ln -sfn "$ASSETS_DIR" "$OUTPUT_ES005/assets"
fi

python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_ES005" \
    --embed-strength 0.005

python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_ES005/generated" \
    --caption-dir "$CAPTION_DIR" \
    --output-dir "$OUTPUT_ES005/eval" \
    --method-name "L2_direct_es0.005"

echo ""

# ================================================================
# 对比结果
# ================================================================
echo "================================================================"
echo "         Quick Validation Results"
echo "================================================================"
echo ""
echo "Expected values (from historical experiments):"
echo "  ctrl:     CLIP=0.9397, XCLIP=0.7260"
echo "  es=0.005: CLIP=0.9446, XCLIP=0.7541"
echo ""
echo "Actual results:"
echo "─────────────────────────────────────────────────────"

for EXP_DIR in "$OUTPUT_BASE"/*/eval; do
    if [ -f "$EXP_DIR/eval_summary.md" ]; then
        METHOD=$(basename $(dirname "$EXP_DIR"))
        CLIP=$(grep "orig_gen_clip" "$EXP_DIR/eval_summary.md" | grep -v "text" | head -1 | awk -F'|' '{gsub(/[* ]/,"",$3); print $3}')
        XCLIP=$(grep "orig_gen_xclip" "$EXP_DIR/eval_summary.md" | grep -v "text" | head -1 | awk -F'|' '{gsub(/[* ]/,"",$3); print $3}')
        echo "  $METHOD: CLIP=$CLIP, XCLIP=$XCLIP"
    fi
done

echo ""
echo "─────────────────────────────────────────────────────"
echo "判断标准:"
echo "  ✓ ctrl CLIP 与 0.9397 差距 < 0.005 → pipeline 正确"
echo "  ✓ es0.005 CLIP > ctrl CLIP → L2 注入有正向增益"
echo "  ✓ es0.005 接近 0.9446 → 修复成功，可扩到全样本"
echo ""
echo "如果 ctrl 偏差大: 检查 seed 策略 (--per-sample-seed)"
echo "如果 es0.005 无增益: 检查 hook 是否正确挂载"
echo "================================================================"
echo "Validation complete — $(date)"
echo "================================================================"
