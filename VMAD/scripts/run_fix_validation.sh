#!/bin/bash
# ================================================================
# VMAD Layer 2 & Layer 3 修复验证实验
# ================================================================
#
# 验证目的：
#   本脚本包含 5 组实验，逐步验证 Layer 2 和 Layer 3 的修复是否正确。
#
#   实验 1 (路径修复验证): blend_alpha 极小 + 不注入 Δe
#       意义: 当 blend_alpha=0.0001 且 embed_strength=0 时，结果应该
#             与纯 P-Flow 控制组 (CLIP=0.8842, XCLIP=0.7431) 几乎完全一致。
#             如果不一致，说明 _prepare_blended_latents 的随机数路径仍有问题。
#       成功标准: |CLIP - 0.8842| < 0.002 且 |XCLIP - 0.7431| < 0.005
#
#   实验 2 (Layer 2 单独验证): 只开 Layer 2，不开 Layer 3
#       意义: 在 Layer 3 关闭的情况下，单独测试 norm-adaptive Δe 注入。
#             embed_strength=0.1 意味着注入 10% 信号强度的扰动。
#             之前 Layer 2 失败是因为 alpha 共用 + 无归一化导致注入方向错误。
#             修复后应看到 XCLIP 正增益（哪怕很小）。
#       成功标准: XCLIP > 0.7431 (高于控制组)
#
#   实验 3 (Layer 3 单独验证): 只开 Layer 3，不开 Layer 2
#       意义: 在 Δe 关闭的情况下，单独测试修复后的 noise prior blend。
#             之前 Layer 3 导致 CLIP 暴跌 -0.10~-0.15，修复后应该只有
#             微小的正向或中性影响（blend_alpha=0.001 极小）。
#       成功标准: CLIP > 0.88 (不暴跌) 且 XCLIP ≥ 0.74
#
#   实验 4 (Layer 2 强度扫描): embed_strength 从 0.01 到 0.2
#       意义: 寻找 embed_strength 的最优值。过小无效果，过大引入噪声。
#             这将确定 norm-adaptive 缩放后的最佳注入比例。
#       成功标准: 存在某个值使 XCLIP > 控制组
#
#   实验 5 (Layer 2 + Layer 3 联合): 最优 embed_strength + blend_alpha=0.001
#       意义: 验证两层修复后能否叠加产生正增益。根据早期发现，
#             Layer 2 提升 XCLIP (时序)，Layer 3 提升 CLIP (外观)，
#             两者应互补而非冲突。
#       成功标准: CLIP ≥ 0.8842 且 XCLIP > 0.7431
#
# 使用方法:
#   1. SSH 到 AutoDL 服务器
#   2. cd ~/autodl-tmp/videofake/VMAD
#   3. bash scripts/run_fix_validation.sh 2>&1 | tee logs/fix_validation_$(date +%Y%m%d_%H%M).log
#
# 预估时间: 10 sample × 2.7min/sample × 7组 ≈ 3.2 小时
# ================================================================

set -e

# ─── 环境变量 ───
VMAD_DIR="$HOME/autodl-tmp/videofake/VMAD"
VIDEO_DIR="$HOME/autodl-tmp/videofake/P-Flow/data/generated_videos"
CAPTION_DIR="$HOME/autodl-tmp/videofake/P-Flow/data/captions_qwen"
APPLY_CAPTION_DIR="$HOME/autodl-tmp/videofake/P-Flow/data/captions_iter1"
OUTPUT_BASE="$HOME/autodl-tmp/videofake/VMAD/outputs/fix_validation"
SAMPLE_IDS="7 17 21 31 32 33 34 43 46 47"

# ─── 断点续跑支持 ───
# 用法: bash run_fix_validation.sh          # 从头跑（跳过已完成的实验）
#        bash run_fix_validation.sh --from 3 # 从 Exp 3 开始跑
START_FROM=${1:-0}
if [[ "$1" == "--from" ]]; then
    START_FROM=${2:-0}
fi

# 检查某个实验是否已完成（eval 输出存在）
exp_done() {
    local exp_dir="$1"
    if [ -f "$exp_dir/eval/eval_summary.md" ]; then
        return 0  # done
    fi
    return 1  # not done
}

# ─── 公共参数 ───
COMMON_ARGS="--video-dir $VIDEO_DIR \
    --caption-dir $CAPTION_DIR \
    --apply-caption-dir $APPLY_CAPTION_DIR \
    --content SELF \
    --apply-only \
    --per-sample-seed \
    --no-token_decode \
    --sample-ids $SAMPLE_IDS \
    --seed 42 \
    --resume \
    -v"

cd "$VMAD_DIR"
mkdir -p logs

echo "================================================================"
echo "VMAD Fix Validation - $(date)"
echo "Start from: Exp $START_FROM"
echo "================================================================"
echo ""

# ================================================================
# 实验 0: 控制组 (纯 P-Flow，无任何注入)
# ================================================================
# 目的: 建立 baseline，确认 per-sample-seed 对齐后控制组依然是 0.8842/0.7431
# 这是对比其他实验的基准。
OUTPUT_0="$OUTPUT_BASE/exp0_ctrl"
if [[ $START_FROM -le 0 ]] && ! exp_done "$OUTPUT_0"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 0] Control group: no L2, no L3 (pure P-Flow equivalent)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_0" \
    --no-velocity \
    --no-blend

echo "[Exp 0] Evaluating..."
python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_0/generated" \
    --caption-dir "$APPLY_CAPTION_DIR" \
    --output-dir "$OUTPUT_0/eval" \
    --method-name "ctrl_no_L2_no_L3"
else
echo "[Exp 0] SKIP (already done or --from > 0)"
fi
echo ""

# ================================================================
# 实验 1: Layer 3 路径修复验证 (blend_alpha 极小)
# ================================================================
# 目的: blend_alpha=0.0001 时应该 ≈ 不开 blend
# 如果结果和 Exp 0 一致，说明 prepare_latents 路径修复正确
OUTPUT_1="$OUTPUT_BASE/exp1_l3_path_verify"
if [[ $START_FROM -le 1 ]] && ! exp_done "$OUTPUT_1"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 1] Layer 3 path fix verify: blend_alpha=0.0001, no L2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_1" \
    --no-velocity \
    --blend-alpha 0.0001

echo "[Exp 1] Evaluating..."
python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_1/generated" \
    --caption-dir "$APPLY_CAPTION_DIR" \
    --output-dir "$OUTPUT_1/eval" \
    --method-name "L3_path_verify_ba0001"
else
echo "[Exp 1] SKIP (already done or --from > 1)"
fi
echo ""

# ================================================================
# 实验 2: Layer 2 单独验证 (norm-adaptive, embed_strength=0.1)
# ================================================================
# 目的: 单独测试修复后的 Layer 2 注入
# embed_strength=0.1 = 注入 ||e0|| 的 10% 强度
# 不开 Layer 3 以隔离变量
OUTPUT_2="$OUTPUT_BASE/exp2_l2_only_es01"
if [[ $START_FROM -le 2 ]] && ! exp_done "$OUTPUT_2"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 2] Layer 2 only: embed_strength=0.1, no L3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_2" \
    --no-blend \
    --embed-strength 0.1

echo "[Exp 2] Evaluating..."
python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_2/generated" \
    --caption-dir "$APPLY_CAPTION_DIR" \
    --output-dir "$OUTPUT_2/eval" \
    --method-name "L2_only_es0.1"
else
echo "[Exp 2] SKIP (already done or --from > 2)"
fi
echo ""

# ================================================================
# 实验 3: Layer 3 单独验证 (blend_alpha=0.001, 无 Layer 2)
# ================================================================
# 目的: 测试修复后的 noise prior 是否不再暴跌
# 之前: Layer 3 → CLIP 暴跌 -0.10~-0.15
# 修复后预期: CLIP 几乎不变或微升
OUTPUT_3="$OUTPUT_BASE/exp3_l3_only_ba001"
if [[ $START_FROM -le 3 ]] && ! exp_done "$OUTPUT_3"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 3] Layer 3 only: blend_alpha=0.001, no L2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_3" \
    --no-velocity \
    --blend-alpha 0.001

echo "[Exp 3] Evaluating..."
python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_3/generated" \
    --caption-dir "$APPLY_CAPTION_DIR" \
    --output-dir "$OUTPUT_3/eval" \
    --method-name "L3_only_ba0.001"
else
echo "[Exp 3] SKIP (already done or --from > 3)"
fi
echo ""

# ================================================================
# 实验 4: Layer 2 强度扫描 (embed_strength sweep)
# ================================================================
# 目的: 寻找 embed_strength 最优值
# 扫描范围: 0.01, 0.05, 0.1, 0.2
# 过小 → 信号太弱无效果; 过大 → 扰动太强破坏质量
if [[ $START_FROM -le 4 ]]; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 4] Layer 2 strength sweep: es=0.01, 0.05, 0.2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for ES in 0.01 0.05 0.2; do
    OUTPUT_4="$OUTPUT_BASE/exp4_l2_es${ES}"
    if exp_done "$OUTPUT_4"; then
        echo "  [Exp 4] es=$ES SKIP (already done)"
        continue
    fi
    echo "  [Exp 4] embed_strength=$ES ..."
    python run_batch_extract.py \
        $COMMON_ARGS \
        --output-dir "$OUTPUT_4" \
        --no-blend \
        --embed-strength "$ES"

    python evaluation/run_reproduction_eval.py \
        --orig-dir "$VIDEO_DIR" \
        --gen-dir "$OUTPUT_4/generated" \
        --caption-dir "$APPLY_CAPTION_DIR" \
        --output-dir "$OUTPUT_4/eval" \
        --method-name "L2_es${ES}"
done
else
echo "[Exp 4] SKIP (--from > 4)"
fi
echo ""

# ================================================================
# 实验 5: Layer 2 + Layer 3 联合 (最优 embed_strength + blend_alpha)
# ================================================================
# 目的: 验证两层修复后叠加效果
# 理论预期: L2 提升 XCLIP (时序), L3 提升 CLIP (外观), 互补正增益
OUTPUT_5="$OUTPUT_BASE/exp5_l2_l3_combined"
if [[ $START_FROM -le 5 ]] && ! exp_done "$OUTPUT_5"; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Exp 5] Layer 2 + Layer 3 combined"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_batch_extract.py \
    $COMMON_ARGS \
    --output-dir "$OUTPUT_5" \
    --embed-strength 0.1 \
    --blend-alpha 0.001

echo "[Exp 5] Evaluating..."
python evaluation/run_reproduction_eval.py \
    --orig-dir "$VIDEO_DIR" \
    --gen-dir "$OUTPUT_5/generated" \
    --caption-dir "$APPLY_CAPTION_DIR" \
    --output-dir "$OUTPUT_5/eval" \
    --method-name "L2_es0.1_L3_ba0.001"
else
echo "[Exp 5] SKIP (already done or --from > 5)"
fi
echo ""

# ================================================================
# 汇总结果
# ================================================================
echo ""
echo "================================================================"
echo "               汇总: 所有实验核心指标"
echo "================================================================"
echo ""
echo "| Experiment | CLIP | XCLIP | 说明 |"
echo "|------------|------|-------|------|"

for EXP_DIR in "$OUTPUT_BASE"/exp*/eval; do
    if [ -f "$EXP_DIR/eval_summary.md" ]; then
        # Extract CLIP and XCLIP from summary
        CLIP=$(grep "orig_gen_clip" "$EXP_DIR/eval_summary.md" | grep -v "text" | head -1 | awk -F'|' '{gsub(/[* ]/,"",$3); print $3}')
        XCLIP=$(grep "orig_gen_xclip" "$EXP_DIR/eval_summary.md" | grep -v "text" | head -1 | awk -F'|' '{gsub(/[* ]/,"",$3); print $3}')
        METHOD=$(grep "Method:" "$EXP_DIR/eval_summary.md" | awk -F': ' '{print $2}' | tr -d ' ')
        EXP_NAME=$(basename $(dirname "$EXP_DIR"))
        echo "| $EXP_NAME | $CLIP | $XCLIP | $METHOD |"
    fi
done

echo ""
echo "P-Flow Reference: CLIP=0.8842, XCLIP=0.7431 (per-sample-seed control)"
echo ""
echo "================================================================"
echo "验证完成 - $(date)"
echo "================================================================"
