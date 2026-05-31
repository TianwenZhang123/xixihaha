#!/bin/bash
# Layer 3 消融实验 + Alpha 精扫
# 预计总耗时：~7 分钟（8 个 apply-only 实验）
# 输出目录：/root/autodl-tmp/outputs111/

set -e
cd /root/autodl-tmp/videofake/VMAD
OUTBASE=/root/autodl-tmp/outputs111
mkdir -p ${OUTBASE}

echo "=========================================="
echo "开始时间: $(date)"
echo "=========================================="

# ============================================================
# E) Baseline caption + Layer 3 only
# ============================================================
echo ""
echo "=== [E] Baseline caption + Layer 3 only ==="
outdir=${OUTBASE}/vmad_l3_only
mkdir -p ${outdir}
ln -sf /root/autodl-tmp/outputs/vmad_phase3a_full/assets ${outdir}/assets

python run_batch_extract.py \
    --apply-only \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir} \
    --content SELF --sample-ids 7 \
    --no-velocity --no-token_decode \
    --alpha 0.001 --seed 42 -v

python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${outdir}/generated \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir}/eval --limit 1

# ============================================================
# F) Baseline caption + Layer 2 + Layer 3
# ============================================================
echo ""
echo "=== [F] Baseline caption + Layer 2 + Layer 3 ==="
outdir=${OUTBASE}/vmad_l2l3
mkdir -p ${outdir}
ln -sf /root/autodl-tmp/outputs/vmad_phase3a_full/assets ${outdir}/assets

python run_batch_extract.py \
    --apply-only \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir} \
    --content SELF --sample-ids 7 \
    --no-token_decode \
    --alpha 0.01 --seed 42 -v

python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${outdir}/generated \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir}/eval --limit 1

# ============================================================
# G) V4 caption + Layer 3 only
# ============================================================
echo ""
echo "=== [G] V4 caption + Layer 3 only ==="
outdir=${OUTBASE}/vmad_v4caption_l3
mkdir -p ${outdir}
ln -sf /root/autodl-tmp/outputs/vmad_v4caption_l2/assets ${outdir}/assets

python run_batch_extract.py \
    --apply-only \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir} \
    --content SELF --sample-ids 7 \
    --no-velocity --no-token_decode \
    --alpha 0.001 --seed 42 -v

python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${outdir}/generated \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir}/eval --limit 1

# ============================================================
# H) V4 caption + Layer 2 + Layer 3 (三层全开)
# ============================================================
echo ""
echo "=== [H] V4 caption + Layer 2 + Layer 3 (三层全开) ==="
outdir=${OUTBASE}/vmad_v4caption_l2l3
mkdir -p ${outdir}
ln -sf /root/autodl-tmp/outputs/vmad_v4caption_l2/assets ${outdir}/assets

python run_batch_extract.py \
    --apply-only \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir} \
    --content SELF --sample-ids 7 \
    --no-token_decode \
    --alpha 0.01 --seed 42 -v

python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${outdir}/generated \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir}/eval --limit 1

# ============================================================
# I-L) Alpha 精扫 (V4 caption + Layer 2 only, 不开 blend)
# ============================================================
echo ""
echo "=== [I-L] V4 caption + Layer 2 Alpha 精扫 ==="
for alpha in 0.005 0.008 0.015 0.02; do
    echo ""
    echo "--- Alpha = ${alpha} ---"
    outdir=${OUTBASE}/vmad_v4_alpha_${alpha}
    mkdir -p ${outdir}
    ln -sf /root/autodl-tmp/outputs/vmad_v4caption_l2/assets ${outdir}/assets

    python run_batch_extract.py \
        --apply-only \
        --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
        --output-dir ${outdir} \
        --sample-ids 7 \
        --alpha ${alpha} \
        --no-blend --no-token_decode \
        --content SELF --seed 42 -v

    python evaluation/run_reproduction_eval.py \
        --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --gen-dir ${outdir}/generated \
        --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
        --output-dir ${outdir}/eval --limit 1
done

# ============================================================
# 汇总结果
# ============================================================
echo ""
echo "=========================================="
echo "=== 全部完成！结果汇总 ==="
echo "=========================================="
echo ""
echo "| 实验 | 配置 | CLIP | XCLIP |"
echo "|------|------|------|-------|"
for d in vmad_l3_only vmad_l2l3 vmad_v4caption_l3 vmad_v4caption_l2l3 vmad_v4_alpha_0.005 vmad_v4_alpha_0.008 vmad_v4_alpha_0.015 vmad_v4_alpha_0.02; do
    eval_file="${OUTBASE}/${d}/eval/metrics_summary.json"
    if [ -f "${eval_file}" ]; then
        clip=$(python -c "import json; d=json.load(open('${eval_file}')); print(f\"{d.get('orig_gen_clip_mean', d.get('orig_gen_clip', 'N/A')):.4f}\")" 2>/dev/null || echo "N/A")
        xclip=$(python -c "import json; d=json.load(open('${eval_file}')); print(f\"{d.get('orig_gen_xclip_mean', d.get('orig_gen_xclip', 'N/A')):.4f}\")" 2>/dev/null || echo "N/A")
        echo "| ${d} | - | ${clip} | ${xclip} |"
    else
        echo "| ${d} | - | MISSING | MISSING |"
    fi
done

echo ""
echo "结束时间: $(date)"
echo "=========================================="
