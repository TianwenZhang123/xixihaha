#!/bin/bash
# ============================================================================
# P-Flow 逐层验证一键复现脚本
#
# 使用方法:
#   cd P-Flow
#   bash scripts/reproduce.sh
#
# 前提:
#   1. models/ 下已放好模型 (软链接或实际目录)
#   2. data/videos/ 下已放好参考视频 ({id}.mp4)
#   3. data/captions_qwen/ 下已放好 VLM 原始 caption ({id}.txt)
#   4. 已安装依赖: pip install -r requirements.txt
#   5. 已设置 DASHSCOPE_API_KEY 环境变量 (Layer 1 改写需要)
# ============================================================================

set -e

# ── 配置 (可按需修改) ──
DATA_DIR="data/videos"
CAPTION_QWEN="data/captions_qwen"
CAPTION_HYBRID="data/captions_hybrid"
OUTPUT_BASE="outputs"
SAMPLE_IDS="7 17 21 31 32 33 34 43 46 47"
SEED=42

echo "============================================"
echo " P-Flow 逐层验证复现"
echo " 样本: $SAMPLE_IDS"
echo " 种子: $SEED"
echo "============================================"
echo ""

# ── 检查前置条件 ──
if [ ! -d "models/Wan2.1-T2V-1.3B-Diffusers" ]; then
    echo "错误: models/Wan2.1-T2V-1.3B-Diffusers 不存在"
    echo "请先准备模型: mkdir -p models && ln -s /path/to/model models/Wan2.1-T2V-1.3B-Diffusers"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "错误: $DATA_DIR 不存在"
    echo "请先准备数据: mkdir -p data/videos && ln -sf /path/to/videos/*.mp4 data/videos/"
    exit 1
fi

if [ ! -d "$CAPTION_QWEN" ]; then
    echo "错误: $CAPTION_QWEN 不存在"
    echo "请先准备 VLM caption: mkdir -p data/captions_qwen"
    exit 1
fi

# ── Step 0: Baseline ──
echo "[Step 0] Baseline (VLM caption 直出)..."
python run.py \
    --data_dir $DATA_DIR \
    --caption_dir $CAPTION_QWEN \
    --output_dir $OUTPUT_BASE/step0_baseline \
    --sample_ids $SAMPLE_IDS \
    --seed $SEED --resume

# ── Step 1: Layer 1 (Hybrid Prompt Rewrite) ──
echo ""
echo "[Step 1] Layer 1: Hybrid Prompt Rewrite..."

# 1a. 生成 hybrid caption (一次性 LLM 改写，非迭代)
python scripts/rewrite_hybrid.py \
    --input-dir $CAPTION_QWEN \
    --output-dir $CAPTION_HYBRID \
    --backend dashscope --model qwen-plus \
    --sample-ids $SAMPLE_IDS \
    --skip-existing

# 1b. 用 hybrid caption 生成视频
python run.py \
    --data_dir $DATA_DIR \
    --caption_dir $CAPTION_HYBRID \
    --output_dir $OUTPUT_BASE/step1_L1_hybrid \
    --sample_ids $SAMPLE_IDS \
    --seed $SEED --resume

# ── Step 2: Layer 1 + Layer 2 (Noise Prior) ──
echo ""
echo "[Step 2] Layer 1 + Layer 2: SVD Noise Prior (alpha=0.004)..."
python run.py \
    --data_dir $DATA_DIR \
    --caption_dir $CAPTION_HYBRID \
    --output_dir $OUTPUT_BASE/step2_L1L2_noise_prior \
    --noise_prior --alpha 0.004 \
    --sample_ids $SAMPLE_IDS \
    --seed $SEED --resume

# ── Step 3a: L1 + L2 + L3v1 (Velocity) ──
echo ""
echo "[Step 3a] Layer 1 + Layer 2 + Layer 3v1: Velocity Matching..."
python run.py \
    --data_dir $DATA_DIR \
    --caption_dir $CAPTION_HYBRID \
    --output_dir $OUTPUT_BASE/step3_L1L2L3v1 \
    --velocity_full --alpha 0.004 --embed_strength 0.02 \
    --sample_ids $SAMPLE_IDS \
    --seed $SEED --resume

# ── Step 3b: L1 + L2 + L3v2 (Best Config) ──
echo ""
echo "[Step 3b] Layer 1 + Layer 2 + Layer 3v2: Best Config..."
python run.py \
    --data_dir $DATA_DIR \
    --caption_dir $CAPTION_HYBRID \
    --output_dir $OUTPUT_BASE/step3_L1L2L3v2_best \
    --velocity_full --alpha 0.004 --embed_strength 0.02 \
    --velocity_K 4 --velocity_motion_weight 1.0 \
    --sample_ids $SAMPLE_IDS \
    --seed $SEED --resume

# ── 统一评估 ──
echo ""
echo "============================================"
echo " 开始评估..."
echo "============================================"

for step_dir in step0_baseline step1_L1_hybrid step2_L1L2_noise_prior step3_L1L2L3v1 step3_L1L2L3v2_best; do
    echo ""
    echo ">>> 评估: $step_dir"
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir $DATA_DIR \
        --gen-dir $OUTPUT_BASE/$step_dir \
        --caption-dir $CAPTION_HYBRID \
        --output-dir $OUTPUT_BASE/$step_dir/eval_clip
done

echo ""
echo "============================================"
echo " 全部完成！"
echo " 结果保存在 $OUTPUT_BASE/ 各子目录的 eval_clip/ 下"
echo "============================================"
