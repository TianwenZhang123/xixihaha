# P-Flow: Full Paper Reproduction

**Paper:** P-Flow: Training-Free Visual Effects Customization via Test-Time Prompt Optimization (arXiv:2603.22091)

**Target Hardware:** Single A800 (80GB)

---

## Overview

This is a complete reproduction of the P-Flow framework, implementing Algorithm 1 with:

- **Video Model:** Wan 2.1-14B (T2V and I2V), single GPU + CPU offload
- **VLM:** DashScope Qwen-VL (qwen-vl-max)
- **Dataset:** Open-VFX (15 categories, 1003 samples)
- **Evaluation:** FID-VID, FVD, Dynamic Degree, Human Evaluation
- **Processing:** One video at a time (sequential)

## Directory Structure

```
P-Flow/
├── src/                    # Core source code
│   ├── pipeline.py         # Algorithm 1 main orchestration
│   ├── noise_prior.py      # Flow Inversion + SVD + Blending
│   ├── svd_filter.py       # Two-stage SVD filtering
│   ├── flow_matching.py    # Euler ODE inversion (t=1→0)
│   ├── vlm_client.py       # DashScope Qwen-VL client
│   ├── prompt_optimizer.py # Vertical composite + VLM refinement
│   ├── video_utils.py      # Video I/O and processing
│   ├── trajectory.py       # History management
│   └── distributed.py      # Single-GPU inference utilities
├── configs/
│   ├── paper_default.yaml  # Paper-exact settings (single A800)
│   └── ablation.yaml       # Ablation experiment configs
├── scripts/
│   ├── prepare_dataset.py  # Open-VFX dataset preparation
│   └── run_experiment.py   # Experiment runner (single/batch)
├── evaluation/
│   ├── metrics.py          # FID-VID, FVD, Dynamic Degree
│   ├── human_eval.py       # Human evaluation tools
│   └── run_evaluation.py   # Complete evaluation pipeline
└── requirements.txt        # Python dependencies
```

## Quick Start (Single A800 Deployment)

### 1. Environment Setup

```bash
# On A800 machine
conda create -n pflow python=3.10 -y
conda activate pflow

# Clone and enter project
cd /root/autodl-tmp
git clone <repo_url> videofake
cd videofake/P-Flow

# Install dependencies
pip install -r requirements.txt

# Install flash-attention for A800
pip install flash-attn --no-build-isolation
```

### 2. Download Models

```bash
# Wan 2.1-14B T2V
huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-14B

# Wan 2.1-14B I2V (optional, for I2V experiments)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B \
    --local-dir /root/autodl-tmp/models/Wan2.1-I2V-14B
```

### 3. Configure API Keys

```bash
# DashScope Qwen-VL (primary VLM)
export DASHSCOPE_API_KEY="your-dashscope-key"
```

### 4. Prepare Dataset

```bash
python scripts/prepare_dataset.py \
    --output_dir /root/autodl-tmp/datasets/Open-VFX

# If you have raw videos:
python scripts/prepare_dataset.py \
    --output_dir /root/autodl-tmp/datasets/Open-VFX \
    --preprocess \
    --input_dir /path/to/raw/videos
```

### 5. Run Experiment (One Video at a Time)

```bash
# Single sample — this is the primary usage mode
python scripts/run_experiment.py \
    --video /root/autodl-tmp/datasets/Open-VFX/videos/fire_effects/fire_0001.mp4 \
    --prompt "A campfire with dancing orange flames creating swirling patterns" \
    --output_dir /root/autodl-tmp/outputs/pflow/test_fire_0001 \
    --seed 42

# Quick run entry point
python run.py \
    --video /path/to/reference.mp4 \
    --prompt "description of effect" \
    --output /root/autodl-tmp/outputs/pflow/my_test \
    --seed 42

# With mock VLM (for testing without API key)
python run.py \
    --video reference.mp4 \
    --prompt "fire effect" \
    --mock_vlm

# Batch processing (one by one, sequentially)
python scripts/run_experiment.py \
    --dataset /root/autodl-tmp/datasets/Open-VFX \
    --split test \
    --output_dir /root/autodl-tmp/outputs/pflow/full_test \
    --seed 42
```

### 6. Run Evaluation

```bash
# Single experiment
python evaluation/run_evaluation.py \
    --experiment_dir /root/autodl-tmp/outputs/pflow/test_fire_0001 \
    --reference_video /root/autodl-tmp/datasets/Open-VFX/videos/fire_effects/fire_0001.mp4

# Select best iteration
python evaluation/run_evaluation.py \
    --select_best \
    --experiment_dir /root/autodl-tmp/outputs/pflow/test_fire_0001 \
    --metric dynamic
```

## Paper Settings Reference

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Wan 2.1-14B | Section 3.6 |
| VLM | Qwen-VL (DashScope) | Adapted |
| Resolution | 480×832 | Section 3.6 |
| Frames | 81 (5s @ 16fps) | Section 3.6 |
| Guidance Scale | 5.0 | Section 3.6 |
| Inference Steps | 50 | Section 3.6 |
| α (noise blend) | 0.001 | Eq. 7 |
| ρ_s (spatial SVD) | 0.1 | Eq. 4 |
| ρ_m (temporal SVD) | 0.9 | Eq. 6 |
| i_max (iterations) | 10 | Algorithm 1 |
| Inversion steps | 50 | Section 3.3 |

## Performance (Single A800-80GB)

| Stage | Time |
|-------|------|
| Video generation | ~90-120s/video |
| VLM inference (Qwen-VL) | ~5-10s |
| Total per iteration | ~100-130s |
| Total per sample (10 iter) | ~17-22 min |
| Peak VRAM | ~40-50GB |
| CPU RAM needed | ~32GB+ |

## Ablation Experiments

```bash
# Without noise prior (random noise only)
python scripts/run_experiment.py \
    --config configs/ablation.yaml \
    --video /path/to/video.mp4 \
    --prompt "description" \
    --output_dir /root/autodl-tmp/outputs/ablation/no_noise_prior

# Without VLM (noise prior only, no prompt optimization)
python scripts/run_experiment.py \
    --config configs/paper_default.yaml \
    --video /path/to/video.mp4 \
    --prompt "description" \
    --mock_vlm \
    --output_dir /root/autodl-tmp/outputs/ablation/no_vlm
```

See `configs/ablation.yaml` for all ablation configurations matching Tables 3 and 4.

## Key Implementation Notes

1. **Single GPU:** Model runs on one A800 with `enable_model_cpu_offload()`, ~40-50GB peak VRAM
2. **Sequential Processing:** Videos are processed one at a time
3. **No Early Stopping:** Paper explicitly uses fixed 10 iterations (noted as limitation)
4. **No Confidence Score:** VLM output is {analysis, refined_prompt} only
5. **Offline Selection:** Best video selected by metrics, not during optimization
6. **Vertical Composite:** VLM sees [ref | prev | current] stacked vertically
7. **Per-iteration Blending:** Fresh noise blended each iteration for exploration
8. **Paper uses P_0 for inversion:** Not empty prompt, but initial user prompt
