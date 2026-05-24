# P-Flow: Video Reproduction via Test-Time Prompt Optimization

**Paper:** P-Flow: Training-Free Visual Effects Customization via Test-Time Prompt Optimization (arXiv:2603.22091)

**Hardware:** Single 4090 (24GB) — Wan2.1-1.3B

---

## Overview

Adapted from the P-Flow framework for **video reproduction** task. Given a reference video, iteratively optimizes a T2V prompt so the generated video faithfully reproduces the reference.

- **Video Model:** Wan 2.1-1.3B (T2V, 480P only, 832×480, 81 frames)
- **VLM:** DashScope Qwen-VL (qwen-vl-max)
- **Prompt Strategy:** V2 — T2V model-aware, 80-120 word budget, top-1 difference per iteration
- **Evaluation:** FID-VID, FVD, Dynamic Degree

## Directory Structure

```
P-Flow/
├── src/                    # Core source code
│   ├── pipeline.py         # Algorithm 1 main orchestration
│   ├── noise_prior.py      # Flow Inversion + SVD + Blending
│   ├── svd_filter.py       # Two-stage SVD filtering
│   ├── flow_matching.py    # Euler ODE inversion (t=1→0)
│   ├── vlm_client.py       # V1 VLM client (DashScope Qwen-VL)
│   ├── vlm_client_v2.py    # V2 VLM client (T2V-aware, token budget)
│   ├── prompt_optimizer.py # Vertical composite + VLM refinement
│   ├── video_utils.py      # Video I/O and processing
│   ├── trajectory.py       # History management
│   └── distributed.py      # Single-GPU inference (4090)
├── configs/
│   ├── paper_default.yaml  # Default settings (1.3B + 4090)
│   └── ablation.yaml       # Ablation experiment configs
├── evaluation/
│   ├── metrics.py          # FID-VID, FVD, Dynamic Degree
│   ├── eval_reproduction.py # Per-iteration quality metrics
│   ├── human_eval.py       # Human evaluation tools
│   └── run_evaluation.py   # Complete evaluation pipeline
├── scripts/
│   ├── check_env.py        # Environment verification
│   ├── prepare_dataset.py  # Dataset preparation
│   ├── run_experiment.py   # Experiment runner (single/batch)
│   └── eval_existing.sh    # Evaluate existing results
├── data/
│   └── MovieGenVideoBench.txt  # 1003 prompt benchmark
├── run.py                  # Quick-start entry point
├── run_ab_test.py          # V1 vs V2 prompt strategy comparison
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Environment Setup

```bash
# On AutoDL 4090 machine
cd /root/autodl-tmp/videofake/P-Flow

# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/check_env.py
```

### 2. Model

Model is already at: `/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers`

### 3. Configure API Keys

```bash
export DASHSCOPE_API_KEY="your-dashscope-key"
```

### 4. Run

```bash
# Basic: provide a reference video + initial prompt
python run.py --video reference.mp4 --prompt "a cat jumping on a table"

# Auto-generate initial prompt from video (recommended)
python run.py --video reference.mp4 --auto_prompt

# Adjust motion guidance strength
python run.py --video reference.mp4 --auto_prompt --alpha 0.2

# Quick test with mock VLM (no API key needed)
python run.py --video reference.mp4 --prompt "..." --mock_vlm
```

### 5. Run Evaluation

```bash
python evaluation/run_evaluation.py \
    --experiment_dir /root/autodl-tmp/outputs/video_reproduction \
    --reference_video /path/to/reference.mp4
```

## Model Specs (Wan2.1-1.3B)

| Parameter | Value |
|-----------|-------|
| Resolution | 480P only (832×480) |
| Frames | 81 (5s @ 16fps) |
| Frame constraint | 4n+1 (17, 33, 49, 65, 81...) |
| T5 encoder | UMT5-XXL, text_len=512 |
| Effective prompt window | ~100-150 English words |
| Model size | ~2.6GB (bfloat16) |
| Peak VRAM | ~12-16GB |
| Generation time | ~30-50s per video |

## Pipeline Settings

| Parameter | Value | Source |
|-----------|-------|--------|
| α (noise blend) | 0.1 | Reproduction default |
| ρ_s (spatial SVD) | 0.1 | Eq. 4 |
| ρ_m (temporal SVD) | 0.9 | Eq. 6 |
| i_max (iterations) | 10 | Algorithm 1 |
| Guidance Scale | 5.0 | Section 3.6 |
| Inference Steps | 50 | Section 3.6 |
| VLM prompt budget | 80-120 words | V2 strategy |

## Performance (Single 4090)

| Stage | Time |
|-------|------|
| Video generation | ~30-50s/video |
| VLM inference (Qwen-VL) | ~5-10s |
| Total per iteration | ~40-60s |
| Total per sample (10 iter) | ~7-10 min |
| Peak VRAM | ~12-16GB |
