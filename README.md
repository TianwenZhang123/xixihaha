# P-Flow: Prompting Visual Effects Generation

Unofficial reproduction of the paper "P-Flow: A Training-Free Framework for Customizing Dynamic Visual Effects via Test-Time Prompt Optimization" (arXiv:2603.22091).

Includes VISTA (arXiv:2510.15831) multi-agent optimizer as an ablation module.

## Features

- **Dual video backend**: Wan 2.1-T2V-1.3B (local GPU) + Wan 2.7-T2V (DashScope API)
- **VLM prompt optimization**: Gemini 2.0 Flash via LinkAPI relay
- **Noise Prior Enhancement**: Flow matching inversion → SVD filtering → noise blending
- **VISTA ablation**: SVPP + Binary Tournament + MMAC + DTPA

## Project Structure

```
videofake/
├── config/
│   └── default.yaml              # Default hyperparameters
├── pflow/
│   ├── __init__.py
│   ├── pipeline.py               # Main P-Flow pipeline (local model)
│   ├── noise_prior.py            # Noise Prior Enhancement module
│   ├── prompt_optimizer.py       # Test-Time Prompt Optimization
│   ├── vista_optimizer.py        # VISTA multi-agent optimizer (ablation)
│   ├── trajectory.py             # Historical Trajectory Maintenance
│   ├── flow_matching.py          # Flow Matching inversion utilities
│   ├── svd_filter.py             # SVD spatial/temporal filtering
│   ├── video_utils.py            # Video I/O and processing
│   ├── vlm_client.py             # VLM client (gemini-2.0-flash via LinkAPI)
│   └── wan_api_client.py         # Wan 2.7 API client (DashScope)
├── scripts/
│   ├── run_pflow.py              # Path A: Local Wan 2.1 inference
│   ├── run_pflow_api.py          # Path B: Wan 2.7 API inference
│   ├── run_ablation.py           # VISTA ablation experiments
│   └── evaluate.py               # Evaluation metrics (FID-VID, FVD)
├── docs/
│   ├── autodl_4090_guide.md      # AutoDL deployment guide
│   ├── experiment_guide.md       # Complete experiment guide
│   └── experiment_logic.md       # Experiment design logic
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### Path A: Local Wan 2.1 (requires GPU)

```bash
python scripts/run_pflow.py \
    --reference_video path/to/reference.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --model /path/to/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --output_dir outputs/local_run \
    --seed 42
```

### Path B: Wan 2.7 API (no GPU needed)

```bash
export DASHSCOPE_API_KEY="your-dashscope-key"
export OPENAI_API_KEY="your-linkapi-key"

python scripts/run_pflow_api.py \
    --reference_video path/to/reference.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir outputs/api_run \
    --max_iterations 5 \
    --seed 42
```

### Mock Mode (testing without GPU or API)

```bash
python scripts/run_pflow.py \
    --reference_video path/to/reference.mp4 \
    --prompt "test" --mock --output_dir outputs/mock
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.001 | Noise blending weight |
| ρ_s (rho_s) | 0.1 | Spatial SVD retention ratio |
| ρ_m (rho_m) | 0.9 | Temporal SVD retention ratio |
| i_max | 10 | Maximum optimization iterations |
| Resolution | 480×832 | Video resolution (local) / 720P-1080P (API) |
| Frames | 81 | Number of frames (local) |

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- diffusers (for local Wan 2.1 model)
- dashscope (for Wan 2.7 API)
- openai (for VLM via LinkAPI relay)
- eva-decord (for video decoding on Python 3.12)

## Deployment

See [docs/autodl_4090_guide.md](docs/autodl_4090_guide.md) for complete AutoDL 4090 deployment instructions.

See [docs/experiment_guide.md](docs/experiment_guide.md) for the full experiment plan.

## Citation

```bibtex
@article{hu2025pflow,
  title={P-Flow: A Training-Free Framework for Customizing Dynamic Visual Effects via Test-Time Prompt Optimization},
  author={Hu, Junhao and others},
  journal={arXiv preprint arXiv:2603.22091},
  year={2025}
}
```
