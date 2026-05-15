# P-Flow: Prompting Visual Effects Generation

Unofficial reproduction of the paper "P-Flow: A Training-Free Framework for Customizing Dynamic Visual Effects via Test-Time Prompt Optimization" (arXiv:2603.22091).

## Project Structure

```
videofake/
├── config/
│   └── default.yaml          # Default hyperparameters
├── pflow/
│   ├── __init__.py
│   ├── pipeline.py            # Main P-Flow pipeline
│   ├── noise_prior.py         # Noise Prior Enhancement module
│   ├── prompt_optimizer.py    # Test-Time Prompt Optimization module
│   ├── trajectory.py          # Historical Trajectory Maintenance
│   ├── flow_matching.py       # Flow Matching inversion utilities
│   ├── svd_filter.py          # SVD spatial/temporal filtering
│   ├── video_utils.py         # Video I/O and processing utilities
│   └── vlm_client.py         # VLM (Gemini) API client
├── scripts/
│   ├── run_pflow.py           # Main entry point
│   └── evaluate.py            # Evaluation metrics (FID-VID, FVD)
├── requirements.txt
└── README.md
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.001 | Noise blending weight |
| ρ_s (rho_s) | 0.1 | Spatial SVD retention ratio |
| ρ_m (rho_m) | 0.9 | Temporal SVD retention ratio |
| i_max | 10 | Maximum optimization iterations |
| Resolution | 480×832 | Video resolution |
| Frames | 81 | Number of frames |

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- diffusers (for Wan 2.1 model)
- google-generativeai (for Gemini VLM)
- opencv-python
- numpy, scipy

## Usage

```bash
python scripts/run_pflow.py \
    --reference_video path/to/reference.mp4 \
    --prompt "A cat walking through magical sparkles" \
    --output_dir outputs/ \
    --config config/default.yaml
```

## Citation

```bibtex
@article{hu2025pflow,
  title={P-Flow: A Training-Free Framework for Customizing Dynamic Visual Effects via Test-Time Prompt Optimization},
  author={Hu, Junhao and others},
  journal={arXiv preprint arXiv:2603.22091},
  year={2025}
}
```
# xixihaha
