#!/usr/bin/env python3
"""
Download all evaluation models to a local directory.

After running this script, all evaluation metrics can run with local_files_only=True
(no internet required on the server).

Usage:
    # Download to default models directory
    python evaluation/download_eval_models.py

    # Download to custom directory
    python evaluation/download_eval_models.py --model-dir /root/autodl-tmp/models

    # Download only specific models
    python evaluation/download_eval_models.py --only dinov2 raft lpips

Models downloaded:
    - dinov2-vitb14  (~330MB) — for DINO-Score
    - raft-large     (~100MB) — for Flow EPE / Dynamic Degree
    - lpips-vgg      (~300MB) — for LPIPS
    - r3d_18         (~30MB)  — for FVD (ResNet3D-18 fallback)
"""

import argparse
from pathlib import Path


def download_dinov2(model_dir: Path):
    """Download DINOv2 ViT-B/14 from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel

    save_dir = model_dir / "dinov2-vitb14"
    if save_dir.exists() and any(save_dir.glob("*.safetensors")):
        print(f"  DINOv2 already exists at: {save_dir}", flush=True)
        return

    print(f"  Downloading DINOv2 ViT-B/14...", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-vitb14")
    processor.save_pretrained(save_dir)

    model = AutoModel.from_pretrained("facebook/dinov2-vitb14")
    model.save_pretrained(save_dir)

    print(f"  Saved to: {save_dir}", flush=True)


def download_raft(model_dir: Path):
    """Download RAFT Large and save as standalone weights."""
    import torch

    save_dir = model_dir / "raft-large"
    weights_path = save_dir / "raft_large.pth"
    if weights_path.exists():
        print(f"  RAFT already exists at: {weights_path}", flush=True)
        return

    print(f"  Downloading RAFT Large...", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    raft = torch.hub.load("intel-isc/raft", "raft_large", pretrained=True, verbose=False)
    torch.save(raft.state_dict(), weights_path)
    print(f"  Saved to: {weights_path}", flush=True)


def download_lpips(model_dir: Path):
    """Download LPIPS VGG weights."""
    import torch

    save_dir = model_dir / "lpips-vgg"
    weights_path = save_dir / "vgg_lpips.pth"
    if weights_path.exists():
        print(f"  LPIPS VGG already exists at: {weights_path}", flush=True)
        return

    print(f"  Downloading LPIPS VGG weights...", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Trigger the lpips package's own download to its cache
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False)

    # lpips stores weights in its package directory
    # Find where lpips cached the VGG weights
    lpips_cache = Path(lpips.__file__).parent / "weights" / "vgg_lpips.pth"
    if lpips_cache.exists():
        import shutil
        shutil.copy2(lpips_cache, weights_path)
        print(f"  Copied from lpips cache: {lpips_cache}", flush=True)
    else:
        # Fallback: save the model state directly
        torch.save(net.state_dict(), weights_path)
        print(f"  Saved model state to: {weights_path}", flush=True)


def download_r3d18(model_dir: Path):
    """Download ResNet3D-18 (Kinetics-400) for FVD fallback."""
    import torch

    save_dir = model_dir / "r3d18-kinetics400"
    weights_path = save_dir / "r3d18_kinetics400.pth"
    if weights_path.exists():
        print(f"  R3D18 already exists at: {weights_path}", flush=True)
        return

    print(f"  Downloading R3D18 (Kinetics-400)...", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    from torchvision.models.video import r3d_18, R3D_18_Weights
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    torch.save(model.state_dict(), weights_path)
    print(f"  Saved to: {weights_path}", flush=True)


DOWNLOADERS = {
    "dinov2": download_dinov2,
    "raft": download_raft,
    "lpips": download_lpips,
    "r3d18": download_r3d18,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation models to a local directory"
    )
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/root/autodl-tmp/models"),
                        help="Directory to save models (default: /root/autodl-tmp/models)")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=list(DOWNLOADERS.keys()),
                        help="Only download specific models (default: all)")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {args.model_dir}\n", flush=True)

    models_to_download = args.only if args.only else list(DOWNLOADERS.keys())

    for name in models_to_download:
        print(f"[{name}]", flush=True)
        try:
            DOWNLOADERS[name](args.model_dir)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
        print("", flush=True)

    # Summary
    print("=" * 50, flush=True)
    print("Download complete! Directory contents:", flush=True)
    for d in sorted(args.model_dir.iterdir()):
        if d.is_dir():
            files = list(d.glob("*"))
            print(f"  {d.name}/  ({len(files)} files)", flush=True)

    print(f"\nUsage in evaluation scripts:", flush=True)
    for name in models_to_download:
        if name == "dinov2":
            print(f"  --dinov2-model {args.model_dir / 'dinov2-vitb14'}", flush=True)
        elif name == "raft":
            print(f"  --raft-weights {args.model_dir / 'raft-large' / 'raft_large.pth'}", flush=True)
        elif name == "lpips":
            print(f"  --lpips-weights {args.model_dir / 'lpips-vgg' / 'vgg_lpips.pth'}", flush=True)


if __name__ == "__main__":
    main()
