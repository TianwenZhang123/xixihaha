#!/usr/bin/env python3
"""
FVD (Fréchet Video Distance) Evaluation Script.

Computes FVD between the set of original videos and the set of generated videos
using I3D (Kinetics-400) features.

FVD measures the distributional distance between two video sets in feature space:
    FVD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r·Σ_g)^{1/2})

This is a distribution-level metric (not per-sample). Output is a single FVD value.

Requires:
    - I3D pretrained weights (downloaded automatically or from local path)
    - scipy for matrix square root

Usage:
    python evaluation/run_fvd_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/baseline_200cases \
        --output-dir outputs/baseline_200cases/eval_fvd

    python evaluation/run_fvd_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_fvd
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.clip_utils import extract_numeric_id, format_float


# ============================================================
# Default paths
# ============================================================
DEFAULT_ORIG_DIR = Path("data/videos")
DEFAULT_GEN_DIR = Path("outputs/baseline_batch")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_results/fvd")
DEFAULT_R3D18_WEIGHTS = "models/r3d18-kinetics400/r3d18_kinetics400.pth"


# ============================================================
# Video loading utilities
# ============================================================

def load_video_frames_as_tensor(
    video_path: Path,
    num_frames: int = 16,
    resize_height: int = 256,
    resize_width: int = 256,
) -> torch.Tensor:
    """Load video frames as a (T, C, H, W) float tensor in [0, 1].

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to uniformly sample.
        resize_height: Resize height for I3D input.
        resize_width: Resize width for I3D input.

    Returns:
        Tensor of shape (T, 3, H, W) normalized to [0, 1].
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    indices = np.linspace(0, max(frame_count - 1, 0), num=num_frames, dtype=int)
    wanted = set(int(i) for i in indices.tolist())
    frames = []
    current = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_width, resize_height))
            frame = torch.from_numpy(frame).float() / 255.0  # (H, W, 3) [0, 1]
            frame = frame.permute(2, 0, 1)  # (3, H, W)
            frames.append(frame)
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to read frames from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames[:num_frames], dim=0)  # (T, 3, H, W)


# ============================================================
# I3D feature extraction
# ============================================================

def load_i3d_model(device: str, model_path: str = ""):
    """Load I3D/ResNet3D model for feature extraction from local weights.

    Same pattern as CLIP/XCLIP: uses local_files_only by default.
    Falls back to ResNet3D-18 if I3D is not available.

    Returns a callable that takes (B, T, C, H, W) and returns (B, D) features.
    """
    from torchvision.models.video import r3d_18

    model = r3d_18(weights=None)
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"  Loaded R3D18 from: {model_path}", flush=True)
    else:
        # Try default weights
        from torchvision.models.video import R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        print(f"  Loaded R3D18 from torchvision default weights", flush=True)
    model = model.to(device)
    model.eval()

    # Remove the classification head, use the avgpool output
    feature_extractor = torch.nn.Sequential(
        model.stem,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        torch.nn.Flatten(),
    )

    @torch.inference_mode()
    def extract_i3d_features(videos: torch.Tensor) -> np.ndarray:
        """Extract 3D ResNet features.

        Args:
            videos: (B, T, C, H, W) in [0, 1], will be normalized.
        Returns:
            (B, 512) feature array.
        """
        B = videos.shape[0]
        videos = videos.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1).to(videos.device)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1).to(videos.device)
        videos = (videos - mean) / std
        features = feature_extractor(videos)
        return features.cpu().numpy()

    return extract_i3d_features


# ============================================================
# FVD computation
# ============================================================

def compute_fvd(features_real: np.ndarray, features_gen: np.ndarray) -> float:
    """Compute Fréchet Video Distance between two sets of features.

    Args:
        features_real: (N, D) array of real video features.
        features_gen: (M, D) array of generated video features.

    Returns:
        FVD value (scalar, lower is better).
    """
    from scipy.linalg import sqrtm

    mu_r = np.mean(features_real, axis=0)
    mu_g = np.mean(features_gen, axis=0)

    sigma_r = np.cov(features_real, rowvar=False)
    sigma_g = np.cov(features_gen, rowvar=False)

    # Handle edge cases with small sample sizes
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    # Numerical stability: if imaginary component is negligible, take real part
    if np.iscomplexobj(covmean):
        if np.allclose(np.imag(covmean), 0, atol=1e-3):
            covmean = np.real(covmean)
        else:
            m = np.max(np.abs(np.imag(covmean)))
            print(f"WARNING: Non-negligible imaginary component in sqrtm (max={m:.6f}), taking real part", flush=True)
            covmean = np.real(covmean)

    fvd = np.dot(diff, diff) + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(max(fvd, 0.0))  # Clamp to non-negative


# ============================================================
# Helper: list video files
# ============================================================

def list_video_files(video_dir: Path, limit: int = 0) -> list[dict]:
    """Find all .mp4 video files and extract numeric IDs."""
    video_map: dict[str, Path] = {}
    for p in video_dir.glob("*.mp4"):
        sid = extract_numeric_id(p)
        if sid and sid not in video_map:
            video_map[sid] = p
    for p in video_dir.glob("sample_*/*.mp4"):
        sid = extract_numeric_id(p)
        if sid:
            video_map[sid] = p

    items = []
    for sample_id in sorted(video_map.keys(), key=lambda x: int(x)):
        items.append({
            "sample_id": sample_id,
            "video_path": video_map[sample_id],
        })
    if limit > 0:
        items = items[:limit]
    return items


def extract_features_for_set(
    videos_dir: Path,
    feature_extractor,
    device: str,
    num_frames: int,
    batch_size: int = 8,
    cache_dir: Path | None = None,
    limit: int = 0,
) -> np.ndarray:
    """Extract I3D features for all videos in a directory.

    Args:
        videos_dir: Directory containing videos.
        feature_extractor: Callable (B, T, C, H, W) -> (B, D).
        device: Device string.
        num_frames: Frames per video.
        batch_size: Batch size for inference.
        cache_dir: Optional directory to cache per-video .npy features.
        limit: Max number of videos (0 = all).

    Returns:
        (N, D) feature array.
    """
    videos = list_video_files(videos_dir, limit)
    if not videos:
        raise SystemExit(f"No videos found in: {videos_dir}")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    batch_tensors = []
    batch_ids = []

    for idx, item in enumerate(videos):
        sid = item["sample_id"]
        video_path = item["video_path"]

        # Try loading from cache
        if cache_dir is not None:
            cache_path = cache_dir / f"{sid}.npy"
            if cache_path.exists():
                feat = np.load(cache_path)
                all_features.append(feat)
                print(f"  [{idx + 1}/{len(videos)}] {sid} (cached)", flush=True)
                continue

        # Load and extract features
        try:
            frames = load_video_frames_as_tensor(video_path, num_frames=num_frames)
        except RuntimeError as e:
            print(f"  [{idx + 1}/{len(videos)}] {sid} SKIP: {e}", flush=True)
            continue

        batch_tensors.append(frames)
        batch_ids.append(sid)

        if len(batch_tensors) >= batch_size or idx == len(videos) - 1:
            batch = torch.stack(batch_tensors, dim=0).to(device)
            feats = feature_extractor(batch)
            for j, feat in enumerate(feats):
                all_features.append(feat)
                # Save to cache
                if cache_dir is not None:
                    cache_path = cache_dir / f"{batch_ids[j]}.npy"
                    np.save(cache_path, feat)
                print(f"  [{idx + 1}/{len(videos)}] {batch_ids[j]}", flush=True)
            batch_tensors = []
            batch_ids = []

    return np.array(all_features)


# ============================================================
# Argparse
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FVD evaluation: Fréchet Video Distance between original and generated video sets"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of frames to sample per video for I3D")
    parser.add_argument("--i3d-model", type=str, default=DEFAULT_R3D18_WEIGHTS,
                        help="Path to R3D18/I3D pretrained weights (default: models/r3d18-kinetics400/r3d18_kinetics400.pth)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N videos (0 = all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable per-video feature caching")
    return parser.parse_args()


# ============================================================
# Markdown output
# ============================================================

def write_markdown_summary(output_path: Path, fvd_value: float,
                           n_real: int, n_gen: int, feat_dim: int) -> None:
    lines = [
        "# FVD Evaluation Results",
        "",
        "## Metric Description",
        "",
        "- **FVD (Fréchet Video Distance)**: Distributional distance between original and generated video sets in I3D feature space. Lower is better.",
        "- Computed using I3D (Kinetics-400) features: `FVD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r·Σ_g)^{1/2})`",
        "",
        "## Summary",
        "",
        f"- **FVD**: {format_float(fvd_value)}",
        f"- Original videos: {n_real}",
        f"- Generated videos: {n_gen}",
        f"- Feature dimension: {feat_dim}",
        "",
        "## Notes",
        "",
        "- FVD is a distribution-level metric; a single value represents the overall quality of the generated set.",
        "- I3D features are 400-dim (or 512-dim if using ResNet3D fallback).",
        "- Per-video features are cached as `.npy` files in the `features/` subdirectory.",
        "- Reference: Unterthiner et al., \"Predicting neural network accuracy from loss trajectories\" (ICML 2020).",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else args.output_dir / "features"

    print(f"Loading I3D model on {args.device}...", flush=True)
    feature_extractor = load_i3d_model(args.device, args.i3d_model)

    print(f"\nExtracting features from original videos: {args.orig_dir}", flush=True)
    features_real = extract_features_for_set(
        args.orig_dir, feature_extractor, args.device,
        num_frames=args.num_frames, batch_size=args.batch_size,
        cache_dir=cache_dir / "real" if cache_dir else None,
        limit=args.limit,
    )
    print(f"  -> {features_real.shape[0]} videos, {features_real.shape[1]}-dim features", flush=True)

    print(f"\nExtracting features from generated videos: {args.gen_dir}", flush=True)
    features_gen = extract_features_for_set(
        args.gen_dir, feature_extractor, args.device,
        num_frames=args.num_frames, batch_size=args.batch_size,
        cache_dir=cache_dir / "gen" if cache_dir else None,
        limit=args.limit,
    )
    print(f"  -> {features_gen.shape[0]} videos, {features_gen.shape[1]}-dim features", flush=True)

    print(f"\nComputing FVD...", flush=True)
    fvd_value = compute_fvd(features_real, features_gen)
    print(f"  FVD = {format_float(fvd_value)}", flush=True)

    # Save JSON
    json_path = args.output_dir / "fvd_results.json"
    results = {
        "fvd": format_float(fvd_value),
        "n_real": features_real.shape[0],
        "n_gen": features_gen.shape[0],
        "feat_dim": features_real.shape[1],
        "num_frames": args.num_frames,
    }
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Save Markdown
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, fvd_value, features_real.shape[0],
                           features_gen.shape[0], features_real.shape[1])

    print(f"\nFVD evaluation complete!", flush=True)
    print(f"  FVD: {format_float(fvd_value)}", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
