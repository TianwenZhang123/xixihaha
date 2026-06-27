#!/usr/bin/env python3
"""
DINO-Score Evaluation Script for Video Reproduction.

Computes DINOv2-based metrics between original and generated videos:
    1. dino_temporal  : Adjacent-frame cosine similarity WITHIN generated video (temporal consistency)
    2. dino_orig_gen  : Frame-level cosine similarity between original and generated video

DINOv2's self-supervised training makes it highly sensitive to fine-grained visual
details (texture, shape, instance identity), complementing CLIP's semantic focus.

Usage:
    python evaluation/run_dino_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_dino

    # Use local DINOv2 model
    python evaluation/run_dino_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --dinov2-model models/dinov2-vitb14 \
        --output-dir outputs/pflow_200cases/eval_dino
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.clip_utils import (
    extract_numeric_id,
    format_float,
    mean_of,
    sample_video_frames,
)


# ============================================================
# Default paths
# ============================================================
DEFAULT_ORIG_DIR = Path("data/videos")
DEFAULT_GEN_DIR = Path("outputs/baseline_batch")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_results/dino")
DEFAULT_DINOV2_MODEL = "models/dinov2-vitb14"


# ============================================================
# Video pair finder
# ============================================================

def list_eval_items(
    orig_dir: Path, gen_dir: Path, limit: int = 0
) -> list[dict]:
    """Find aligned original + generated video pairs by numeric ID."""
    orig_map = {extract_numeric_id(p): p for p in orig_dir.glob("*.mp4")
                if extract_numeric_id(p)}

    gen_map: dict[str, Path] = {}
    for p in gen_dir.glob("*.mp4"):
        sid = extract_numeric_id(p)
        if sid and sid not in gen_map:
            gen_map[sid] = p
    for p in gen_dir.glob("sample_*/*.mp4"):
        sid = extract_numeric_id(p)
        if sid:
            gen_map[sid] = p

    items = []
    for sample_id in sorted(gen_map.keys(), key=lambda x: int(x)):
        orig_path = orig_map.get(sample_id)
        if orig_path:
            items.append({
                "sample_id": sample_id,
                "orig_path": orig_path,
                "gen_path": gen_map[sample_id],
            })
    if limit > 0:
        items = items[:limit]
    return items


# ============================================================
# DINOv2 feature extraction
# ============================================================

@torch.inference_mode()
def get_dinov2_frame_features(
    frames: list,
    processor,
    model,
    device: str,
) -> np.ndarray:
    """Extract DINOv2 [CLS] token features for a list of frames.

    Args:
        frames: List of PIL Images.
        processor: DINOv2 image processor.
        model: DINOv2 model.
        device: Device string.

    Returns:
        (N_frames, D) feature array from [CLS] token.
    """
    from transformers import AutoImageProcessor

    inputs = processor(images=frames, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Use [CLS] token (first token) from last hidden state
    cls_features = outputs.last_hidden_state[:, 0, :]  # (N, D)
    return cls_features.cpu().numpy()


def load_dinov2_model(model_name: str, device: str):
    """Load DINOv2 model and processor from local path (same pattern as CLIP/XCLIP)."""
    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading DINOv2 model from: {model_name}", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)

    model = model.to(device).eval()
    return processor, model


def compute_dino_temporal(frame_features: np.ndarray) -> float:
    """Compute temporal consistency as mean adjacent-frame cosine similarity.

    Args:
        frame_features: (T, D) DINOv2 features.

    Returns:
        Mean cosine similarity between adjacent frames (higher = more consistent).
    """
    if frame_features.shape[0] < 2:
        return 1.0

    sims = []
    for i in range(frame_features.shape[0] - 1):
        a = frame_features[i].astype(np.float32)
        b = frame_features[i + 1].astype(np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
    return float(np.mean(sims))


def compute_dino_orig_gen(
    orig_features: np.ndarray, gen_features: np.ndarray
) -> float:
    """Compute frame-wise DINO similarity between original and generated video.

    Args:
        orig_features: (T1, D) features from original video.
        gen_features: (T2, D) features from generated video.

    Returns:
        Mean cosine similarity between corresponding frames (higher = more similar).
    """
    # Match by index (both videos have same number of sampled frames)
    n = min(orig_features.shape[0], gen_features.shape[0])
    if n == 0:
        return 0.0

    sims = []
    for i in range(n):
        a = orig_features[i].astype(np.float32)
        b = gen_features[i].astype(np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
    return float(np.mean(sims))


# ============================================================
# Argparse
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINO-Score evaluation: frame-level consistency using DINOv2 features"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--dinov2-model", type=str, default=DEFAULT_DINOV2_MODEL,
                        help="DINOv2 model local path (default: models/dinov2-vitb14, local_files_only)")
    parser.add_argument("--sample-frames", type=int, default=8,
                        help="Number of frames to uniformly sample per video")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N samples (0 = all)")
    return parser.parse_args()


# ============================================================
# Markdown output
# ============================================================

def write_markdown_summary(output_path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# DINO-Score Evaluation Results",
        "",
        "## Metrics Description",
        "",
        "- `dino_temporal`: Mean adjacent-frame cosine similarity **within** generated video using DINOv2 [CLS] features. "
          "Measures temporal consistency of subject identity/appearance. **Higher is better**.",
        "- `dino_orig_gen`: Mean frame-wise cosine similarity between original and generated video using DINOv2 [CLS] features. "
          "Measures fine-grained visual fidelity (texture, shape, instance identity). **Higher is better**.",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- dino_temporal mean: {format_float(summary['dino_temporal_mean'])}",
        f"- dino_orig_gen mean: {format_float(summary['dino_orig_gen_mean'])}",
        "",
        "## Per-Sample Results",
        "",
        "| ID | dino_temporal ↑ | dino_orig_gen ↑ |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | "
            f"{format_float(row['dino_temporal'])} | "
            f"{format_float(row['dino_orig_gen'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find aligned video pairs
    items = list_eval_items(args.orig_dir, args.gen_dir, args.limit)
    if not items:
        raise SystemExit(
            f"No aligned samples found.\n"
            f"  orig-dir: {args.orig_dir}\n"
            f"  gen-dir: {args.gen_dir}"
        )

    print(f"Found {len(items)} aligned samples for evaluation", flush=True)

    # Load DINOv2
    processor, model = load_dinov2_model(args.dinov2_model, args.device)

    # Evaluate each sample
    rows = []
    for idx, item in enumerate(items, 1):
        sample_id = item["sample_id"]
        orig_frames = sample_video_frames(item["orig_path"], args.sample_frames)
        gen_frames = sample_video_frames(item["gen_path"], args.sample_frames)

        # Extract DINOv2 features
        orig_feats = get_dinov2_frame_features(orig_frames, processor, model, args.device)
        gen_feats = get_dinov2_frame_features(gen_frames, processor, model, args.device)

        # Compute metrics
        dino_temporal = compute_dino_temporal(gen_feats)
        dino_orig_gen = compute_dino_orig_gen(orig_feats, gen_feats)

        row = {
            "sample_id": sample_id,
            "dino_temporal": dino_temporal,
            "dino_orig_gen": dino_orig_gen,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(items)}] {sample_id} "
            f"dino_temporal={dino_temporal:.4f} dino_orig_gen={dino_orig_gen:.4f}",
            flush=True,
        )

    # Sort by sample ID
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Save CSV
    csv_path = args.output_dir / "dino_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "dino_temporal", "dino_orig_gen"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Compute summary
    summary = {
        "count": len(rows),
        "dino_temporal_mean": mean_of(rows, "dino_temporal"),
        "dino_orig_gen_mean": mean_of(rows, "dino_orig_gen"),
    }

    # Save JSON
    json_path = args.output_dir / "dino_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save Markdown
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, rows, summary)

    print(f"\nDINO-Score evaluation complete!", flush=True)
    print(f"  dino_temporal mean: {format_float(summary['dino_temporal_mean'])}", flush=True)
    print(f"  dino_orig_gen mean: {format_float(summary['dino_orig_gen_mean'])}", flush=True)
    print(f"  CSV: {csv_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
