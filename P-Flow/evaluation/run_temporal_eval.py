#!/usr/bin/env python3
"""
Temporal Flickering + Dynamic Degree Evaluation Script.

Computes temporal quality metrics for generated videos (no original video needed):
    1. temporal_flicker : Mean adjacent-frame pixel MAE (lower is better)
    2. dynamic_degree   : Mean optical flow magnitude (higher is better)

Temporal Flickering detects high-frequency visual artifacts (texture jumping,
lighting flicker). Dynamic Degree measures how much motion is present.

Both metrics only use the generated video, making them applicable even without
an original reference.

Usage:
    python evaluation/run_temporal_eval.py \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_temporal

    # Also compute flow-based dynamic degree (slower)
    python evaluation/run_temporal_eval.py \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_temporal \
        --use-raft
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluation.clip_utils import extract_numeric_id, format_float, mean_of


# ============================================================
# Default paths
# ============================================================
DEFAULT_GEN_DIR = Path("outputs/baseline_batch")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_results/temporal")


# ============================================================
# Video finder
# ============================================================

def list_gen_videos(gen_dir: Path, limit: int = 0) -> list[dict]:
    """Find all generated video files by numeric ID."""
    video_map: dict[str, Path] = {}
    for p in gen_dir.glob("*.mp4"):
        sid = extract_numeric_id(p)
        if sid and sid not in video_map:
            video_map[sid] = p
    for p in gen_dir.glob("sample_*/*.mp4"):
        sid = extract_numeric_id(p)
        if sid:
            video_map[sid] = p

    items = []
    for sample_id in sorted(video_map.keys(), key=lambda x: int(x)):
        items.append({
            "sample_id": sample_id,
            "gen_path": video_map[sample_id],
        })
    if limit > 0:
        items = items[:limit]
    return items


# ============================================================
# Frame loading
# ============================================================

def load_video_frames_numpy(
    video_path: Path,
    num_frames: int = 16,
    resize_height: int = 0,
    resize_width: int = 0,
) -> list[np.ndarray]:
    """Load video frames as numpy arrays (H, W, 3) uint8 RGB."""
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
            if resize_height > 0 and resize_width > 0:
                frame = cv2.resize(frame, (resize_width, resize_height))
            frames.append(frame)
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to read frames from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


# ============================================================
# Metrics
# ============================================================

def compute_temporal_flicker(frames: list[np.ndarray]) -> float:
    """Compute temporal flickering as mean adjacent-frame pixel MAE.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.

    Returns:
        Mean absolute error between adjacent frames (lower is better).
    """
    if len(frames) < 2:
        return 0.0

    maes = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32))
        maes.append(float(np.mean(diff)))
    return float(np.mean(maes))


def compute_pixel_dynamic_degree(frames: list[np.ndarray]) -> float:
    """Compute dynamic degree from adjacent-frame pixel differences.

    A lightweight alternative to RAFT-based dynamic degree that doesn't
    require the RAFT model.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.

    Returns:
        Mean absolute pixel difference (higher = more dynamic).
    """
    if len(frames) < 2:
        return 0.0

    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32))
        diffs.append(float(np.mean(diff)))
    return float(np.mean(diffs))


def load_raft_model(device: str):
    """Load RAFT optical flow model."""
    raft = torch.hub.load(
        "intel-isc/raft", "raft_large", pretrained=True, verbose=False
    )
    raft = raft.to(device)
    raft.eval()
    return raft


@torch.inference_mode()
def compute_raft_dynamic_degree(
    frames: list[np.ndarray], raft_model, device: str
) -> float:
    """Compute dynamic degree using RAFT optical flow magnitude.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        raft_model: RAFT model.
        device: Device string.

    Returns:
        Mean optical flow magnitude (higher = more dynamic).
    """
    if len(frames) < 2:
        return 0.0

    magnitudes = []
    for i in range(len(frames) - 1):
        img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float() / 255.0
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)

        flow_list, _ = raft_model(img1, img2, iters=20, test_mode=True)
        flow = flow_list[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        mag = np.sqrt(np.sum(flow.astype(np.float32) ** 2, axis=2))
        magnitudes.append(float(np.mean(mag)))

    return float(np.mean(magnitudes)) if magnitudes else 0.0


# ============================================================
# Argparse
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal quality evaluation: flickering and dynamic degree for generated videos"
    )
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--sample-frames", type=int, default=16,
                        help="Number of frames to sample per video")
    parser.add_argument("--resize-width", type=int, default=640,
                        help="Resize width before flow estimation (0 = keep original)")
    parser.add_argument("--resize-height", type=int, default=360,
                        help="Resize height before flow estimation (0 = keep original)")
    parser.add_argument("--use-raft", action="store_true",
                        help="Use RAFT for dynamic degree (slower but more accurate)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N samples (0 = all)")
    return parser.parse_args()


# ============================================================
# Markdown output
# ============================================================

def write_markdown_summary(output_path: Path, rows: list[dict], summary: dict,
                           use_raft: bool) -> None:
    dyn_label = "dynamic_degree (RAFT)" if use_raft else "dynamic_degree (pixel diff)"
    lines = [
        "# Temporal Quality Evaluation Results",
        "",
        "## Metrics Description",
        "",
        "- `temporal_flicker`: Mean adjacent-frame pixel MAE. Detects high-frequency visual artifacts "
          "(texture jumping, lighting flicker). **Lower is better**.",
        f"- `{dyn_label}`: Mean motion magnitude of generated video. Measures how dynamic the output is. "
          "**Higher is better** (proves the method doesn't freeze the video).",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- temporal_flicker mean: {format_float(summary['temporal_flicker_mean'])}",
        f"- dynamic_degree mean: {format_float(summary['dynamic_degree_mean'])}",
        "",
        "## Per-Sample Results",
        "",
        "| ID | temporal_flicker ↓ | dynamic_degree ↑ |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | "
            f"{format_float(row['temporal_flicker'])} | "
            f"{format_float(row['dynamic_degree'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find generated videos
    items = list_gen_videos(args.gen_dir, args.limit)
    if not items:
        raise SystemExit(f"No generated videos found in: {args.gen_dir}")

    print(f"Found {len(items)} generated videos for evaluation", flush=True)

    # Optionally load RAFT
    raft_model = None
    if args.use_raft:
        print(f"Loading RAFT model on {args.device}...", flush=True)
        raft_model = load_raft_model(args.device)

    # Evaluate each sample
    rows = []
    for idx, item in enumerate(items, 1):
        sample_id = item["sample_id"]

        try:
            frames = load_video_frames_numpy(
                item["gen_path"], args.sample_frames,
                args.resize_height, args.resize_width,
            )
        except RuntimeError as e:
            print(f"[{idx}/{len(items)}] {sample_id} SKIP: {e}", flush=True)
            continue

        # Temporal flickering (pixel MAE)
        temporal_flicker = compute_temporal_flicker(frames)

        # Dynamic degree
        if raft_model is not None:
            dynamic_degree = compute_raft_dynamic_degree(frames, raft_model, args.device)
        else:
            dynamic_degree = compute_pixel_dynamic_degree(frames)

        row = {
            "sample_id": sample_id,
            "temporal_flicker": temporal_flicker,
            "dynamic_degree": dynamic_degree,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(items)}] {sample_id} "
            f"flicker={temporal_flicker:.4f} dyn={dynamic_degree:.4f}",
            flush=True,
        )

    # Sort by sample ID
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Save CSV
    csv_path = args.output_dir / "temporal_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "temporal_flicker", "dynamic_degree"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Compute summary
    summary = {
        "count": len(rows),
        "temporal_flicker_mean": mean_of(rows, "temporal_flicker"),
        "dynamic_degree_mean": mean_of(rows, "dynamic_degree"),
    }

    # Save JSON
    json_path = args.output_dir / "temporal_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save Markdown
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, rows, summary, args.use_raft)

    print(f"\nTemporal evaluation complete!", flush=True)
    print(f"  temporal_flicker mean: {format_float(summary['temporal_flicker_mean'])}", flush=True)
    print(f"  dynamic_degree mean: {format_float(summary['dynamic_degree_mean'])}", flush=True)
    print(f"  CSV: {csv_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
