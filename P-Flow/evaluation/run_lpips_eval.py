#!/usr/bin/env python3
"""
LPIPS (Learned Perceptual Image Patch Similarity) Evaluation Script.

Computes per-frame LPIPS between original and generated videos:
    - lpips_avg: Mean LPIPS across all frames (lower is better)
    - lpips_max: Maximum LPIPS (worst frame)
    - lpips_min: Minimum LPIPS (best frame)

LPIPS uses a VGG backbone trained on human perceptual judgments, capturing
local texture distortions that CLIP misses.

Usage:
    python evaluation/run_lpips_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_lpips
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.clip_utils import extract_numeric_id, format_float, mean_of


# ============================================================
# Default paths
# ============================================================
DEFAULT_ORIG_DIR = Path("data/videos")
DEFAULT_GEN_DIR = Path("outputs/baseline_batch")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_results/lpips")


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
# Video frame loading
# ============================================================

def load_aligned_frames(
    video_path: Path,
    num_frames: int = 8,
    resize_height: int = 0,
    resize_width: int = 0,
) -> list[np.ndarray]:
    """Load video frames as numpy arrays (H, W, 3) in RGB float32 [0, 1].

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to uniformly sample.
        resize_height: Resize height (0 = keep original).
        resize_width: Resize width (0 = keep original).

    Returns:
        List of (H, W, 3) float32 numpy arrays in [0, 1].
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
            if resize_height > 0 and resize_width > 0:
                frame = cv2.resize(frame, (resize_width, resize_height))
            frames.append(frame.astype(np.float32) / 255.0)
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
# LPIPS computation
# ============================================================

def load_lpips_net(device: str):
    """Load LPIPS network (VGG backbone)."""
    import lpips
    net = lpips.LPIPS(net="vgg").to(device)
    net.eval()
    return net


@torch.inference_mode()
def compute_lpips_video(
    orig_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
    lpips_net,
    device: str,
    batch_size: int = 8,
) -> tuple[float, float, float]:
    """Compute LPIPS between aligned original and generated frames.

    Args:
        orig_frames: List of (H, W, 3) float32 arrays in [0, 1].
        gen_frames: List of (H, W, 3) float32 arrays in [0, 1].
        lpips_net: LPIPS network.
        device: Device string.
        batch_size: Batch size for inference.

    Returns:
        Tuple of (lpips_avg, lpips_max, lpips_min).
    """
    n = min(len(orig_frames), len(gen_frames))
    if n == 0:
        return 0.0, 0.0, 0.0

    scores = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        orig_batch = np.stack(orig_frames[start:end], axis=0)  # (B, H, W, 3)
        gen_batch = np.stack(gen_frames[start:end], axis=0)

        # LPIPS expects (B, 3, H, W) in [-1, 1]
        orig_tensor = torch.from_numpy(orig_batch).permute(0, 3, 1, 2).to(device)
        gen_tensor = torch.from_numpy(gen_batch).permute(0, 3, 1, 2).to(device)

        # Scale from [0, 1] to [-1, 1]
        orig_tensor = orig_tensor * 2.0 - 1.0
        gen_tensor = gen_tensor * 2.0 - 1.0

        score = lpips_net(orig_tensor, gen_tensor)
        scores.extend(score.flatten().cpu().tolist())

    scores = np.array(scores)
    return float(np.mean(scores)), float(np.max(scores)), float(np.min(scores))


# ============================================================
# Argparse
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LPIPS evaluation: perceptual similarity between original and generated videos"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--sample-frames", type=int, default=8,
                        help="Number of frames to uniformly sample per video")
    parser.add_argument("--resize-width", type=int, default=0,
                        help="Resize width before LPIPS (0 = keep original)")
    parser.add_argument("--resize-height", type=int, default=0,
                        help="Resize height before LPIPS (0 = keep original)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for LPIPS inference")
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
        "# LPIPS Evaluation Results",
        "",
        "## Metrics Description",
        "",
        "- `lpips_avg`: Mean LPIPS across all frames. Perceptual distance between original and generated video. **Lower is better**.",
        "- `lpips_max`: Maximum per-frame LPIPS (worst frame perceptual distortion). **Lower is better**.",
        "- `lpips_min`: Minimum per-frame LPIPS (best frame perceptual similarity). **Lower is better**.",
        "",
        "LPIPS uses VGG backbone trained on human perceptual judgments. It captures local texture "
        "and structural distortions that CLIP misses.",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- lpips_avg mean: {format_float(summary['lpips_avg_mean'])}",
        f"- lpips_max mean: {format_float(summary['lpips_max_mean'])}",
        f"- lpips_min mean: {format_float(summary['lpips_min_mean'])}",
        "",
        "## Per-Sample Results",
        "",
        "| ID | lpips_avg ↓ | lpips_max ↓ | lpips_min ↓ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | "
            f"{format_float(row['lpips_avg'])} | "
            f"{format_float(row['lpips_max'])} | "
            f"{format_float(row['lpips_min'])} |"
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

    # Load LPIPS
    print(f"Loading LPIPS (VGG) on {args.device}...", flush=True)
    lpips_net = load_lpips_net(args.device)

    # Evaluate each sample
    rows = []
    for idx, item in enumerate(items, 1):
        sample_id = item["sample_id"]

        try:
            orig_frames = load_aligned_frames(
                item["orig_path"], args.sample_frames,
                args.resize_height, args.resize_width,
            )
            gen_frames = load_aligned_frames(
                item["gen_path"], args.sample_frames,
                args.resize_height, args.resize_width,
            )
        except RuntimeError as e:
            print(f"[{idx}/{len(items)}] {sample_id} SKIP: {e}", flush=True)
            continue

        lpips_avg, lpips_max, lpips_min = compute_lpips_video(
            orig_frames, gen_frames, lpips_net, args.device, args.batch_size,
        )

        row = {
            "sample_id": sample_id,
            "lpips_avg": lpips_avg,
            "lpips_max": lpips_max,
            "lpips_min": lpips_min,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(items)}] {sample_id} "
            f"lpips_avg={lpips_avg:.4f} max={lpips_max:.4f} min={lpips_min:.4f}",
            flush=True,
        )

    # Sort by sample ID
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Save CSV
    csv_path = args.output_dir / "lpips_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "lpips_avg", "lpips_max", "lpips_min"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Compute summary
    summary = {
        "count": len(rows),
        "lpips_avg_mean": mean_of(rows, "lpips_avg"),
        "lpips_max_mean": mean_of(rows, "lpips_max"),
        "lpips_min_mean": mean_of(rows, "lpips_min"),
    }

    # Save JSON
    json_path = args.output_dir / "lpips_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save Markdown
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, rows, summary)

    print(f"\nLPIPS evaluation complete!", flush=True)
    print(f"  lpips_avg mean: {format_float(summary['lpips_avg_mean'])}", flush=True)
    print(f"  CSV: {csv_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
