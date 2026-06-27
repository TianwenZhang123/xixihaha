#!/usr/bin/env python3
"""
STREAM Evaluation Script for Video Reproduction.

Computes STREAM-T / STREAM-F / STREAM-D metrics between original and generated video sets.

STREAM measures distributional similarity between video sets:
    - STREAM-T: Temporal frequency domain statistical closeness (lower = more similar)
    - STREAM-F: Precision-type coverage of generated over original (higher = better)
    - STREAM-D: Recall-type coverage of generated over original (higher = better)

Requires: pip install v-stream

Usage:
    # Evaluate baseline outputs
    python evaluation/run_stream_eval.py \
        --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --gen-dir /root/autodl-tmp/outputs/baseline_batch \
        --output-dir /root/autodl-tmp/outputs/baseline_batch/eval_stream

    # Evaluate P-Flow outputs
    python evaluation/run_stream_eval.py \
        --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --gen-dir /root/autodl-tmp/outputs/pflow_batch \
        --output-dir /root/autodl-tmp/outputs/pflow_batch/eval_stream

    # Only prepare npy files (skip STREAM computation)
    python evaluation/run_stream_eval.py --prepare-only

    # Custom resolution and frame count
    python evaluation/run_stream_eval.py --resize-width 448 --resize-height 256 --num-frames 16
"""

import argparse
import csv
import json
from pathlib import Path

from evaluation.clip_utils import extract_numeric_id, format_float

import cv2
import numpy as np
import torch


# ============================================================
# Default paths (AutoDL server)
# ============================================================
DEFAULT_ORIG_DIR = Path("/root/autodl-tmp/data/video-200/water_mark_out")
DEFAULT_GEN_DIR = Path("/root/autodl-tmp/outputs/baseline_batch")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/outputs/eval_results/stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STREAM evaluation: compute STREAM-T/F/D between original and generated video sets"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of frames to uniformly sample per video (must match STREAM config)")
    parser.add_argument("--model", type=str, default="dinov2", choices=["swav", "dinov2"],
                        help="STREAM backbone model (dinov2 recommended)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N aligned samples (0 = all)")
    parser.add_argument("--resize-width", type=int, default=448,
                        help="Resize width before saving to npy (0 = keep original)")
    parser.add_argument("--resize-height", type=int, default=256,
                        help="Resize height before saving to npy (0 = keep original)")
    parser.add_argument("--overwrite-prepared", action="store_true",
                        help="Overwrite existing npy cache files")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only do mp4 -> npy preprocessing, skip STREAM scoring")
    return parser.parse_args()


# ============================================================
# Utility functions
# ============================================================

def list_eval_items(orig_dir: Path, gen_dir: Path, limit: int = 0) -> list[dict]:
    """Find aligned pairs: original video + generated video."""
    orig_map = {p.stem: p for p in orig_dir.glob("*.mp4")}
    items = []
    for gen_path in sorted(gen_dir.glob("*.mp4"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        sample_id = extract_numeric_id(gen_path)
        if not sample_id:
            continue
        orig_path = orig_map.get(sample_id)
        if not orig_path:
            continue
        items.append({
            "sample_id": sample_id,
            "orig_path": orig_path,
            "gen_path": gen_path,
        })
    if limit > 0:
        items = items[:limit]
    return items


def sample_video_frames(
    video_path: Path,
    num_frames: int,
    resize_width: int = 0,
    resize_height: int = 0,
) -> np.ndarray:
    """
    Uniformly sample frames from video and return as uint8 array.

    Returns:
        np.ndarray of shape (num_frames, H, W, 3), dtype=uint8
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video frame count: {video_path}")

    indices = np.linspace(0, max(frame_count - 1, 0), num=num_frames, dtype=int)
    wanted = set(int(i) for i in indices.tolist())
    frames: list[np.ndarray] = []
    current = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_width > 0 and resize_height > 0:
                rgb = cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            frames.append(rgb.astype(np.uint8))
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to sample frames from video: {video_path}")

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames[:num_frames], axis=0).astype(np.uint8)


def ensure_stream_installed():
    """Check that the v-stream package is installed."""
    try:
        from stream import STREAM  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Package 'stream' not found. Please install it:\n"
            "    pip install v-stream\n"
            "Then retry."
        ) from exc


# ============================================================
# Preprocessing: mp4 -> npy
# ============================================================

def prepare_npy_datasets(
    items: list[dict],
    output_dir: Path,
    num_frames: int,
    resize_width: int,
    resize_height: int,
    overwrite_prepared: bool,
) -> tuple[Path, Path, Path]:
    """
    Convert video pairs to npy arrays for STREAM evaluation.

    STREAM expects: directory of vid_00000.npy files, each (F, H, W, C) uint8.
    """
    prepared_root = output_dir / "prepared"
    real_dir = prepared_root / "real"
    fake_dir = prepared_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "prepared_metadata.csv"

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "sample_id", "orig_video", "gen_video", "real_npy", "fake_npy"],
        )
        writer.writeheader()

        for index, item in enumerate(items):
            real_name = f"vid_{index:05d}.npy"
            fake_name = f"vid_{index:05d}.npy"
            real_npy_path = real_dir / real_name
            fake_npy_path = fake_dir / fake_name

            if overwrite_prepared or not real_npy_path.exists():
                real_frames = sample_video_frames(
                    item["orig_path"],
                    num_frames=num_frames,
                    resize_width=resize_width,
                    resize_height=resize_height,
                )
                np.save(real_npy_path, real_frames)

            if overwrite_prepared or not fake_npy_path.exists():
                fake_frames = sample_video_frames(
                    item["gen_path"],
                    num_frames=num_frames,
                    resize_width=resize_width,
                    resize_height=resize_height,
                )
                np.save(fake_npy_path, fake_frames)

            writer.writerow({
                "index": index,
                "sample_id": item["sample_id"],
                "orig_video": str(item["orig_path"]),
                "gen_video": str(item["gen_path"]),
                "real_npy": str(real_npy_path),
                "fake_npy": str(fake_npy_path),
            })

    return real_dir, fake_dir, metadata_path


# ============================================================
# Output formatting
# ============================================================

def write_markdown_summary(output_path: Path, summary: dict) -> None:
    resize_text = "original resolution"
    if summary["resize_width"] > 0 and summary["resize_height"] > 0:
        resize_text = f"{summary['resize_width']}x{summary['resize_height']}"

    lines = [
        "# STREAM Evaluation Results",
        "",
        "## Metrics Description",
        "",
        "- `STREAM-T`: Temporal frequency domain statistical closeness (lower = more similar)",
        "- `STREAM-F`: Precision-type coverage of generated over original (higher = better)",
        "- `STREAM-D`: Recall-type coverage of generated over original (higher = better)",
        "",
        "## Configuration",
        "",
        f"- Original video dir: `{summary['orig_dir']}`",
        f"- Generated video dir: `{summary['gen_dir']}`",
        f"- Aligned sample count: {summary['count']}",
        f"- Frames per video: {summary['num_frames']}",
        f"- Export resolution: {resize_text}",
        f"- STREAM backbone: `{summary['model']}`",
        f"- Device: `{summary['device']}`",
        "",
        "## Results",
        "",
        f"- STREAM-T: {format_float(summary['stream_t'])}",
        f"- STREAM-F: {format_float(summary['stream_f'])}",
        f"- STREAM-D: {format_float(summary['stream_d'])}",
        "",
        "## Output Files",
        "",
        f"- Alignment metadata: `{summary['metadata_path']}`",
        f"- JSON results: `{summary['json_path']}`",
        f"- Feature cache: `{summary['feature_dir']}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    ensure_stream_installed()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find aligned samples
    items = list_eval_items(args.orig_dir, args.gen_dir, limit=args.limit)
    if not items:
        raise SystemExit(
            f"No aligned samples found.\n"
            f"  orig-dir: {args.orig_dir}\n"
            f"  gen-dir: {args.gen_dir}"
        )

    print(f"Found {len(items)} aligned samples for STREAM evaluation", flush=True)

    # Prepare npy datasets
    real_dir, fake_dir, metadata_path = prepare_npy_datasets(
        items=items,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        overwrite_prepared=args.overwrite_prepared,
    )
    print(f"npy preprocessing complete: {real_dir} / {fake_dir}", flush=True)

    if args.prepare_only:
        print("--prepare-only enabled, skipping STREAM metric computation.", flush=True)
        return

    # Compute STREAM metrics
    from stream import STREAM

    print(
        f"Computing STREAM metrics: model={args.model}, num_frames={args.num_frames}, device={args.device}",
        flush=True,
    )
    evaluator = STREAM(num_frame=args.num_frames, model=args.model)

    real_skewness, real_mean_signal = evaluator.calculate_skewness(
        str(real_dir), device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    fake_skewness, fake_mean_signal = evaluator.calculate_skewness(
        str(fake_dir), device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
    )

    stream_t = float(evaluator.stream_T(fake_skewness, real_skewness))
    stream_s = evaluator.stream_S(fake_mean_signal, real_mean_signal)
    stream_f = float(stream_s["stream_F"])
    stream_d = float(stream_s["stream_D"])

    # Save feature arrays for reproducibility
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "real_skewness.npy", real_skewness.cpu().numpy())
    np.save(feature_dir / "fake_skewness.npy", fake_skewness.cpu().numpy())
    np.save(feature_dir / "real_mean_signal.npy", real_mean_signal.cpu().numpy())
    np.save(feature_dir / "fake_mean_signal.npy", fake_mean_signal.cpu().numpy())

    # Build summary
    summary = {
        "orig_dir": str(args.orig_dir),
        "gen_dir": str(args.gen_dir),
        "count": len(items),
        "num_frames": args.num_frames,
        "resize_width": args.resize_width,
        "resize_height": args.resize_height,
        "model": args.model,
        "device": args.device,
        "stream_t": stream_t,
        "stream_f": stream_f,
        "stream_d": stream_d,
        "metadata_path": str(metadata_path),
        "feature_dir": str(feature_dir),
    }

    # Save JSON
    json_path = args.output_dir / "stream_results.json"
    summary["json_path"] = str(json_path)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save Markdown
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, summary)

    # Print results
    print(f"\nSTREAM Evaluation Results:", flush=True)
    print(f"  STREAM-T: {stream_t:.6f}", flush=True)
    print(f"  STREAM-F: {stream_f:.6f}", flush=True)
    print(f"  STREAM-D: {stream_d:.6f}", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
