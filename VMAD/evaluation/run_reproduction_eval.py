#!/usr/bin/env python3
"""
VMAD Reproduction Evaluation — Video Prompt Inversion Quality Assessment.

Evaluates how faithfully VMAD can reproduce original videos via three-layer
progressive encoding. This is the PRIMARY evaluation for VMAD's core claim:
compress video into a reusable prompt asset, then reproduce it.

Core Metrics (higher = better reproduction):
    1. orig_gen_clip   : CLIP cosine(original_video_frames, generated_video_frames)
       - Frame-level visual similarity (appearance, color, composition)
    2. orig_gen_xclip  : X-CLIP cosine(original_video, generated_video)
       - Temporal semantic similarity (motion, dynamics, pacing)

Auxiliary Metrics (for diagnosis):
    3. orig_text_clip  : CLIP cosine(original_video_frames, caption)
    4. gen_text_clip   : CLIP cosine(generated_video_frames, caption)
    5. orig_text_xclip : X-CLIP cosine(original_video, caption)
    6. gen_text_xclip  : X-CLIP cosine(generated_video, caption)

Usage:
    # Evaluate VMAD reproduction (after run_batch_extract.py --content SELF)
    python evaluation/run_reproduction_eval.py \
        --orig-dir /path/to/original_videos \
        --gen-dir ./outputs/vmad_reproduce/generated \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/vmad_reproduce/eval_reproduction

    # Compare against P-Flow baseline
    python evaluation/run_reproduction_eval.py \
        --orig-dir /path/to/original_videos \
        --gen-dir /path/to/pflow_outputs/flat \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/pflow_baseline/eval_reproduction

    # Quick test (first 10 samples)
    python evaluation/run_reproduction_eval.py \
        --orig-dir /path/to/original_videos \
        --gen-dir ./outputs/vmad_reproduce/generated \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/eval_debug \
        --limit 10

    # Ablation comparison (Layer 2 only vs full three-layer)
    python evaluation/run_reproduction_eval.py \
        --orig-dir /path/to/original_videos \
        --gen-dir ./outputs/ablation_layer2_only/generated \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/ablation_layer2_only/eval \
        --method-name "Layer2-only"

Models:
    - CLIP: openai/clip-vit-base-patch32 (frame-level visual similarity)
    - X-CLIP: microsoft/xclip-base-patch32 (temporal-aware video similarity)

Relationship to P-Flow's run_clip_xclip_eval.py:
    This script computes the same 6 metrics as P-Flow's evaluation, ensuring
    direct comparability. The key reproduction metrics are orig_gen_clip and
    orig_gen_xclip — these measure how close the generated video is to the
    original, which is the fundamental goal of Video Prompt Inversion.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, XCLIPModel


# ============================================================
# Default paths (AutoDL server)
# ============================================================
DEFAULT_ORIG_DIR = Path("/root/autodl-tmp/data/video-200/water_mark_out")
DEFAULT_GEN_DIR = Path("/root/autodl-tmp/outputs/vmad_reproduce/generated")
DEFAULT_CAPTION_DIR = Path("/root/autodl-tmp/data/video-200/captions_qwen")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/outputs/vmad_reproduce/eval_reproduction")
DEFAULT_CLIP_MODEL = "/root/autodl-tmp/models/clip-vit-base-patch32"
DEFAULT_XCLIP_MODEL = "/root/autodl-tmp/models/xclip-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VMAD Reproduction Evaluation: orig vs generated (CLIP + X-CLIP)"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--caption-dir", type=Path, default=DEFAULT_CAPTION_DIR,
                        help="Directory containing captions ({id}.txt)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL,
                        help="Path to CLIP model (local or HuggingFace name)")
    parser.add_argument("--xclip-model", type=str, default=DEFAULT_XCLIP_MODEL,
                        help="Path to X-CLIP model (local or HuggingFace name)")
    parser.add_argument("--sample-frames", type=int, default=8,
                        help="Number of frames to uniformly sample per video")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N samples (0 = all)")
    parser.add_argument("--method-name", type=str, default="VMAD",
                        help="Method name for report header")
    parser.add_argument("--baseline-json", type=Path, default=None,
                        help="Path to baseline eval_results.json for comparison")
    return parser.parse_args()


# ============================================================
# Utility functions
# ============================================================

def l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return float(np.dot(a, b))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_numeric_id(path: Path) -> Optional[str]:
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else None


def list_eval_items(orig_dir: Path, gen_dir: Path, caption_dir: Path,
                    limit: int = 0) -> list:
    """Find aligned triplets: original video + generated video + caption."""
    orig_map = {p.stem: p for p in orig_dir.glob("*.mp4")}
    caption_map = {p.stem: p for p in caption_dir.glob("*.txt")}

    items = []
    for gen_path in sorted(gen_dir.glob("*.mp4"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        sample_id = extract_numeric_id(gen_path)
        if not sample_id:
            continue
        orig_path = orig_map.get(sample_id)
        caption_path = caption_map.get(sample_id)
        if orig_path and caption_path:
            items.append({
                "sample_id": sample_id,
                "gen_name": gen_path.name,
                "orig_path": orig_path,
                "gen_path": gen_path,
                "caption_path": caption_path,
            })

    if limit > 0:
        items = items[:limit]
    return items


def sample_video_frames(video_path: Path, num_frames: int) -> list:
    """Uniformly sample frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video frame count: {video_path}")

    indices = np.linspace(0, max(frame_count - 1, 0), num=num_frames, dtype=int)
    frames = []
    wanted = set(int(i) for i in indices.tolist())
    current = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to sample frames from video: {video_path}")

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


# ============================================================
# Model loading
# ============================================================

def build_models(device: str, clip_model_name: str, xclip_model_name: str):
    """Load CLIP and X-CLIP models."""
    print(f"Loading CLIP model from: {clip_model_name}", flush=True)
    try:
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
    except OSError:
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    print(f"Loading X-CLIP model from: {xclip_model_name}", flush=True)
    try:
        xclip_processor = AutoProcessor.from_pretrained(xclip_model_name, local_files_only=True)
        xclip_model = XCLIPModel.from_pretrained(xclip_model_name, local_files_only=True).to(device)
    except OSError:
        xclip_processor = AutoProcessor.from_pretrained(xclip_model_name)
        xclip_model = XCLIPModel.from_pretrained(xclip_model_name).to(device)
    xclip_model.eval()

    return clip_processor, clip_model, xclip_processor, xclip_model


# ============================================================
# Feature extraction
# ============================================================

@torch.inference_mode()
def get_clip_text_feature(text: str, processor, model, device: str) -> np.ndarray:
    """Get CLIP text embedding."""
    inputs = processor(text=[text], return_tensors="pt", truncation=True).to(device)
    features = model.get_text_features(**inputs)
    return features[0].detach().float().cpu().numpy()


@torch.inference_mode()
def get_clip_video_feature(frames: list, processor, model, device: str) -> np.ndarray:
    """Get CLIP image embedding (mean over frames)."""
    inputs = processor(images=frames, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    mean_feature = features.detach().float().cpu().numpy().mean(axis=0)
    return mean_feature


@torch.inference_mode()
def get_xclip_text_feature(text: str, processor, model, device: str) -> np.ndarray:
    """Get X-CLIP text embedding."""
    inputs = processor(text=[text], return_tensors="pt", truncation=True).to(device)
    features = model.get_text_features(**inputs)
    return features[0].detach().float().cpu().numpy()


@torch.inference_mode()
def get_xclip_video_feature(frames: list, processor, model, device: str) -> np.ndarray:
    """Get X-CLIP video embedding (temporal-aware)."""
    np_frames = [np.array(frame) for frame in frames]
    inputs = processor(videos=[np_frames], return_tensors="pt").to(device)
    features = model.get_video_features(**inputs)
    return features[0].detach().float().cpu().numpy()


# ============================================================
# Output formatting
# ============================================================

def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def mean_of(rows: list, key: str) -> float:
    if not rows:
        return float("nan")
    return float(sum(row[key] for row in rows) / len(rows))


def write_markdown_summary(output_path: Path, rows: list, summary: dict,
                           method_name: str, baseline: Optional[dict] = None) -> None:
    """Write detailed markdown report with optional baseline comparison."""
    lines = [
        f"# {method_name} — Video Reproduction Evaluation",
        "",
        "## Core Reproduction Metrics (higher = better)",
        "",
        f"| Metric | Value |{'Baseline | Δ |' if baseline else ''}",
        f"| --- | ---: |{'---: | ---: |' if baseline else ''}",
    ]

    # Core metrics with optional baseline comparison
    for metric, label in [
        ("orig_gen_clip_mean", "orig_gen_clip (frame similarity)"),
        ("orig_gen_xclip_mean", "orig_gen_xclip (temporal similarity)"),
    ]:
        val = format_float(summary[metric])
        if baseline and metric in baseline:
            base_val = format_float(baseline[metric])
            delta = summary[metric] - baseline[metric]
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            lines.append(f"| **{label}** | **{val}** | {base_val} | {delta_str} |")
        else:
            lines.append(f"| **{label}** | **{val}** |")

    lines.extend([
        "",
        "## All Metrics",
        "",
        f"| Metric | Mean |{'Baseline | Δ |' if baseline else ''}",
        f"| --- | ---: |{'---: | ---: |' if baseline else ''}",
    ])

    for metric in ["orig_text_clip_mean", "gen_text_clip_mean", "orig_gen_clip_mean",
                   "orig_text_xclip_mean", "gen_text_xclip_mean", "orig_gen_xclip_mean"]:
        val = format_float(summary[metric])
        label = metric.replace("_mean", "")
        if baseline and metric in baseline:
            base_val = format_float(baseline[metric])
            delta = summary[metric] - baseline[metric]
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            lines.append(f"| {label} | {val} | {base_val} | {delta_str} |")
        else:
            lines.append(f"| {label} | {val} |")

    lines.extend([
        "",
        "## Per-Sample Results",
        "",
        "| ID | orig_gen_clip | orig_gen_xclip | orig_text_clip | gen_text_clip | orig_text_xclip | gen_text_xclip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for row in rows:
        lines.append(
            f"| {row['sample_id']} | "
            f"{format_float(row['orig_gen_clip'])} | "
            f"{format_float(row['orig_gen_xclip'])} | "
            f"{format_float(row['orig_text_clip'])} | "
            f"{format_float(row['gen_text_clip'])} | "
            f"{format_float(row['orig_text_xclip'])} | "
            f"{format_float(row['gen_text_xclip'])} |"
        )

    lines.extend([
        "",
        f"## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- Method: {method_name}",
        "",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Find aligned samples
    items = list_eval_items(args.orig_dir, args.gen_dir, args.caption_dir, limit=args.limit)
    if not items:
        raise SystemExit(
            f"No aligned samples found.\n"
            f"  orig-dir: {args.orig_dir}\n"
            f"  gen-dir: {args.gen_dir}\n"
            f"  caption-dir: {args.caption_dir}"
        )

    print(f"[{args.method_name}] Found {len(items)} aligned samples for evaluation", flush=True)

    # Load models
    clip_processor, clip_model, xclip_processor, xclip_model = build_models(
        args.device, args.clip_model, args.xclip_model
    )

    # Evaluate each sample
    rows = []
    for idx, item in enumerate(items, 1):
        sample_id = item["sample_id"]
        caption = read_text(item["caption_path"])
        orig_frames = sample_video_frames(item["orig_path"], args.sample_frames)
        gen_frames = sample_video_frames(item["gen_path"], args.sample_frames)

        # CLIP features
        clip_text = get_clip_text_feature(caption, clip_processor, clip_model, args.device)
        clip_orig = get_clip_video_feature(orig_frames, clip_processor, clip_model, args.device)
        clip_gen = get_clip_video_feature(gen_frames, clip_processor, clip_model, args.device)

        # X-CLIP features
        xclip_text = get_xclip_text_feature(caption, xclip_processor, xclip_model, args.device)
        xclip_orig = get_xclip_video_feature(orig_frames, xclip_processor, xclip_model, args.device)
        xclip_gen = get_xclip_video_feature(gen_frames, xclip_processor, xclip_model, args.device)

        row = {
            "sample_id": sample_id,
            "gen_name": item["gen_name"],
            "orig_text_clip": cosine_similarity(clip_orig, clip_text),
            "gen_text_clip": cosine_similarity(clip_gen, clip_text),
            "orig_gen_clip": cosine_similarity(clip_orig, clip_gen),
            "orig_text_xclip": cosine_similarity(xclip_orig, xclip_text),
            "gen_text_xclip": cosine_similarity(xclip_gen, xclip_text),
            "orig_gen_xclip": cosine_similarity(xclip_orig, xclip_gen),
        }
        rows.append(row)

        print(
            f"  [{idx}/{len(items)}] {sample_id}: "
            f"orig_gen_clip={row['orig_gen_clip']:.4f}, "
            f"orig_gen_xclip={row['orig_gen_xclip']:.4f}",
            flush=True,
        )

    # Sort by sample ID
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Compute summary
    summary = {
        "method": args.method_name,
        "count": len(rows),
        "orig_text_clip_mean": mean_of(rows, "orig_text_clip"),
        "gen_text_clip_mean": mean_of(rows, "gen_text_clip"),
        "orig_gen_clip_mean": mean_of(rows, "orig_gen_clip"),
        "orig_text_xclip_mean": mean_of(rows, "orig_text_xclip"),
        "gen_text_xclip_mean": mean_of(rows, "gen_text_xclip"),
        "orig_gen_xclip_mean": mean_of(rows, "orig_gen_xclip"),
    }

    # Load baseline for comparison if provided
    baseline = None
    if args.baseline_json and args.baseline_json.exists():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        print(f"\n  Baseline loaded from: {args.baseline_json}", flush=True)

    # Save CSV
    csv_path = args.output_dir / "reproduction_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id", "gen_name",
                "orig_text_clip", "gen_text_clip", "orig_gen_clip",
                "orig_text_xclip", "gen_text_xclip", "orig_gen_xclip",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON summary (compatible with P-Flow's format for comparison)
    json_path = args.output_dir / "eval_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save Markdown report
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, rows, summary, args.method_name, baseline)

    # Print final summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  [{args.method_name}] Reproduction Evaluation Complete", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Samples:          {summary['count']}", flush=True)
    print(f"  orig_gen_clip:    {format_float(summary['orig_gen_clip_mean'])}", flush=True)
    print(f"  orig_gen_xclip:   {format_float(summary['orig_gen_xclip_mean'])}", flush=True)
    if baseline:
        dc = summary['orig_gen_clip_mean'] - baseline.get('orig_gen_clip_mean', 0)
        dx = summary['orig_gen_xclip_mean'] - baseline.get('orig_gen_xclip_mean', 0)
        print(f"  Δ CLIP vs baseline: {dc:+.4f}", flush=True)
        print(f"  Δ XCLIP vs baseline: {dx:+.4f}", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"  Results: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
