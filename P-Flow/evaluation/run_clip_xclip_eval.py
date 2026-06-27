#!/usr/bin/env python3
"""
CLIP / X-CLIP Evaluation Script for Video Reproduction.

Computes 6 similarity metrics between original videos, generated videos, and captions:
    1. orig_text_clip   : CLIP cosine(original_video_frames, caption)
    2. gen_text_clip    : CLIP cosine(generated_video_frames, caption)
    3. orig_gen_clip    : CLIP cosine(original_video_frames, generated_video_frames)
    4. orig_text_xclip  : X-CLIP cosine(original_video, caption)
    5. gen_text_xclip   : X-CLIP cosine(generated_video, caption)
    6. orig_gen_xclip   : X-CLIP cosine(original_video, generated_video)

Usage:
    # Evaluate baseline outputs
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --gen-dir /root/autodl-tmp/outputs/baseline_batch \
        --caption-dir /root/autodl-tmp/data/video-200/captions_qwen \
        --output-dir /root/autodl-tmp/outputs/baseline_batch/eval_clip

    # Evaluate P-Flow outputs
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --gen-dir /root/autodl-tmp/outputs/pflow_batch \
        --caption-dir /root/autodl-tmp/data/video-200/captions_qwen \
        --output-dir /root/autodl-tmp/outputs/pflow_batch/eval_clip

Models used (same as Video2Prompt collaborator):
    - CLIP: openai/clip-vit-base-patch32
    - X-CLIP: microsoft/xclip-base-patch32
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from evaluation.clip_utils import (
    cosine_similarity,
    sample_video_frames,
    build_models,
    get_clip_text_feature,
    get_clip_video_feature,
    get_xclip_text_feature,
    get_xclip_video_feature,
    format_float,
    mean_of,
    read_text,
    extract_numeric_id,
)


# ============================================================
# Default paths (AutoDL server)
# ============================================================
DEFAULT_ORIG_DIR = Path("data/videos")
DEFAULT_GEN_DIR = Path("outputs/baseline_batch")
DEFAULT_CAPTION_DIR = Path("data/captions_hybrid")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_results/clip_xclip")
DEFAULT_CLIP_MODEL = "models/clip-vit-base-patch32"
DEFAULT_XCLIP_MODEL = "models/xclip-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP / X-CLIP evaluation: original vs generated videos vs captions"
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
    return parser.parse_args()


def list_eval_items(orig_dir: Path, gen_dir: Path, caption_dir: Path,
                    limit: int = 0) -> list[dict]:
    """Find aligned triplets: original video + generated video + caption.

    Supports two gen_dir layouts:
      1. Flat: gen_dir/{id}.mp4
      2. Nested: gen_dir/sample_{id}/{id}.mp4 (run.py default output)
    """
    orig_map = {p.stem: p for p in orig_dir.glob("*.mp4")}
    caption_map = {p.stem: p for p in caption_dir.glob("*.txt")}

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
        gen_path = gen_map[sample_id]
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


def write_markdown_summary(output_path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# CLIP / X-CLIP Evaluation Results",
        "",
        "## Metrics Description",
        "",
        "- `orig_text_clip`: CLIP similarity between original video and caption",
        "- `gen_text_clip`: CLIP similarity between generated video and caption",
        "- `orig_gen_clip`: CLIP similarity between original and generated video",
        "- `orig_text_xclip`: X-CLIP similarity between original video and caption",
        "- `gen_text_xclip`: X-CLIP similarity between generated video and caption",
        "- `orig_gen_xclip`: X-CLIP similarity between original and generated video",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- orig_text_clip mean: {format_float(summary['orig_text_clip_mean'])}",
        f"- gen_text_clip mean: {format_float(summary['gen_text_clip_mean'])}",
        f"- orig_gen_clip mean: {format_float(summary['orig_gen_clip_mean'])}",
        f"- orig_text_xclip mean: {format_float(summary['orig_text_xclip_mean'])}",
        f"- gen_text_xclip mean: {format_float(summary['gen_text_xclip_mean'])}",
        f"- orig_gen_xclip mean: {format_float(summary['orig_gen_xclip_mean'])}",
        "",
        "## Per-Sample Results",
        "",
        "| ID | Generated | orig_text_clip | gen_text_clip | orig_gen_clip | orig_text_xclip | gen_text_xclip | orig_gen_xclip |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['gen_name']} | {format_float(row['orig_text_clip'])} | "
            f"{format_float(row['gen_text_clip'])} | {format_float(row['orig_gen_clip'])} | "
            f"{format_float(row['orig_text_xclip'])} | {format_float(row['gen_text_xclip'])} | "
            f"{format_float(row['orig_gen_xclip'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find aligned samples
    items = list_eval_items(args.orig_dir, args.gen_dir, args.caption_dir, limit=args.limit)
    if not items:
        raise SystemExit(
            f"No aligned samples found.\n"
            f"  orig-dir: {args.orig_dir}\n"
            f"  gen-dir: {args.gen_dir}\n"
            f"  caption-dir: {args.caption_dir}"
        )

    print(f"Found {len(items)} aligned samples for evaluation", flush=True)

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
            f"[{idx}/{len(items)}] {sample_id} "
            f"CLIP(o-t={row['orig_text_clip']:.4f}, g-t={row['gen_text_clip']:.4f}, o-g={row['orig_gen_clip']:.4f}) "
            f"XCLIP(o-t={row['orig_text_xclip']:.4f}, g-t={row['gen_text_xclip']:.4f}, o-g={row['orig_gen_xclip']:.4f})",
            flush=True,
        )

    # Sort by sample ID
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Save CSV
    csv_path = args.output_dir / "clip_xclip_results.csv"
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

    # Compute summary
    summary = {
        "count": len(rows),
        "orig_text_clip_mean": mean_of(rows, "orig_text_clip"),
        "gen_text_clip_mean": mean_of(rows, "gen_text_clip"),
        "orig_gen_clip_mean": mean_of(rows, "orig_gen_clip"),
        "orig_text_xclip_mean": mean_of(rows, "orig_text_xclip"),
        "gen_text_xclip_mean": mean_of(rows, "gen_text_xclip"),
        "orig_gen_xclip_mean": mean_of(rows, "orig_gen_xclip"),
    }

    # Save Markdown summary
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, rows, summary)

    # Save JSON summary
    json_path = args.output_dir / "eval_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nEvaluation complete!", flush=True)
    print(f"  Samples: {summary['count']}", flush=True)
    print(f"  orig_gen_clip mean:  {format_float(summary['orig_gen_clip_mean'])}", flush=True)
    print(f"  orig_gen_xclip mean: {format_float(summary['orig_gen_xclip_mean'])}", flush=True)
    print(f"  CSV: {csv_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
