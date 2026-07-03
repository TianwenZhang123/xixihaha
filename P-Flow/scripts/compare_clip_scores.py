#!/usr/bin/env python3
"""
Compare CLIP/X-CLIP scores across three caption directories:

1. video3_captions       — original VLM captions
2. video3_captions_v10   — v10 head/tail rewrite
3. video3_captions_v10_vlm — v10 + VLM factual correction

Usage:
    python scripts/compare_clip_scores.py \
        --video-dir data/video3 \
        --caption-dirs data/video3_captions data/video3_captions_v10 data/video3_captions_v10_vlm \
        --caption-names original v10 v10_vlm \
        --clip-model models/clip-vit-base-patch32 \
        --xclip-model models/xclip-base-patch32 \
        --output-file data/clip_comparison.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.clip_utils import (
    cosine_similarity,
    sample_video_frames,
    build_models,
    get_clip_text_feature,
    get_clip_video_feature,
    get_xclip_text_feature,
    get_xclip_video_feature,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CLIP/X-CLIP scores across multiple caption directories"
    )
    parser.add_argument("--video-dir", type=Path, required=True,
                        help="原始视频目录 (contains {id}.mp4)")
    parser.add_argument("--caption-dirs", type=Path, nargs="+", required=True,
                        help="Caption directories to compare")
    parser.add_argument("--caption-names", type=str, nargs="+", default=None,
                        help="Display names for each caption directory")
    parser.add_argument("--clip-model", type=str, default="models/clip-vit-base-patch32",
                        help="CLIP model path")
    parser.add_argument("--xclip-model", type=str, default="models/xclip-base-patch32",
                        help="X-CLIP model path")
    parser.add_argument("--sample-frames", type=int, default=8,
                        help="Frames to sample per video")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=None,
                        help="Specific sample IDs to evaluate (overrides --limit)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only first N samples (0=all)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-file", type=Path, default=None,
                        help="Save results to CSV file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine names
    if args.caption_names and len(args.caption_names) == len(args.caption_dirs):
        names = args.caption_names
    else:
        names = [d.name for d in args.caption_dirs]

    # Determine sample IDs
    if args.sample_ids:
        sample_ids = [str(sid) for sid in args.sample_ids]
    else:
        # Collect all IDs from the first caption directory
        all_ids = sorted(
            [p.stem for p in args.caption_dirs[0].glob("*.txt") if p.stem.isdigit()],
            key=lambda x: int(x)
        )
        if args.limit > 0:
            all_ids = all_ids[:args.limit]
        sample_ids = all_ids

    print(f"Sample IDs: {sample_ids}")
    print(f"Caption groups: {names}")
    print(f"Video dir: {args.video_dir}")
    print(f"Device: {args.device}")
    print()

    # Load models
    clip_processor, clip_model, xclip_processor, xclip_model = build_models(
        args.device, args.clip_model, args.xclip_model
    )
    print("Models loaded\n", flush=True)

    # Storage for results
    results = {name: {} for name in names}
    csv_rows = []

    # Main evaluation loop
    for idx, sid in enumerate(sample_ids, 1):
        video_path = args.video_dir / f"{sid}.mp4"
        if not video_path.exists():
            print(f"  [{idx}/{len(sample_ids)}] {sid}: video not found, skipping")
            continue

        # Sample frames and compute video features once per video
        frames = sample_video_frames(video_path, args.sample_frames)
        clip_video_feat = get_clip_video_feature(frames, clip_processor, clip_model, args.device)
        xclip_video_feat = get_xclip_video_feature(frames, xclip_processor, xclip_model, args.device)

        print(f"  [{idx}/{len(sample_ids)}] Sample {sid}:")

        row = {"sample_id": sid}

        for name, cap_dir in zip(names, args.caption_dirs):
            cap_file = cap_dir / f"{sid}.txt"
            if not cap_file.exists():
                results[name][sid] = {"clip": None, "xclip": None}
                row[f"{name}_clip"] = None
                row[f"{name}_xclip"] = None
                print(f"    {name:15s}: [missing]")
                continue

            caption = cap_file.read_text(encoding="utf-8").strip()
            word_count = len(caption.split())

            # CLIP score
            clip_text_feat = get_clip_text_feature(caption, clip_processor, clip_model, args.device)
            clip_score = cosine_similarity(clip_video_feat, clip_text_feat)

            # X-CLIP score
            xclip_text_feat = get_xclip_text_feature(caption, xclip_processor, xclip_model, args.device)
            xclip_score = cosine_similarity(xclip_video_feat, xclip_text_feat)

            results[name][sid] = {"clip": clip_score, "xclip": xclip_score, "words": word_count}
            row[f"{name}_clip"] = clip_score
            row[f"{name}_xclip"] = xclip_score
            row[f"{name}_words"] = word_count

            print(f"    {name:15s}: CLIP={clip_score:.4f}  X-CLIP={xclip_score:.4f}  ({word_count}w)")

        csv_rows.append(row)
        print()

    # ============================================================
    # Summary: CLIP scores
    # ============================================================
    print("=" * 90)
    print("Summary: CLIP (caption vs original video)")
    print("=" * 90)
    print()

    header = f"{'Sample':<8}" + "".join(f"{name:<18}" for name in names)
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        row = f"{sid:<8}"
        for name in names:
            entry = results[name].get(sid)
            if entry and entry["clip"] is not None:
                row += f"{entry['clip']:.4f}           "
            else:
                row += f"{'N/A':<18}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'AVG':<8}"
    for name in names:
        scores = [results[name][sid]["clip"] for sid in sample_ids
                  if sid in results[name] and results[name][sid]["clip"] is not None]
        if scores:
            avg = sum(scores) / len(scores)
            avg_row += f"{avg:.4f}           "
        else:
            avg_row += f"{'N/A':<18}"
    print(avg_row)

    # ============================================================
    # Summary: X-CLIP scores
    # ============================================================
    print()
    print("=" * 90)
    print("Summary: X-CLIP (caption vs original video)")
    print("=" * 90)
    print()

    header = f"{'Sample':<8}" + "".join(f"{name:<18}" for name in names)
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        row = f"{sid:<8}"
        for name in names:
            entry = results[name].get(sid)
            if entry and entry["xclip"] is not None:
                row += f"{entry['xclip']:.4f}           "
            else:
                row += f"{'N/A':<18}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'AVG':<8}"
    for name in names:
        scores = [results[name][sid]["xclip"] for sid in sample_ids
                  if sid in results[name] and results[name][sid]["xclip"] is not None]
        if scores:
            avg = sum(scores) / len(scores)
            avg_row += f"{avg:.4f}           "
        else:
            avg_row += f"{'N/A':<18}"
    print(avg_row)

    # ============================================================
    # Improvement analysis (if 3 groups provided)
    # ============================================================
    if len(names) == 3:
        print()
        print("=" * 90)
        print("Improvement Analysis: v10 vs original, v10_vlm vs v10")
        print("=" * 90)
        print()

        orig_name, v10_name, vlm_name = names[0], names[1], names[2]

        # CLIP improvements
        clip_improvements_v10 = []
        clip_improvements_vlm = []
        xclip_improvements_v10 = []
        xclip_improvements_vlm = []

        for sid in sample_ids:
            orig = results[orig_name].get(sid, {})
            v10 = results[v10_name].get(sid, {})
            vlm = results[vlm_name].get(sid, {})

            if orig.get("clip") and v10.get("clip"):
                clip_improvements_v10.append(v10["clip"] - orig["clip"])
            if v10.get("clip") and vlm.get("clip"):
                clip_improvements_vlm.append(vlm["clip"] - v10["clip"])
            if orig.get("xclip") and v10.get("xclip"):
                xclip_improvements_v10.append(v10["xclip"] - orig["xclip"])
            if v10.get("xclip") and vlm.get("xclip"):
                xclip_improvements_vlm.append(vlm["xclip"] - v10["xclip"])

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        print("CLIP Score:")
        print(f"  v10 vs original:     Δ = {avg(clip_improvements_v10):+.4f}  (n={len(clip_improvements_v10)})")
        print(f"  v10_vlm vs v10:      Δ = {avg(clip_improvements_vlm):+.4f}  (n={len(clip_improvements_vlm)})")
        print()
        print("X-CLIP Score:")
        print(f"  v10 vs original:     Δ = {avg(xclip_improvements_v10):+.4f}  (n={len(xclip_improvements_v10)})")
        print(f"  v10_vlm vs v10:      Δ = {avg(xclip_improvements_vlm):+.4f}  (n={len(xclip_improvements_vlm)})")

    # ============================================================
    # Save CSV
    # ============================================================
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sample_id"]
        for name in names:
            fieldnames.extend([f"{name}_clip", f"{name}_xclip", f"{name}_words"])

        with open(args.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print()
        print(f"Results saved to: {args.output_file}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
