#!/usr/bin/env python3
"""
Caption Quality Evaluation — 轻量级 CLIP/X-CLIP text-video 相似度对比

不需要生成视频，只评估 caption 和原始视频的匹配程度。
用于对比不同改写策略的 prompt 质量。

指标:
    - orig_text_clip:  CLIP cosine(原始视频帧均值, caption_text)
    - orig_text_xclip: X-CLIP cosine(原始视频temporal, caption_text)

用法:
    python scripts/eval_caption_quality.py \
        --video-dir data/videos \
        --caption-dirs data/captions_qwen data/captions_hybrid_old data/captions_hybrid data/captions_hybrid_v4 data/captions_negative \
        --caption-names baseline old_hybrid hybrid_v3 hybrid_v4 negative \
        --clip-model models/clip-vit-base-patch32 \
        --xclip-model models/xclip-base-patch32 \
        --limit 10
"""

import argparse
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
)


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Caption vs Original Video similarity evaluation")
    parser.add_argument("--video-dir", type=Path, required=True,
                        help="原始视频目录 (包含 {id}.mp4)")
    parser.add_argument("--caption-dirs", type=Path, nargs="+", required=True,
                        help="待对比的 caption 目录列表")
    parser.add_argument("--caption-names", type=str, nargs="+", default=None,
                        help="每个 caption 目录的别名（用于显示，与 --caption-dirs 一一对应）")
    parser.add_argument("--clip-model", type=str, default="models/clip-vit-base-patch32",
                        help="CLIP 模型路径")
    parser.add_argument("--xclip-model", type=str, default="models/xclip-base-patch32",
                        help="X-CLIP 模型路径")
    parser.add_argument("--sample-frames", type=int, default=8,
                        help="每个视频均匀采样帧数")
    parser.add_argument("--limit", type=int, default=10,
                        help="只评估前 N 个样本 (0=全部)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=None,
                        help="指定样本 ID 列表（优先于 --limit）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.caption_names and len(args.caption_names) == len(args.caption_dirs):
        names = args.caption_names
    else:
        names = [d.name for d in args.caption_dirs]

    if args.sample_ids:
        sample_ids = [str(sid) for sid in args.sample_ids]
    else:
        all_ids = sorted(
            [p.stem for p in args.caption_dirs[0].glob("*.txt") if p.stem.isdigit()],
            key=lambda x: int(x)
        )
        if args.limit > 0:
            all_ids = all_ids[:args.limit]
        sample_ids = all_ids

    print(f"评估样本: {sample_ids}")
    print(f"Caption 组: {names}")
    print(f"设备: {args.device}")
    print()

    clip_processor, clip_model, xclip_processor, xclip_model = build_models(
        args.device, args.clip_model, args.xclip_model
    )
    print("模型加载完成\n", flush=True)

    results = {name: {} for name in names}

    for idx, sid in enumerate(sample_ids, 1):
        video_path = args.video_dir / f"{sid}.mp4"
        if not video_path.exists():
            print(f"  [{idx}/{len(sample_ids)}] {sid}: 视频不存在，跳过")
            continue

        frames = sample_video_frames(video_path, args.sample_frames)
        clip_video_feat = get_clip_video_feature(frames, clip_processor, clip_model, args.device)
        xclip_video_feat = get_xclip_video_feature(frames, xclip_processor, xclip_model, args.device)

        print(f"  [{idx}/{len(sample_ids)}] Sample {sid}:")

        for name, cap_dir in zip(names, args.caption_dirs):
            cap_file = cap_dir / f"{sid}.txt"
            if not cap_file.exists():
                results[name][sid] = {"clip": None, "xclip": None}
                print(f"    {name:20s}: [missing]")
                continue

            caption = cap_file.read_text(encoding="utf-8").strip()

            clip_text_feat = get_clip_text_feature(caption, clip_processor, clip_model, args.device)
            clip_score = cosine_similarity(clip_video_feat, clip_text_feat)

            xclip_text_feat = get_xclip_text_feature(caption, xclip_processor, xclip_model, args.device)
            xclip_score = cosine_similarity(xclip_video_feat, xclip_text_feat)

            results[name][sid] = {"clip": clip_score, "xclip": xclip_score}
            print(f"    {name:20s}: CLIP={clip_score:.4f}  X-CLIP={xclip_score:.4f}")

        print()

    # ============================================================
    # 汇总
    # ============================================================
    print("=" * 80)
    print("汇总: orig_text_clip (caption 与原始视频的 CLIP 相似度)")
    print("=" * 80)
    print()

    header = f"{'Sample':<8}" + "".join(f"{name:<20}" for name in names)
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        row = f"{sid:<8}"
        for name in names:
            entry = results[name].get(sid)
            if entry and entry["clip"] is not None:
                row += f"{entry['clip']:.4f}              "
            else:
                row += f"{'N/A':<20}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'AVG':<8}"
    for name in names:
        scores = [results[name][sid]["clip"] for sid in sample_ids
                  if sid in results[name] and results[name][sid]["clip"] is not None]
        if scores:
            avg_row += f"{sum(scores)/len(scores):.4f}              "
        else:
            avg_row += f"{'N/A':<20}"
    print(avg_row)

    print()
    print("=" * 80)
    print("汇总: orig_text_xclip (caption 与原始视频的 X-CLIP 相似度)")
    print("=" * 80)
    print()

    header = f"{'Sample':<8}" + "".join(f"{name:<20}" for name in names)
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        row = f"{sid:<8}"
        for name in names:
            entry = results[name].get(sid)
            if entry and entry["xclip"] is not None:
                row += f"{entry['xclip']:.4f}              "
            else:
                row += f"{'N/A':<20}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'AVG':<8}"
    for name in names:
        scores = [results[name][sid]["xclip"] for sid in sample_ids
                  if sid in results[name] and results[name][sid]["xclip"] is not None]
        if scores:
            avg_row += f"{sum(scores)/len(scores):.4f}              "
        else:
            avg_row += f"{'N/A':<20}"
    print(avg_row)

    print()
    print("完成!")


if __name__ == "__main__":
    main()
