#!/usr/bin/env python3
"""
VMAD Content Consistency & Disentanglement Evaluation.

评估两个关键属性:
    1. 内容一致性: 生成视频是否忠实于新 content prompt
    2. 内容解耦度: motion asset 是否泄漏了源视频的内容信息

指标:
    1. content_clip_score: CLIP(generated_video, content_prompt)
       - 生成视频与目标内容描述的匹配度 (越高越好)

    2. content_leakage_score: CLIP(generated_video, source_caption)
       - 生成视频与源视频描述的相似度 (越低越好, 表示无内容泄漏)

    3. cross_content_motion_var: 同一 asset 应用到不同 content 的运动一致性
       - X-CLIP 特征方差 (越低越好, 表示运动不受内容影响)

    4. disentangle_ratio: content_clip_score / content_leakage_score
       - 解耦比 (越高越好, >1 表示新内容主导)

用法:
    # 评估内容一致性 (需要 content prompts)
    python evaluation/run_content_consistency_eval.py \
        --gen-dir ./outputs/vmad_batch \
        --content-dir ./outputs/vmad_batch/content_prompts \
        --source-caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/vmad_batch/eval_content

    # 评估跨内容运动一致性 (同一 asset, 多个 content)
    python evaluation/run_content_consistency_eval.py \
        --cross-content-dir ./outputs/cross_content_experiment \
        --output-dir ./outputs/eval_cross_content

数据目录结构:
    gen-dir/{id}.mp4                    # 生成视频
    content-dir/{id}.txt                # 对应的 content prompt
    source-caption-dir/{id}.txt         # 源视频的 caption
    cross-content-dir/{asset_id}/
        content_00/generated.mp4        # asset 应用到 content 0
        content_01/generated.mp4        # asset 应用到 content 1
        ...
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


# ============================================================
# Default paths
# ============================================================
DEFAULT_GEN_DIR = Path("/root/autodl-tmp/outputs/vmad_batch")
DEFAULT_CONTENT_DIR = Path("/root/autodl-tmp/outputs/vmad_batch/content_prompts")
DEFAULT_SOURCE_CAPTION_DIR = Path("/root/autodl-tmp/data/video-200/captions_qwen")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/outputs/eval_results/content_consistency")
DEFAULT_CLIP_MODEL = "/root/autodl-tmp/models/clip-vit-base-patch32"
DEFAULT_XCLIP_MODEL = "/root/autodl-tmp/models/xclip-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VMAD Content Consistency & Disentanglement Evaluation"
    )

    # Mode 1: Per-sample evaluation
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--content-dir", type=Path, default=DEFAULT_CONTENT_DIR,
                        help="Directory containing content prompts ({id}.txt)")
    parser.add_argument("--source-caption-dir", type=Path, default=DEFAULT_SOURCE_CAPTION_DIR,
                        help="Directory containing source video captions ({id}.txt)")

    # Mode 2: Cross-content evaluation
    parser.add_argument("--cross-content-dir", type=Path, default=None,
                        help="Directory for cross-content experiment (overrides Mode 1)")

    # Output
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")

    # Models
    parser.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--xclip-model", type=str, default=DEFAULT_XCLIP_MODEL)

    # Options
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


# ============================================================
# Utility functions
# ============================================================

def l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array)
    return array / norm if norm > 0 else array


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return float(np.dot(a, b))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_numeric_id(path: Path) -> Optional[str]:
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else None


def sample_video_frames(video_path: Path, num_frames: int) -> list:
    """Uniformly sample frames from a video file."""
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


# ============================================================
# Model loading
# ============================================================

def load_models(device: str, clip_model_name: str, xclip_model_name: str):
    """Load CLIP and X-CLIP models."""
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, XCLIPModel

    print(f"Loading CLIP: {clip_model_name}", flush=True)
    try:
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
    except Exception:
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    print(f"Loading X-CLIP: {xclip_model_name}", flush=True)
    try:
        xclip_processor = AutoProcessor.from_pretrained(xclip_model_name, local_files_only=True)
        xclip_model = XCLIPModel.from_pretrained(xclip_model_name, local_files_only=True).to(device)
    except Exception:
        xclip_processor = AutoProcessor.from_pretrained(xclip_model_name)
        xclip_model = XCLIPModel.from_pretrained(xclip_model_name).to(device)
    xclip_model.eval()

    return clip_processor, clip_model, xclip_processor, xclip_model


@torch.inference_mode()
def get_clip_text_feature(text: str, processor, model, device: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", truncation=True).to(device)
    features = model.get_text_features(**inputs)
    return features[0].detach().float().cpu().numpy()


@torch.inference_mode()
def get_clip_video_feature(frames: list, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=frames, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    return features.detach().float().cpu().numpy().mean(axis=0)


@torch.inference_mode()
def get_xclip_video_feature(frames: list, processor, model, device: str) -> np.ndarray:
    np_frames = [np.array(frame) for frame in frames]
    inputs = processor(videos=[np_frames], return_tensors="pt").to(device)
    features = model.get_video_features(**inputs)
    return features[0].detach().float().cpu().numpy()


# ============================================================
# Mode 1: Per-sample content consistency
# ============================================================

def evaluate_per_sample(args, clip_processor, clip_model, xclip_processor, xclip_model):
    """Evaluate content consistency per sample."""
    gen_dir = args.gen_dir
    content_dir = args.content_dir
    source_dir = args.source_caption_dir

    # Find aligned items
    content_map = {p.stem: p for p in content_dir.glob("*.txt")}
    source_map = {p.stem: p for p in source_dir.glob("*.txt")}

    items = []
    for gen_path in sorted(gen_dir.glob("*.mp4"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        sample_id = extract_numeric_id(gen_path)
        if not sample_id:
            continue
        content_path = content_map.get(sample_id)
        source_path = source_map.get(sample_id)
        if content_path and source_path:
            items.append({
                "sample_id": sample_id,
                "gen_path": gen_path,
                "content_path": content_path,
                "source_path": source_path,
            })

    if args.limit > 0:
        items = items[:args.limit]

    if not items:
        raise SystemExit(
            f"No aligned samples found.\n"
            f"  gen-dir: {gen_dir}\n"
            f"  content-dir: {content_dir}\n"
            f"  source-caption-dir: {source_dir}"
        )

    print(f"Found {len(items)} samples for content consistency evaluation", flush=True)

    rows = []
    for idx, item in enumerate(items, 1):
        content_prompt = read_text(item["content_path"])
        source_caption = read_text(item["source_path"])

        gen_frames = sample_video_frames(item["gen_path"], args.sample_frames)

        # CLIP features
        clip_gen = get_clip_video_feature(gen_frames, clip_processor, clip_model, args.device)
        clip_content = get_clip_text_feature(content_prompt, clip_processor, clip_model, args.device)
        clip_source = get_clip_text_feature(source_caption, clip_processor, clip_model, args.device)

        content_score = cosine_similarity(clip_gen, clip_content)
        leakage_score = cosine_similarity(clip_gen, clip_source)
        disentangle_ratio = content_score / (leakage_score + 1e-8)

        row = {
            "sample_id": item["sample_id"],
            "content_clip_score": content_score,
            "content_leakage_score": leakage_score,
            "disentangle_ratio": disentangle_ratio,
            "content_prompt": content_prompt[:80],
        }
        rows.append(row)

        if args.verbose or idx % 10 == 0:
            print(
                f"[{idx}/{len(items)}] {row['sample_id']} "
                f"Content={content_score:.4f} Leakage={leakage_score:.4f} "
                f"Ratio={disentangle_ratio:.3f}",
                flush=True,
            )

    return rows


# ============================================================
# Mode 2: Cross-content motion consistency
# ============================================================

def evaluate_cross_content(args, clip_processor, clip_model, xclip_processor, xclip_model):
    """
    Evaluate cross-content motion consistency.

    For each asset, multiple content prompts are applied.
    We measure the variance of X-CLIP motion features across contents.
    Low variance = motion is content-independent = good disentanglement.
    """
    cross_dir = args.cross_content_dir
    if not cross_dir or not cross_dir.exists():
        return []

    # Expected structure: cross_dir/{asset_id}/content_{xx}/generated.mp4
    asset_dirs = sorted([d for d in cross_dir.iterdir() if d.is_dir()])

    if not asset_dirs:
        print(f"No asset directories found in {cross_dir}", flush=True)
        return []

    print(f"Found {len(asset_dirs)} assets for cross-content evaluation", flush=True)

    rows = []
    for asset_dir in asset_dirs:
        asset_id = asset_dir.name

        # Find all content variants
        content_dirs = sorted([d for d in asset_dir.iterdir() if d.is_dir()])
        video_paths = []
        for cd in content_dirs:
            video_path = cd / "generated.mp4"
            if video_path.exists():
                video_paths.append(video_path)

        if len(video_paths) < 2:
            continue

        # Extract X-CLIP features for each variant
        xclip_features = []
        for vp in video_paths:
            frames = sample_video_frames(vp, args.sample_frames)
            feat = get_xclip_video_feature(frames, xclip_processor, xclip_model, args.device)
            xclip_features.append(feat)

        # Compute variance of features across contents
        features_array = np.stack(xclip_features, axis=0)  # (N_contents, D)
        feature_variance = float(features_array.var(axis=0).mean())

        # Compute pairwise X-CLIP similarity
        n = len(xclip_features)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(xclip_features[i], xclip_features[j])
                pairwise_sims.append(sim)

        mean_pairwise_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0

        row = {
            "asset_id": asset_id,
            "num_contents": len(video_paths),
            "xclip_feature_variance": feature_variance,
            "xclip_pairwise_sim_mean": mean_pairwise_sim,
        }
        rows.append(row)

        if args.verbose:
            print(
                f"  Asset {asset_id}: {len(video_paths)} contents, "
                f"Var={feature_variance:.6f}, PairSim={mean_pairwise_sim:.4f}",
                flush=True,
            )

    return rows


# ============================================================
# Output
# ============================================================

def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def write_per_sample_results(output_dir: Path, rows: list):
    """Save per-sample content consistency results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "content_consistency_results.csv"
    fieldnames = ["sample_id", "content_clip_score", "content_leakage_score",
                  "disentangle_ratio", "content_prompt"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    def mean_of(key):
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "count": len(rows),
        "content_clip_score_mean": mean_of("content_clip_score"),
        "content_leakage_score_mean": mean_of("content_leakage_score"),
        "disentangle_ratio_mean": mean_of("disentangle_ratio"),
    }

    json_path = output_dir / "content_consistency_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Markdown
    md_path = output_dir / "content_consistency_report.md"
    lines = [
        "# VMAD Content Consistency Evaluation",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- Content CLIP Score (mean): {format_float(summary['content_clip_score_mean'])} (higher is better)",
        f"- Content Leakage Score (mean): {format_float(summary['content_leakage_score_mean'])} (lower is better)",
        f"- Disentangle Ratio (mean): {format_float(summary['disentangle_ratio_mean'])} (higher is better, >1 = good)",
        "",
        "## Interpretation",
        "",
        "- Content CLIP Score > 0.25: Generated video matches target content",
        "- Content Leakage < 0.20: No significant source content leakage",
        "- Disentangle Ratio > 1.2: Good content-motion separation",
        "",
        "## Per-Sample Results",
        "",
        "| ID | Content Score | Leakage Score | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {format_float(row['content_clip_score'])} | "
            f"{format_float(row['content_leakage_score'])} | "
            f"{format_float(row['disentangle_ratio'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def write_cross_content_results(output_dir: Path, rows: list):
    """Save cross-content evaluation results."""
    if not rows:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "cross_content_results.csv"
    fieldnames = ["asset_id", "num_contents", "xclip_feature_variance", "xclip_pairwise_sim_mean"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def mean_of(key):
        vals = [r[key] for r in rows]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "num_assets": len(rows),
        "xclip_variance_mean": mean_of("xclip_feature_variance"),
        "xclip_pairwise_sim_mean": mean_of("xclip_pairwise_sim_mean"),
    }

    json_path = output_dir / "cross_content_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Load models
    clip_processor, clip_model, xclip_processor, xclip_model = load_models(
        args.device, args.clip_model, args.xclip_model
    )

    # Mode 2: Cross-content evaluation
    if args.cross_content_dir and args.cross_content_dir.exists():
        print("\n=== Cross-Content Motion Consistency Evaluation ===", flush=True)
        cross_rows = evaluate_cross_content(
            args, clip_processor, clip_model, xclip_processor, xclip_model
        )
        cross_summary = write_cross_content_results(args.output_dir / "cross_content", cross_rows)

        if cross_summary:
            print(f"\nCross-Content Results:", flush=True)
            print(f"  Assets evaluated: {cross_summary['num_assets']}", flush=True)
            print(f"  X-CLIP Variance (mean): {format_float(cross_summary['xclip_variance_mean'])} (lower is better)", flush=True)
            print(f"  X-CLIP Pairwise Sim (mean): {format_float(cross_summary['xclip_pairwise_sim_mean'])} (higher is better)", flush=True)

    # Mode 1: Per-sample evaluation
    if args.gen_dir.exists() and args.content_dir.exists():
        print("\n=== Per-Sample Content Consistency Evaluation ===", flush=True)
        per_sample_rows = evaluate_per_sample(
            args, clip_processor, clip_model, xclip_processor, xclip_model
        )
        per_sample_summary = write_per_sample_results(args.output_dir, per_sample_rows)

        print(f"\nContent Consistency Results:", flush=True)
        print(f"  Samples: {per_sample_summary['count']}", flush=True)
        print(f"  Content Score (mean): {format_float(per_sample_summary['content_clip_score_mean'])}", flush=True)
        print(f"  Leakage Score (mean): {format_float(per_sample_summary['content_leakage_score_mean'])}", flush=True)
        print(f"  Disentangle Ratio (mean): {format_float(per_sample_summary['disentangle_ratio_mean'])}", flush=True)

    print("\nEvaluation complete!", flush=True)


if __name__ == "__main__":
    main()
