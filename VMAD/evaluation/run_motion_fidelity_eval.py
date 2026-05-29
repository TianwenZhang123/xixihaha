#!/usr/bin/env python3
"""
VMAD Motion Fidelity Evaluation.

评估运动迁移的保真度：生成视频是否保留了源视频的运动模式。

指标:
    1. optical_flow_epe: 光流端点误差 (Optical Flow End-Point Error)
       - 计算源视频和生成视频的光流场，比较运动模式差异
       - 越低越好 (0 = 完美匹配)

    2. xclip_motion_sim: X-CLIP 运动相似度
       - 使用 X-CLIP 的时序感知特征比较两个视频的运动语义
       - 越高越好 (1.0 = 完美匹配)

    3. clip_frame_sim: CLIP 帧级相似度
       - 逐帧 CLIP 特征的平均余弦相似度
       - 衡量视觉层面的运动一致性

    4. motion_magnitude_ratio: 运动幅度比
       - 生成视频运动幅度 / 源视频运动幅度
       - 理想值 = 1.0

用法:
    # 评估单个 asset 的迁移效果
    python evaluation/run_motion_fidelity_eval.py \
        --orig-dir ../P-Flow/data/videos_200 \
        --gen-dir ./outputs/vmad_batch \
        --output-dir ./outputs/vmad_batch/eval_motion

    # 对比 baseline (text-only)
    python evaluation/run_motion_fidelity_eval.py \
        --orig-dir ../P-Flow/data/videos_200 \
        --gen-dir ./outputs/baseline_text_only \
        --output-dir ./outputs/baseline_text_only/eval_motion

    # 限制评估数量 (调试)
    python evaluation/run_motion_fidelity_eval.py \
        --orig-dir ../P-Flow/data/videos_200 \
        --gen-dir ./outputs/vmad_batch \
        --output-dir ./outputs/eval_debug \
        --limit 10

数据目录结构 (复用 P-Flow 的 data):
    P-Flow/data/videos_200/{id}.mp4      # 源视频 (参考运动)
    VMAD/outputs/{method}/{id}.mp4       # 生成视频 (迁移结果)
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
DEFAULT_ORIG_DIR = Path("/root/autodl-tmp/data/video-200/water_mark_out")
DEFAULT_GEN_DIR = Path("/root/autodl-tmp/outputs/vmad_batch")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/outputs/eval_results/motion_fidelity")
DEFAULT_CLIP_MODEL = "/root/autodl-tmp/models/clip-vit-base-patch32"
DEFAULT_XCLIP_MODEL = "/root/autodl-tmp/models/xclip-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VMAD Motion Fidelity Evaluation: optical flow EPE + X-CLIP motion similarity"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL,
                        help="Path to CLIP model")
    parser.add_argument("--xclip-model", type=str, default=DEFAULT_XCLIP_MODEL,
                        help="Path to X-CLIP model")
    parser.add_argument("--sample-frames", type=int, default=16,
                        help="Number of frames to uniformly sample per video")
    parser.add_argument("--flow-method", type=str, default="farneback",
                        choices=["farneback", "raft"],
                        help="Optical flow method (farneback=CPU, raft=GPU)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N samples (0 = all)")
    parser.add_argument("--verbose", "-v", action="store_true")
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


def extract_numeric_id(path: Path) -> Optional[str]:
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else None


def list_eval_items(orig_dir: Path, gen_dir: Path, limit: int = 0) -> list:
    """Find aligned pairs: original video + generated video."""
    orig_map = {p.stem: p for p in orig_dir.glob("*.mp4")}
    items = []
    for gen_path in sorted(gen_dir.glob("*.mp4"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        sample_id = extract_numeric_id(gen_path)
        if not sample_id:
            continue
        orig_path = orig_map.get(sample_id)
        if orig_path:
            items.append({
                "sample_id": sample_id,
                "orig_path": orig_path,
                "gen_path": gen_path,
            })
    if limit > 0:
        items = items[:limit]
    return items


def sample_video_frames(video_path: Path, num_frames: int) -> list:
    """Uniformly sample frames from a video file as PIL Images."""
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


def sample_video_frames_gray(video_path: Path, num_frames: int) -> list:
    """Sample frames as grayscale numpy arrays for optical flow."""
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


# ============================================================
# Optical Flow Computation
# ============================================================

def compute_optical_flow_farneback(frames: list) -> list:
    """
    Compute dense optical flow between consecutive frames using Farneback method.

    Returns:
        List of flow fields, each shape (H, W, 2)
    """
    flows = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)
    return flows


def compute_flow_epe(flows_orig: list, flows_gen: list) -> float:
    """
    Compute End-Point Error between two sets of optical flow fields.

    EPE = mean(||flow_orig - flow_gen||_2) over all pixels and frames.
    """
    total_epe = 0.0
    total_pixels = 0

    for f_orig, f_gen in zip(flows_orig, flows_gen):
        # Resize if shapes don't match
        if f_orig.shape[:2] != f_gen.shape[:2]:
            h, w = min(f_orig.shape[0], f_gen.shape[0]), min(f_orig.shape[1], f_gen.shape[1])
            f_orig = cv2.resize(f_orig, (w, h))
            f_gen = cv2.resize(f_gen, (w, h))

        diff = f_orig - f_gen
        epe = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
        total_epe += epe.sum()
        total_pixels += epe.size

    if total_pixels == 0:
        return 0.0
    return float(total_epe / total_pixels)


def compute_flow_magnitude(flows: list) -> float:
    """Compute average flow magnitude across all frames."""
    total_mag = 0.0
    total_pixels = 0

    for flow in flows:
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        total_mag += mag.sum()
        total_pixels += mag.size

    if total_pixels == 0:
        return 0.0
    return float(total_mag / total_pixels)


def compute_flow_direction_similarity(flows_orig: list, flows_gen: list) -> float:
    """
    Compute cosine similarity of flow directions (ignoring magnitude).

    Measures whether motion directions match, regardless of speed.
    """
    cos_sims = []

    for f_orig, f_gen in zip(flows_orig, flows_gen):
        if f_orig.shape[:2] != f_gen.shape[:2]:
            h, w = min(f_orig.shape[0], f_gen.shape[0]), min(f_orig.shape[1], f_gen.shape[1])
            f_orig = cv2.resize(f_orig, (w, h))
            f_gen = cv2.resize(f_gen, (w, h))

        # Flatten to (N, 2)
        orig_flat = f_orig.reshape(-1, 2)
        gen_flat = f_gen.reshape(-1, 2)

        # Normalize
        orig_norm = np.linalg.norm(orig_flat, axis=1, keepdims=True) + 1e-8
        gen_norm = np.linalg.norm(gen_flat, axis=1, keepdims=True) + 1e-8

        orig_unit = orig_flat / orig_norm
        gen_unit = gen_flat / gen_norm

        # Cosine similarity per pixel
        cos_sim = (orig_unit * gen_unit).sum(axis=1)

        # Only consider pixels with significant motion
        mask = (orig_norm.squeeze() > 0.5) & (gen_norm.squeeze() > 0.5)
        if mask.sum() > 0:
            cos_sims.append(cos_sim[mask].mean())

    if not cos_sims:
        return 0.0
    return float(np.mean(cos_sims))


# ============================================================
# CLIP / X-CLIP Evaluation
# ============================================================

def load_clip_models(device: str, clip_model_name: str, xclip_model_name: str):
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
def get_clip_video_feature(frames: list, processor, model, device: str) -> np.ndarray:
    """Get CLIP image embedding (mean over frames)."""
    inputs = processor(images=frames, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    return features.detach().float().cpu().numpy().mean(axis=0)


@torch.inference_mode()
def get_xclip_video_feature(frames: list, processor, model, device: str) -> np.ndarray:
    """Get X-CLIP video embedding (temporal-aware)."""
    np_frames = [np.array(frame) for frame in frames]
    inputs = processor(videos=[np_frames], return_tensors="pt").to(device)
    features = model.get_video_features(**inputs)
    return features[0].detach().float().cpu().numpy()


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_sample(
    item: dict,
    num_frames: int,
    clip_processor, clip_model,
    xclip_processor, xclip_model,
    device: str,
    flow_method: str = "farneback",
) -> dict:
    """Evaluate a single sample pair."""
    sample_id = item["sample_id"]

    # ── Optical Flow ──
    orig_gray = sample_video_frames_gray(item["orig_path"], num_frames)
    gen_gray = sample_video_frames_gray(item["gen_path"], num_frames)

    if flow_method == "farneback":
        flows_orig = compute_optical_flow_farneback(orig_gray)
        flows_gen = compute_optical_flow_farneback(gen_gray)
    else:
        # RAFT placeholder (requires torchvision)
        flows_orig = compute_optical_flow_farneback(orig_gray)
        flows_gen = compute_optical_flow_farneback(gen_gray)

    epe = compute_flow_epe(flows_orig, flows_gen)
    mag_orig = compute_flow_magnitude(flows_orig)
    mag_gen = compute_flow_magnitude(flows_gen)
    mag_ratio = mag_gen / (mag_orig + 1e-8)
    flow_dir_sim = compute_flow_direction_similarity(flows_orig, flows_gen)

    # ── CLIP / X-CLIP ──
    orig_frames = sample_video_frames(item["orig_path"], 8)
    gen_frames = sample_video_frames(item["gen_path"], 8)

    clip_orig = get_clip_video_feature(orig_frames, clip_processor, clip_model, device)
    clip_gen = get_clip_video_feature(gen_frames, clip_processor, clip_model, device)
    clip_sim = cosine_similarity(clip_orig, clip_gen)

    xclip_orig = get_xclip_video_feature(orig_frames, xclip_processor, xclip_model, device)
    xclip_gen = get_xclip_video_feature(gen_frames, xclip_processor, xclip_model, device)
    xclip_sim = cosine_similarity(xclip_orig, xclip_gen)

    return {
        "sample_id": sample_id,
        "optical_flow_epe": epe,
        "flow_magnitude_orig": mag_orig,
        "flow_magnitude_gen": mag_gen,
        "flow_magnitude_ratio": mag_ratio,
        "flow_direction_sim": flow_dir_sim,
        "clip_frame_sim": clip_sim,
        "xclip_motion_sim": xclip_sim,
    }


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def write_results(output_dir: Path, rows: list, summary: dict):
    """Save evaluation results in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "motion_fidelity_results.csv"
    fieldnames = [
        "sample_id", "optical_flow_epe", "flow_magnitude_orig",
        "flow_magnitude_gen", "flow_magnitude_ratio", "flow_direction_sim",
        "clip_frame_sim", "xclip_motion_sim",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # JSON summary
    json_path = output_dir / "motion_fidelity_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Markdown
    md_path = output_dir / "motion_fidelity_report.md"
    lines = [
        "# VMAD Motion Fidelity Evaluation",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- Optical Flow EPE (mean): {format_float(summary['epe_mean'])} (lower is better)",
        f"- Flow Direction Similarity (mean): {format_float(summary['flow_dir_sim_mean'])} (higher is better)",
        f"- Flow Magnitude Ratio (mean): {format_float(summary['mag_ratio_mean'])} (ideal = 1.0)",
        f"- CLIP Frame Similarity (mean): {format_float(summary['clip_sim_mean'])} (higher is better)",
        f"- X-CLIP Motion Similarity (mean): {format_float(summary['xclip_sim_mean'])} (higher is better)",
        "",
        "## Interpretation",
        "",
        "- EPE < 2.0: Good motion transfer",
        "- EPE < 1.0: Excellent motion transfer",
        "- Flow Dir Sim > 0.7: Motion directions well preserved",
        "- X-CLIP Sim > 0.8: Strong temporal motion consistency",
        "",
        "## Per-Sample Results",
        "",
        "| ID | EPE | Dir Sim | Mag Ratio | CLIP Sim | X-CLIP Sim |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {format_float(row['optical_flow_epe'])} | "
            f"{format_float(row['flow_direction_sim'])} | "
            f"{format_float(row['flow_magnitude_ratio'])} | "
            f"{format_float(row['clip_frame_sim'])} | "
            f"{format_float(row['xclip_motion_sim'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, json_path, md_path


def main():
    args = parse_args()

    # Find aligned samples
    items = list_eval_items(args.orig_dir, args.gen_dir, limit=args.limit)
    if not items:
        raise SystemExit(
            f"No aligned samples found.\n"
            f"  orig-dir: {args.orig_dir}\n"
            f"  gen-dir: {args.gen_dir}"
        )

    print(f"Found {len(items)} aligned samples for motion fidelity evaluation", flush=True)

    # Load models
    clip_processor, clip_model, xclip_processor, xclip_model = load_clip_models(
        args.device, args.clip_model, args.xclip_model
    )

    # Evaluate
    rows = []
    for idx, item in enumerate(items, 1):
        row = evaluate_sample(
            item, args.sample_frames,
            clip_processor, clip_model,
            xclip_processor, xclip_model,
            args.device, args.flow_method,
        )
        rows.append(row)

        if args.verbose or idx % 10 == 0:
            print(
                f"[{idx}/{len(items)}] {row['sample_id']} "
                f"EPE={row['optical_flow_epe']:.3f} "
                f"DirSim={row['flow_direction_sim']:.4f} "
                f"XCLIP={row['xclip_motion_sim']:.4f}",
                flush=True,
            )

    # Sort
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Compute summary
    def mean_of(key):
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "count": len(rows),
        "epe_mean": mean_of("optical_flow_epe"),
        "epe_std": float(np.std([r["optical_flow_epe"] for r in rows])),
        "flow_dir_sim_mean": mean_of("flow_direction_sim"),
        "mag_ratio_mean": mean_of("flow_magnitude_ratio"),
        "clip_sim_mean": mean_of("clip_frame_sim"),
        "xclip_sim_mean": mean_of("xclip_motion_sim"),
        "orig_dir": str(args.orig_dir),
        "gen_dir": str(args.gen_dir),
        "flow_method": args.flow_method,
        "sample_frames": args.sample_frames,
    }

    # Save
    csv_path, json_path, md_path = write_results(args.output_dir, rows, summary)

    # Print
    print(f"\nMotion Fidelity Evaluation Complete!", flush=True)
    print(f"  Samples:          {summary['count']}", flush=True)
    print(f"  EPE (mean):       {format_float(summary['epe_mean'])}", flush=True)
    print(f"  Dir Sim (mean):   {format_float(summary['flow_dir_sim_mean'])}", flush=True)
    print(f"  Mag Ratio (mean): {format_float(summary['mag_ratio_mean'])}", flush=True)
    print(f"  CLIP Sim (mean):  {format_float(summary['clip_sim_mean'])}", flush=True)
    print(f"  X-CLIP Sim (mean):{format_float(summary['xclip_sim_mean'])}", flush=True)
    print(f"  CSV:    {csv_path}", flush=True)
    print(f"  Report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
