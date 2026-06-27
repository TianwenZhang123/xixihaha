#!/usr/bin/env python3
"""
Flow Consistency Evaluation Script for Video Reproduction.

Computes optical flow-based metrics between original and generated videos:
    1. flow_epe       : End-Point Error between original and generated optical flow fields (↓)
    2. flow_cosine_sim: Cosine similarity of flattened flow fields (↑)
    3. dynamic_degree : Mean optical flow magnitude of generated video (↑)

Uses RAFT (torchvision built-in) for optical flow estimation.

This is the most direct metric for validating SVD Noise Prior (L2):
it measures whether the generated video preserves the original motion trajectory.

Usage:
    python evaluation/run_flow_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_flow

    # Custom resolution and frame count
    python evaluation/run_flow_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_flow \
        --resize-width 640 --resize-height 360 --sample-frames 16
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
DEFAULT_ORIG_DIR = Path("data/videos")
DEFAULT_GEN_DIR = Path("outputs/baseline_batch")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_results/flow")
DEFAULT_RAFT_WEIGHTS = "models/raft-large/raft_large.pth"


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

def load_video_frames_numpy(
    video_path: Path,
    num_frames: int = 16,
    resize_height: int = 0,
    resize_width: int = 0,
) -> list[np.ndarray]:
    """Load video frames as numpy arrays (H, W, 3) in RGB uint8.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to uniformly sample.
        resize_height: Resize height (0 = keep original).
        resize_width: Resize width (0 = keep original).

    Returns:
        List of (H, W, 3) uint8 numpy arrays.
    """
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
# Optical flow estimation
# ============================================================

def load_raft_model(device: str, weights_path: str = ""):
    """Load RAFT optical flow model from local weights (same pattern as CLIP/XCLIP).

    Args:
        device: Device string.
        weights_path: Path to raft_large.pth (pre-downloaded).
    """
    import torchvision

    raft = torchvision.models.optical_flow.raft_large(weights=None, progress=False)
    if weights_path and Path(weights_path).exists():
        raft.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        print(f"  Loaded RAFT weights from: {weights_path}", flush=True)
    else:
        # Fallback: try torchvision built-in weights
        raft = torchvision.models.optical_flow.raft_large(
            weights=torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
        )
        print(f"  Loaded RAFT from torchvision default weights", flush=True)
    raft = raft.to(device)
    raft.eval()
    return raft


@torch.inference_mode()
def estimate_flow(
    frames: list[np.ndarray],
    raft_model,
    device: str,
) -> list[np.ndarray]:
    """Estimate optical flow between consecutive frames using RAFT.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        raft_model: RAFT model.
        device: Device string.

    Returns:
        List of (H//4, W//4, 2) float32 flow arrays. Length = len(frames) - 1.
    """
    flows = []
    for i in range(len(frames) - 1):
        img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float() / 255.0

        img1 = img1.unsqueeze(0).to(device)  # (1, 3, H, W)
        img2 = img2.unsqueeze(0).to(device)

        # RAFT outputs flow at 1/4 resolution
        flow_list, _ = raft_model(img1, img2, iters=20, test_mode=True)
        flow = flow_list[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H//4, W//4, 2)
        flows.append(flow)

    return flows


# ============================================================
# Flow metrics
# ============================================================

def compute_flow_epe(flow_orig: np.ndarray, flow_gen: np.ndarray) -> float:
    """Compute End-Point Error between two flow fields.

    Args:
        flow_orig: (H, W, 2) flow field from original video.
        flow_gen: (H, W, 2) flow field from generated video.

    Returns:
        Mean EPE (lower is better).
    """
    # Resize to match if needed
    if flow_orig.shape != flow_gen.shape:
        h, w = min(flow_orig.shape[0], flow_gen.shape[0]), min(flow_orig.shape[1], flow_gen.shape[1])
        flow_orig = flow_orig[:h, :w, :]
        flow_gen = flow_gen[:h, :w, :]

    diff = flow_orig.astype(np.float32) - flow_gen.astype(np.float32)
    epe = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(epe))


def compute_flow_cosine_similarity(flow_orig: np.ndarray, flow_gen: np.ndarray) -> float:
    """Compute cosine similarity between flattened flow fields.

    Args:
        flow_orig: (H, W, 2) flow field from original video.
        flow_gen: (H, W, 2) flow field from generated video.

    Returns:
        Cosine similarity (higher is better).
    """
    if flow_orig.shape != flow_gen.shape:
        h, w = min(flow_orig.shape[0], flow_gen.shape[0]), min(flow_orig.shape[1], flow_gen.shape[1])
        flow_orig = flow_orig[:h, :w, :]
        flow_gen = flow_gen[:h, :w, :]

    a = flow_orig.flatten().astype(np.float32)
    b = flow_gen.flatten().astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_dynamic_degree(flows: list[np.ndarray]) -> float:
    """Compute mean optical flow magnitude (dynamic degree) of a video.

    Args:
        flows: List of (H, W, 2) flow fields.

    Returns:
        Mean flow magnitude (higher = more dynamic).
    """
    magnitudes = []
    for flow in flows:
        mag = np.sqrt(np.sum(flow.astype(np.float32) ** 2, axis=2))
        magnitudes.append(float(np.mean(mag)))
    return float(np.mean(magnitudes)) if magnitudes else 0.0


# ============================================================
# Argparse
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flow consistency evaluation: optical flow EPE between original and generated videos"
    )
    parser.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR,
                        help="Directory containing original reference videos ({id}.mp4)")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR,
                        help="Directory containing generated videos ({id}.mp4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument("--sample-frames", type=int, default=16,
                        help="Number of frames to sample per video")
    parser.add_argument("--resize-width", type=int, default=640,
                        help="Resize width before optical flow estimation (0 = keep original)")
    parser.add_argument("--resize-height", type=int, default=360,
                        help="Resize height before optical flow estimation (0 = keep original)")
    parser.add_argument("--raft-weights", type=str, default=DEFAULT_RAFT_WEIGHTS,
                        help="Path to RAFT large weights (default: models/raft-large/raft_large.pth)")
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
        "# Flow Consistency Evaluation Results",
        "",
        "## Metrics Description",
        "",
        "- `flow_epe`: Mean End-Point Error between original and generated optical flow fields. "
          "Directly measures motion trajectory fidelity. **Lower is better**.",
        "- `flow_cosine_sim`: Cosine similarity of flattened flow fields between original and generated video. "
          "Measures flow direction alignment. **Higher is better**.",
        "- `dynamic_degree`: Mean optical flow magnitude of the **generated** video. "
          "Measures how dynamic the generated video is. **Higher is better** (but should be comparable to original).",
        "",
        "## Summary",
        "",
        f"- Sample count: {summary['count']}",
        f"- flow_epe mean: {format_float(summary['flow_epe_mean'])}",
        f"- flow_cosine_sim mean: {format_float(summary['flow_cosine_sim_mean'])}",
        f"- dynamic_degree mean: {format_float(summary['dynamic_degree_mean'])}",
        "",
        "## Per-Sample Results",
        "",
        "| ID | flow_epe ↓ | flow_cosine_sim ↑ | dynamic_degree ↑ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | "
            f"{format_float(row['flow_epe'])} | "
            f"{format_float(row['flow_cosine_sim'])} | "
            f"{format_float(row['dynamic_degree'])} |"
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

    # Load RAFT
    print(f"Loading RAFT model on {args.device}...", flush=True)
    raft_model = load_raft_model(args.device, args.raft_weights)

    # Evaluate each sample
    rows = []
    for idx, item in enumerate(items, 1):
        sample_id = item["sample_id"]

        try:
            orig_frames = load_video_frames_numpy(
                item["orig_path"], args.sample_frames,
                args.resize_height, args.resize_width,
            )
            gen_frames = load_video_frames_numpy(
                item["gen_path"], args.sample_frames,
                args.resize_height, args.resize_width,
            )
        except RuntimeError as e:
            print(f"[{idx}/{len(items)}] {sample_id} SKIP: {e}", flush=True)
            continue

        # Estimate optical flow
        orig_flows = estimate_flow(orig_frames, raft_model, args.device)
        gen_flows = estimate_flow(gen_frames, raft_model, args.device)

        # Compute metrics
        epe_list = []
        cosine_list = []
        for fo, fg in zip(orig_flows, gen_flows):
            epe_list.append(compute_flow_epe(fo, fg))
            cosine_list.append(compute_flow_cosine_similarity(fo, fg))

        flow_epe = float(np.mean(epe_list)) if epe_list else 0.0
        flow_cosine_sim = float(np.mean(cosine_list)) if cosine_list else 0.0
        dynamic_degree = compute_dynamic_degree(gen_flows)

        row = {
            "sample_id": sample_id,
            "flow_epe": flow_epe,
            "flow_cosine_sim": flow_cosine_sim,
            "dynamic_degree": dynamic_degree,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(items)}] {sample_id} "
            f"epe={flow_epe:.4f} cos_sim={flow_cosine_sim:.4f} dyn={dynamic_degree:.4f}",
            flush=True,
        )

    # Sort by sample ID
    rows.sort(key=lambda x: int(x["sample_id"]))

    # Save CSV
    csv_path = args.output_dir / "flow_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "flow_epe", "flow_cosine_sim", "dynamic_degree"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Compute summary
    summary = {
        "count": len(rows),
        "flow_epe_mean": mean_of(rows, "flow_epe"),
        "flow_cosine_sim_mean": mean_of(rows, "flow_cosine_sim"),
        "dynamic_degree_mean": mean_of(rows, "dynamic_degree"),
    }

    # Save JSON
    json_path = args.output_dir / "flow_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save Markdown
    md_path = args.output_dir / "eval_summary.md"
    write_markdown_summary(md_path, rows, summary)

    print(f"\nFlow evaluation complete!", flush=True)
    print(f"  flow_epe mean: {format_float(summary['flow_epe_mean'])}", flush=True)
    print(f"  flow_cosine_sim mean: {format_float(summary['flow_cosine_sim_mean'])}", flush=True)
    print(f"  dynamic_degree mean: {format_float(summary['dynamic_degree_mean'])}", flush=True)
    print(f"  CSV: {csv_path}", flush=True)
    print(f"  Summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
