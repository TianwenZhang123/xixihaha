#!/usr/bin/env python3
"""
Run All New Evaluation Metrics.

Sequentially runs all 5 new evaluation metrics (FVD, DINO, Flow, LPIPS, Temporal)
and aggregates results into a unified summary.

Usage:
    python evaluation/run_all_metrics.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_all

    # Skip slow metrics (RAFT-based)
    python evaluation/run_all_metrics.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_all \
        --skip-flow --skip-temporal-raft

    # Quick test on 5 samples
    python evaluation/run_all_metrics.py \
        --orig-dir data/videos \
        --gen-dir outputs/pflow_200cases \
        --output-dir outputs/pflow_200cases/eval_all \
        --limit 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Eval scripts to run (relative to project root)
METRICS = [
    {
        "name": "FVD",
        "script": "evaluation/run_fvd_eval.py",
        "needs_orig": True,
        "output_subdir": "eval_fvd",
        "json_file": "fvd_results.json",
        "weight_arg": "--i3d-model",
        "weight_key": "r3d18_weights",
    },
    {
        "name": "DINO-Score",
        "script": "evaluation/run_dino_eval.py",
        "needs_orig": True,
        "output_subdir": "eval_dino",
        "json_file": "dino_results.json",
        "weight_arg": "--dinov2-model",
        "weight_key": "dinov2_model",
    },
    {
        "name": "Flow EPE",
        "script": "evaluation/run_flow_eval.py",
        "needs_orig": True,
        "output_subdir": "eval_flow",
        "json_file": "flow_results.json",
        "skip_flag": "skip_flow",
        "weight_arg": "--raft-weights",
        "weight_key": "raft_weights",
    },
    {
        "name": "LPIPS",
        "script": "evaluation/run_lpips_eval.py",
        "needs_orig": True,
        "output_subdir": "eval_lpips",
        "json_file": "lpips_results.json",
        "weight_arg": "--lpips-weights",
        "weight_key": "lpips_weights",
    },
    {
        "name": "Temporal",
        "script": "evaluation/run_temporal_eval.py",
        "needs_orig": False,
        "output_subdir": "eval_temporal",
        "json_file": "temporal_results.json",
        "skip_flag": "skip_temporal",
        "weight_arg": "--raft-weights",
        "weight_key": "raft_weights",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all new evaluation metrics and aggregate results"
    )
    parser.add_argument("--orig-dir", type=Path, default=Path("data/videos"),
                        help="Directory containing original reference videos")
    parser.add_argument("--gen-dir", type=Path, default=Path("outputs/baseline_batch"),
                        help="Directory containing generated videos")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_results/all"),
                        help="Root output directory (each metric gets a subdirectory)")
    parser.add_argument("--device", type=str, default="",
                        help="Device override (empty = auto)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N samples (0 = all)")
    parser.add_argument("--sample-frames", type=int, default=0,
                        help="Override number of frames to sample (0 = use each script's default)")
    parser.add_argument("--skip-flow", action="store_true",
                        help="Skip Flow EPE evaluation (RAFT is slow)")
    parser.add_argument("--skip-temporal", action="store_true",
                        help="Skip Temporal evaluation")
    parser.add_argument("--skip-fvd", action="store_true",
                        help="Skip FVD evaluation (I3D download may be slow)")
    # Model weight paths (local, same pattern as CLIP/XCLIP)
    parser.add_argument("--dinov2-model", type=str, default="models/dinov2-vitb14",
                        help="DINOv2 model local path")
    parser.add_argument("--r3d18-weights", type=str, default="models/r3d18-kinetics400/r3d18_kinetics400.pth",
                        help="R3D18 weights path for FVD")
    parser.add_argument("--raft-weights", type=str, default="models/raft-large/raft_large.pth",
                        help="RAFT weights path for Flow EPE / Dynamic Degree")
    parser.add_argument("--lpips-weights", type=str, default="models/lpips-vgg/vgg_lpips.pth",
                        help="LPIPS VGG weights path")
    return parser.parse_args()


def run_metric(metric: dict, args: argparse.Namespace, project_root: Path) -> dict | None:
    """Run a single evaluation metric and return its JSON results."""
    cmd = [sys.executable, str(project_root / metric["script"])]

    cmd.extend(["--orig-dir", str(args.orig_dir)])
    cmd.extend(["--gen-dir", str(args.gen_dir)])
    output_dir = args.output_dir / metric["output_subdir"]
    cmd.extend(["--output-dir", str(output_dir)])

    if args.device:
        cmd.extend(["--device", args.device])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.sample_frames > 0:
        cmd.extend(["--sample-frames", str(args.sample_frames)])

    # Pass model weight paths through to sub-scripts
    weight_arg = metric.get("weight_arg")
    weight_key = metric.get("weight_key")
    if weight_arg and weight_key and hasattr(args, weight_key):
        weight_path = getattr(args, weight_key)
        if weight_path:
            cmd.extend([weight_arg, str(weight_path)])

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Running: {metric['name']}", flush=True)
    print(f"  Script:  {metric['script']}", flush=True)
    print(f"  Output:  {output_dir}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            print(f"WARNING: {metric['name']} exited with code {result.returncode}", flush=True)
            return None
    except Exception as e:
        print(f"ERROR: {metric['name']} failed: {e}", flush=True)
        return None

    # Load JSON results
    json_path = output_dir / metric["json_file"]
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_aggregate_summary(output_path: Path, all_results: dict) -> None:
    """Write a unified markdown summary of all metrics."""
    lines = [
        "# P-Flow Evaluation Results (All Metrics)",
        "",
        "## Overview",
        "",
        "This document aggregates results from all evaluation metrics.",
        "",
        "## Metric Summary",
        "",
        "| Metric | Value | Direction | Description |",
        "| ------ | ----- | :-------: | ----------- |",
    ]

    # FVD
    if "fvd" in all_results:
        r = all_results["fvd"]
        lines.append(
            f"| FVD | {r.get('fvd', 'N/A')} | ↓ | "
            f"Distributional distance (N_real={r.get('n_real', '?')}, N_gen={r.get('n_gen', '?')}) |"
        )

    # DINO
    if "dino" in all_results:
        r = all_results["dino"]
        lines.append(
            f"| DINO Temporal | {r.get('dino_temporal_mean', 'N/A')} | ↑ | "
            f"Frame consistency (N={r.get('count', '?')}) |"
        )
        lines.append(
            f"| DINO Orig-Gen | {r.get('dino_orig_gen_mean', 'N/A')} | ↑ | "
            f"Visual fidelity (N={r.get('count', '?')}) |"
        )

    # Flow
    if "flow" in all_results:
        r = all_results["flow"]
        lines.append(
            f"| Flow EPE | {r.get('flow_epe_mean', 'N/A')} | ↓ | "
            f"Motion trajectory fidelity (N={r.get('count', '?')}) |"
        )
        lines.append(
            f"| Flow Cosine Sim | {r.get('flow_cosine_sim_mean', 'N/A')} | ↑ | "
            f"Flow direction alignment (N={r.get('count', '?')}) |"
        )
        lines.append(
            f"| Dynamic Degree (gen) | {r.get('dynamic_degree_mean', 'N/A')} | ↑ | "
            f"Generated video motion magnitude (N={r.get('count', '?')}) |"
        )

    # LPIPS
    if "lpips" in all_results:
        r = all_results["lpips"]
        lines.append(
            f"| LPIPS Avg | {r.get('lpips_avg_mean', 'N/A')} | ↓ | "
            f"Perceptual distance (N={r.get('count', '?')}) |"
        )

    # Temporal
    if "temporal" in all_results:
        r = all_results["temporal"]
        lines.append(
            f"| Temporal Flicker | {r.get('temporal_flicker_mean', 'N/A')} | ↓ | "
            f"Frame-to-frame flickering (N={r.get('count', '?')}) |"
        )

    lines.extend(["", "## Individual Metric Reports", ""])

    # Link to individual reports
    subdirs = ["eval_fvd", "eval_dino", "eval_flow", "eval_lpips", "eval_temporal"]
    for subdir in subdirs:
        md_path = output_path.parent / subdir / "eval_summary.md"
        if md_path.exists():
            lines.append(f"- [{subdir}]({subdir}/eval_summary.md)")
        else:
            lines.append(f"- {subdir} — not available")

    lines.extend([
        "",
        "## Notes",
        "",
        "- All metrics are computed **after** video generation (offline evaluation).",
        "- Per-sample results are available in each subdirectory's CSV file.",
        "- Report `mean +/- std` in the paper; use paired t-test for ablation significance.",
        "",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    args.orig_dir = args.orig_dir.resolve()
    args.gen_dir = args.gen_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}", flush=True)
    print(f"Original videos: {args.orig_dir}", flush=True)
    print(f"Generated videos: {args.gen_dir}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)

    all_results = {}
    skipped = []

    for metric in METRICS:
        # Check skip flags
        skip_flag = metric.get("skip_flag", "")
        if skip_flag == "skip_flow" and args.skip_flow:
            skipped.append(metric["name"])
            continue
        if skip_flag == "skip_temporal" and args.skip_temporal:
            skipped.append(metric["name"])
            continue
        if metric["name"] == "FVD" and args.skip_fvd:
            skipped.append(metric["name"])
            continue

        result = run_metric(metric, args, project_root)
        if result is not None:
            all_results[metric["name"].lower().replace("-", "_").replace(" ", "_")] = result

    # Save aggregated JSON
    agg_json = args.output_dir / "all_results.json"
    agg_json.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Write aggregate markdown summary
    agg_md = args.output_dir / "eval_summary.md"
    write_aggregate_summary(agg_md, all_results)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  All metrics complete!", flush=True)
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}", flush=True)
    print(f"  Aggregated JSON: {agg_json}", flush=True)
    print(f"  Aggregated summary: {agg_md}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
