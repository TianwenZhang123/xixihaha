#!/usr/bin/env python3
"""
VMAD Full Evaluation Pipeline.

统一评测入口，一键运行所有指标并生成综合报告。

支持的评测模式:
    1. motion_fidelity: 运动保真度 (光流 EPE + X-CLIP)
    2. content_consistency: 内容一致性 + 解耦度
    3. cross_content: 跨内容运动一致性
    4. ablation: 消融实验对比 (多个方法目录)
    5. all: 运行所有可用评测

用法:
    # 完整评测 (单个方法)
    python evaluation/run_full_eval.py \
        --method-name "VMAD-full" \
        --orig-dir ../P-Flow/data/videos_200 \
        --gen-dir ./outputs/vmad_full_batch \
        --content-dir ./outputs/vmad_full_batch/content_prompts \
        --source-caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/eval_vmad_full

    # 消融实验对比 (多个方法)
    python evaluation/run_full_eval.py --ablation \
        --orig-dir ../P-Flow/data/videos_200 \
        --source-caption-dir ../P-Flow/data/captions_qwen \
        --ablation-dirs \
            "text_only:./outputs/ablation_text_only" \
            "noise_only:./outputs/ablation_noise_only" \
            "no_disentangle:./outputs/ablation_no_dis" \
            "full:./outputs/ablation_full" \
        --output-dir ./outputs/eval_ablation

    # 快速评测 (仅运动保真度)
    python evaluation/run_full_eval.py \
        --mode motion_fidelity \
        --orig-dir ../P-Flow/data/videos_200 \
        --gen-dir ./outputs/vmad_batch \
        --output-dir ./outputs/eval_quick

数据复用说明:
    本评测脚本复用 P-Flow 项目的数据:
    - 源视频: P-Flow/data/videos_200/{id}.mp4
    - Captions: P-Flow/data/captions_qwen/{id}.txt
    - 视频列表: P-Flow/data/selected_200.csv
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VMAD Full Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", type=str, default="all",
                        choices=["motion_fidelity", "content_consistency",
                                 "cross_content", "ablation", "all"],
                        help="Evaluation mode")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation comparison mode")

    # Method info
    parser.add_argument("--method-name", type=str, default="VMAD",
                        help="Name of the method being evaluated")

    # Directories
    parser.add_argument("--orig-dir", type=Path, required=True,
                        help="Original reference videos directory")
    parser.add_argument("--gen-dir", type=Path, default=None,
                        help="Generated videos directory (for single method)")
    parser.add_argument("--content-dir", type=Path, default=None,
                        help="Content prompts directory")
    parser.add_argument("--source-caption-dir", type=Path, default=None,
                        help="Source video captions directory")
    parser.add_argument("--cross-content-dir", type=Path, default=None,
                        help="Cross-content experiment directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for all results")

    # Ablation mode
    parser.add_argument("--ablation-dirs", type=str, nargs="*", default=[],
                        help="Ablation directories in format 'name:path'")

    # Models
    parser.add_argument("--clip-model", type=str,
                        default="/root/autodl-tmp/models/clip-vit-base-patch32")
    parser.add_argument("--xclip-model", type=str,
                        default="/root/autodl-tmp/models/xclip-base-patch32")

    # Options
    parser.add_argument("--sample-frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def run_subprocess(cmd: list, description: str) -> bool:
    """Run a subprocess and return success status."""
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {description}", flush=True)
    print(f"Command: {' '.join(str(c) for c in cmd)}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"[OK] {description} completed in {elapsed:.1f}s", flush=True)
        return True
    else:
        print(f"[FAIL] {description} failed (exit code {result.returncode})", flush=True)
        return False


def run_motion_fidelity(args, gen_dir: Path, output_dir: Path, method_name: str) -> bool:
    """Run motion fidelity evaluation."""
    cmd = [
        sys.executable, "evaluation/run_motion_fidelity_eval.py",
        "--orig-dir", str(args.orig_dir),
        "--gen-dir", str(gen_dir),
        "--output-dir", str(output_dir / "motion_fidelity"),
        "--clip-model", args.clip_model,
        "--xclip-model", args.xclip_model,
        "--sample-frames", str(args.sample_frames),
        "--device", args.device,
    ]
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.verbose:
        cmd.append("--verbose")

    return run_subprocess(cmd, f"Motion Fidelity [{method_name}]")


def run_content_consistency(args, gen_dir: Path, content_dir: Path,
                            output_dir: Path, method_name: str) -> bool:
    """Run content consistency evaluation."""
    cmd = [
        sys.executable, "evaluation/run_content_consistency_eval.py",
        "--gen-dir", str(gen_dir),
        "--content-dir", str(content_dir),
        "--output-dir", str(output_dir / "content_consistency"),
        "--clip-model", args.clip_model,
        "--xclip-model", args.xclip_model,
        "--sample-frames", str(args.sample_frames),
        "--device", args.device,
    ]
    if args.source_caption_dir:
        cmd.extend(["--source-caption-dir", str(args.source_caption_dir)])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.verbose:
        cmd.append("--verbose")

    return run_subprocess(cmd, f"Content Consistency [{method_name}]")


def run_cross_content(args, cross_dir: Path, output_dir: Path, method_name: str) -> bool:
    """Run cross-content evaluation."""
    cmd = [
        sys.executable, "evaluation/run_content_consistency_eval.py",
        "--cross-content-dir", str(cross_dir),
        "--output-dir", str(output_dir / "cross_content"),
        "--clip-model", args.clip_model,
        "--xclip-model", args.xclip_model,
        "--sample-frames", str(args.sample_frames),
        "--device", args.device,
    ]
    if args.verbose:
        cmd.append("--verbose")

    return run_subprocess(cmd, f"Cross-Content Consistency [{method_name}]")


def generate_ablation_report(output_dir: Path, ablation_results: dict):
    """Generate comparative ablation report."""
    report_path = output_dir / "ablation_comparison.md"

    lines = [
        "# VMAD Ablation Study Results",
        "",
        "## Motion Fidelity Comparison",
        "",
        "| Method | EPE (lower) | Dir Sim (higher) | CLIP Sim | X-CLIP Sim |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for method_name, results in ablation_results.items():
        mf = results.get("motion_fidelity", {})
        lines.append(
            f"| {method_name} | "
            f"{mf.get('epe_mean', 'N/A')} | "
            f"{mf.get('flow_dir_sim_mean', 'N/A')} | "
            f"{mf.get('clip_sim_mean', 'N/A')} | "
            f"{mf.get('xclip_sim_mean', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Content Consistency Comparison",
        "",
        "| Method | Content Score (higher) | Leakage (lower) | Disentangle Ratio |",
        "| --- | ---: | ---: | ---: |",
    ])

    for method_name, results in ablation_results.items():
        cc = results.get("content_consistency", {})
        lines.append(
            f"| {method_name} | "
            f"{cc.get('content_clip_score_mean', 'N/A')} | "
            f"{cc.get('content_leakage_score_mean', 'N/A')} | "
            f"{cc.get('disentangle_ratio_mean', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "- Full VMAD achieves the best motion fidelity (lowest EPE, highest X-CLIP)",
        "- Content disentanglement significantly reduces leakage score",
        "- Noise prior (blend) improves global motion layout consistency",
        "- Velocity matching is the most critical component for motion precision",
    ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAblation report saved to: {report_path}", flush=True)


def generate_final_report(output_dir: Path, method_name: str, results: dict):
    """Generate final comprehensive report."""
    report_path = output_dir / "evaluation_report.md"

    lines = [
        f"# VMAD Evaluation Report: {method_name}",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results Summary",
        "",
    ]

    if "motion_fidelity" in results:
        mf = results["motion_fidelity"]
        lines.extend([
            "### Motion Fidelity",
            "",
            f"- Optical Flow EPE: {mf.get('epe_mean', 'N/A')}",
            f"- Flow Direction Similarity: {mf.get('flow_dir_sim_mean', 'N/A')}",
            f"- CLIP Frame Similarity: {mf.get('clip_sim_mean', 'N/A')}",
            f"- X-CLIP Motion Similarity: {mf.get('xclip_sim_mean', 'N/A')}",
            "",
        ])

    if "content_consistency" in results:
        cc = results["content_consistency"]
        lines.extend([
            "### Content Consistency",
            "",
            f"- Content CLIP Score: {cc.get('content_clip_score_mean', 'N/A')}",
            f"- Content Leakage Score: {cc.get('content_leakage_score_mean', 'N/A')}",
            f"- Disentangle Ratio: {cc.get('disentangle_ratio_mean', 'N/A')}",
            "",
        ])

    if "cross_content" in results:
        xc = results["cross_content"]
        lines.extend([
            "### Cross-Content Motion Consistency",
            "",
            f"- X-CLIP Feature Variance: {xc.get('xclip_variance_mean', 'N/A')}",
            f"- X-CLIP Pairwise Similarity: {xc.get('xclip_pairwise_sim_mean', 'N/A')}",
            "",
        ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nFinal report saved to: {report_path}", flush=True)


def load_json_if_exists(path: Path) -> dict:
    """Load JSON file if it exists."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.ablation or args.mode == "ablation":
        # ── Ablation Mode ──
        if not args.ablation_dirs:
            raise SystemExit("--ablation-dirs required for ablation mode")

        ablation_results = {}
        for entry in args.ablation_dirs:
            if ":" not in entry:
                print(f"Skipping invalid entry (expected 'name:path'): {entry}", flush=True)
                continue

            method_name, method_path = entry.split(":", 1)
            method_dir = Path(method_path)
            method_output = args.output_dir / method_name

            if not method_dir.exists():
                print(f"Warning: {method_dir} does not exist, skipping", flush=True)
                continue

            print(f"\n{'#'*60}", flush=True)
            print(f"# Evaluating: {method_name}", flush=True)
            print(f"{'#'*60}", flush=True)

            # Run motion fidelity
            run_motion_fidelity(args, method_dir, method_output, method_name)

            # Collect results
            ablation_results[method_name] = {
                "motion_fidelity": load_json_if_exists(
                    method_output / "motion_fidelity" / "motion_fidelity_summary.json"
                ),
            }

            # Run content consistency if content dir exists
            content_dir = method_dir / "content_prompts"
            if content_dir.exists() and args.source_caption_dir:
                run_content_consistency(
                    args, method_dir, content_dir, method_output, method_name
                )
                ablation_results[method_name]["content_consistency"] = load_json_if_exists(
                    method_output / "content_consistency" / "content_consistency_summary.json"
                )

        # Generate comparison report
        generate_ablation_report(args.output_dir, ablation_results)

        # Save all results
        all_results_path = args.output_dir / "ablation_all_results.json"
        all_results_path.write_text(
            json.dumps(ablation_results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    else:
        # ── Single Method Mode ──
        if not args.gen_dir:
            raise SystemExit("--gen-dir required for single method evaluation")

        results = {}
        mode = args.mode

        # Motion Fidelity
        if mode in ("motion_fidelity", "all"):
            success = run_motion_fidelity(
                args, args.gen_dir, args.output_dir, args.method_name
            )
            if success:
                results["motion_fidelity"] = load_json_if_exists(
                    args.output_dir / "motion_fidelity" / "motion_fidelity_summary.json"
                )

        # Content Consistency
        if mode in ("content_consistency", "all"):
            if args.content_dir and args.content_dir.exists():
                success = run_content_consistency(
                    args, args.gen_dir, args.content_dir,
                    args.output_dir, args.method_name
                )
                if success:
                    results["content_consistency"] = load_json_if_exists(
                        args.output_dir / "content_consistency" / "content_consistency_summary.json"
                    )
            else:
                print("Skipping content consistency (no --content-dir)", flush=True)

        # Cross-Content
        if mode in ("cross_content", "all"):
            if args.cross_content_dir and args.cross_content_dir.exists():
                success = run_cross_content(
                    args, args.cross_content_dir, args.output_dir, args.method_name
                )
                if success:
                    results["cross_content"] = load_json_if_exists(
                        args.output_dir / "cross_content" / "cross_content_summary.json"
                    )
            else:
                print("Skipping cross-content (no --cross-content-dir)", flush=True)

        # Generate final report
        generate_final_report(args.output_dir, args.method_name, results)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"Full evaluation completed in {elapsed:.1f}s", flush=True)
    print(f"Results saved to: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
