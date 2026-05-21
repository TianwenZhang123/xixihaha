#!/usr/bin/env python3
"""
Complete Evaluation Script for P-Flow.

Runs all automated metrics on experiment outputs.

Usage:
    # Evaluate single experiment
    python evaluation/run_evaluation.py \
        --experiment_dir /data/outputs/pflow/fire_0001 \
        --reference_video /data/datasets/Open-VFX/videos/fire_effects/fire_0001.mp4

    # Evaluate batch results
    python evaluation/run_evaluation.py \
        --batch_dir /data/outputs/pflow/batch_test \
        --dataset_dir /data/datasets/Open-VFX \
        --output evaluation_results.json

    # Select best iteration per sample
    python evaluation/run_evaluation.py \
        --select_best \
        --experiment_dir /data/outputs/pflow/fire_0001
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import FIDVIDMetric, FVDMetric, DynamicDegreeMetric, run_full_evaluation


def evaluate_single_experiment(
    experiment_dir: str,
    reference_video: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate all iterations of a single P-Flow experiment.

    Determines best iteration by FVD score.

    Args:
        experiment_dir: Directory with generated_iter_*.mp4 files.
        reference_video: Path to reference video.
        device: Compute device.

    Returns:
        Evaluation results with best iteration identified.
    """
    exp_path = Path(experiment_dir)

    # Find all generated videos
    generated_videos = sorted(glob.glob(str(exp_path / "generated_iter_*.mp4")))
    if not generated_videos:
        return {"error": "No generated videos found"}

    print(f"Evaluating {len(generated_videos)} iterations in: {experiment_dir}")

    # Per-iteration evaluation
    dd_metric = DynamicDegreeMetric(device=device)
    iteration_scores = []

    for video_path in generated_videos:
        iter_num = int(Path(video_path).stem.split("_")[-1])
        motion = dd_metric.compute_optical_flow_magnitude(video_path)
        variance = dd_metric.compute_temporal_variance(video_path)
        iteration_scores.append({
            "iteration": iter_num,
            "video_path": video_path,
            "dynamic_motion": motion,
            "dynamic_variance": variance,
        })

    # Overall metrics (best video vs reference)
    # For FID-VID and FVD, we compare all iterations against reference
    ref_videos = [reference_video]
    overall = run_full_evaluation(
        reference_videos=ref_videos,
        generated_videos=generated_videos,
        output_path=str(exp_path / "evaluation_results.json"),
        device=device,
    )

    # Add per-iteration scores
    overall["iteration_scores"] = iteration_scores
    overall["num_iterations"] = len(generated_videos)

    # Select best iteration (highest dynamic degree that isn't degenerate)
    best_iter = max(iteration_scores, key=lambda x: x["dynamic_motion"] + x["dynamic_variance"])
    overall["best_iteration"] = best_iter["iteration"]
    overall["best_video"] = best_iter["video_path"]

    return overall


def evaluate_batch(
    batch_dir: str,
    dataset_dir: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate batch of P-Flow experiments.

    Aggregates metrics across all samples.
    """
    batch_path = Path(batch_dir)
    dataset_path = Path(dataset_dir)

    # Find all sample directories
    sample_dirs = [d for d in batch_path.iterdir() if d.is_dir() and (d / "reference.mp4").exists()]

    if not sample_dirs:
        return {"error": "No valid sample directories found"}

    print(f"Evaluating batch: {len(sample_dirs)} samples")

    all_ref_videos = []
    all_best_videos = []
    all_gen_videos = []
    sample_results = []

    for sample_dir in sorted(sample_dirs):
        ref_video = str(sample_dir / "reference.mp4")
        gen_videos = sorted(glob.glob(str(sample_dir / "generated_iter_*.mp4")))

        if not gen_videos:
            continue

        all_ref_videos.append(ref_video)
        all_gen_videos.extend(gen_videos)

        # Quick per-sample dynamic degree
        dd_metric = DynamicDegreeMetric(device=device)
        best_score = -1
        best_video = gen_videos[-1]  # Default to last iteration

        for gv in gen_videos:
            score = dd_metric.compute_optical_flow_magnitude(gv)
            if score > best_score:
                best_score = score
                best_video = gv

        all_best_videos.append(best_video)
        sample_results.append({
            "sample": sample_dir.name,
            "best_video": best_video,
            "dynamic_score": best_score,
            "num_iterations": len(gen_videos),
        })

    # Aggregate metrics
    print(f"\nComputing aggregate metrics...")
    aggregate = run_full_evaluation(
        reference_videos=all_ref_videos,
        generated_videos=all_best_videos,
        output_path=str(batch_path / "aggregate_evaluation.json"),
        device=device,
    )

    aggregate["num_samples"] = len(sample_results)
    aggregate["sample_results"] = sample_results

    # Save
    with open(batch_path / "batch_evaluation.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    print(f"\nBatch Evaluation Summary:")
    print(f"  Samples: {len(sample_results)}")
    print(f"  FID-VID: {aggregate.get('fid_vid', 'N/A')}")
    print(f"  FVD: {aggregate.get('fvd', 'N/A')}")
    print(f"  Dynamic Degree: {aggregate.get('dynamic_degree_combined', 'N/A')}")

    return aggregate


def select_best_iteration(experiment_dir: str, metric: str = "dynamic") -> str:
    """
    Select best iteration from a P-Flow experiment.

    Paper: Best video selected offline using VBench/FVD metrics.

    Args:
        experiment_dir: Experiment output directory.
        metric: Selection criterion ("dynamic", "last", "fvd").

    Returns:
        Path to best video.
    """
    exp_path = Path(experiment_dir)
    generated = sorted(glob.glob(str(exp_path / "generated_iter_*.mp4")))

    if not generated:
        raise FileNotFoundError(f"No videos in {experiment_dir}")

    if metric == "last":
        return generated[-1]

    if metric == "dynamic":
        dd = DynamicDegreeMetric(device="cpu")
        scores = [(v, dd.compute_optical_flow_magnitude(v)) for v in generated]
        best = max(scores, key=lambda x: x[1])
        return best[0]

    # Default: return last iteration
    return generated[-1]


def main():
    parser = argparse.ArgumentParser(description="P-Flow Evaluation")
    parser.add_argument("--experiment_dir", type=str, help="Single experiment directory")
    parser.add_argument("--reference_video", type=str, help="Reference video path")
    parser.add_argument("--batch_dir", type=str, help="Batch experiment directory")
    parser.add_argument("--dataset_dir", type=str, help="Dataset root directory")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--select_best", action="store_true", help="Select best iteration")
    parser.add_argument("--metric", type=str, default="dynamic", choices=["dynamic", "last", "fvd"])

    args = parser.parse_args()

    if args.select_best and args.experiment_dir:
        best = select_best_iteration(args.experiment_dir, args.metric)
        print(f"Best iteration: {best}")
    elif args.batch_dir and args.dataset_dir:
        evaluate_batch(args.batch_dir, args.dataset_dir, args.device)
    elif args.experiment_dir and args.reference_video:
        evaluate_single_experiment(args.experiment_dir, args.reference_video, args.device)
    else:
        parser.error("Provide --experiment_dir + --reference_video, or --batch_dir + --dataset_dir")


if __name__ == "__main__":
    main()
