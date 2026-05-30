#!/usr/bin/env python3
"""
Position Bias Validation Experiment for VMAD.

Validates the U-shape attention bias hypothesis that motivates VMAD's
position-aware velocity matching. This is the empirical evidence that
P-Flow does NOT have — it's a unique contribution of VMAD.

Experiments:
    1. Attention Weight Distribution — Measure cross-attention weights
       across text positions in the DiT model. Verify U-shape pattern.

    2. Position Ablation — Zero out Δe at specific positions and measure
       velocity field deviation. Quantify per-position influence.

    3. Position-Aware vs Uniform — Compare reproduction quality when using
       position-aware gradient scaling vs uniform gradients. Shows that
       concentrating optimization at high-influence positions improves
       convergence speed and final quality.

    4. Spectral Boundary Estimation — Empirically determine T_m* where
       motion and content sensitivities cross over.

Expected Results (supporting paper claims):
    - Position 0 receives 10-15× more attention weight than interior
    - Zeroing Δe[0] causes >50% of total velocity field deviation
    - Position-aware optimization reaches same quality in 60% fewer steps
    - T_m* ≈ 0.25-0.35 (consistent with spectral separation theory)

Usage:
    # Full experiment suite
    python scripts/run_position_bias_experiment.py \
        --video /path/to/reference.mp4 \
        --caption "a cat walking on the street" \
        --output-dir ./outputs/position_experiment

    # Quick validation (just attention measurement)
    python scripts/run_position_bias_experiment.py \
        --video /path/to/reference.mp4 \
        --caption "a cat walking on the street" \
        --output-dir ./outputs/position_experiment \
        --experiment attention_only

    # Full comparison: position-aware vs uniform
    python scripts/run_position_bias_experiment.py \
        --video /path/to/reference.mp4 \
        --caption "a cat walking on the street" \
        --output-dir ./outputs/position_experiment \
        --experiment convergence_comparison \
        --num-opt-steps 200

    # Batch experiment over multiple videos
    python scripts/run_position_bias_experiment.py \
        --video-dir /path/to/videos \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/position_batch \
        --experiment full \
        --limit 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="VMAD Position Bias Validation Experiment",
    )

    # Input (single video or batch)
    p.add_argument("--video", type=str, default=None,
                   help="Single video path")
    p.add_argument("--caption", type=str, default=None,
                   help="Caption for single video")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="Directory of videos (batch mode)")
    p.add_argument("--caption-dir", type=Path, default=None,
                   help="Directory of captions (batch mode)")

    # Output
    p.add_argument("--output-dir", type=Path, required=True)

    # Experiment selection
    p.add_argument("--experiment", type=str, default="full",
                   choices=["full", "attention_only", "ablation", "convergence_comparison", "spectral_boundary"],
                   help="Which experiment(s) to run")

    # Params
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--num-opt-steps", type=int, default=100,
                   help="Steps for convergence comparison")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def run_attention_measurement(pipe, z0, e0, device, output_dir):
    """
    Experiment 1: Measure cross-attention weight distribution.

    Hooks into DiT cross-attention layers and measures per-position attention.
    """
    logger.info("[Exp 1] Measuring cross-attention weight distribution...")

    from src.spectral_analysis import PositionBiasAnalyzer
    analyzer = PositionBiasAnalyzer(pipe, device=device)

    results = analyzer.measure_attention_distribution(z0, e0, timestep=0.15)
    results_mid = analyzer.measure_attention_distribution(z0, e0, timestep=0.5)
    results_late = analyzer.measure_attention_distribution(z0, e0, timestep=0.85)

    combined = {
        "early_t015": results,
        "mid_t050": results_mid,
        "late_t085": results_late,
        "summary": {
            "position_0_weight_early": results["position_0_weight"],
            "position_0_weight_mid": results_mid["position_0_weight"],
            "position_0_weight_late": results_late["position_0_weight"],
            "u_shape_ratio_early": results["u_shape_ratio"],
            "u_shape_ratio_mid": results_mid["u_shape_ratio"],
            "u_shape_ratio_late": results_late["u_shape_ratio"],
        }
    }

    out_path = output_dir / "attention_distribution.json"
    out_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    logger.info(f"  Position 0 weight (t=0.15): {results['position_0_weight']:.4f}")
    logger.info(f"  U-shape ratio (t=0.15): {results['u_shape_ratio']:.2f}x")
    logger.info(f"  Layers captured: {results['num_layers_captured']}")
    return combined


def run_position_ablation(pipe, z0, e0, delta_e, device, output_dir):
    """
    Experiment 2: Per-position ablation study.

    Zero out Δe at each position and measure velocity field deviation.
    """
    logger.info("[Exp 2] Position ablation analysis...")

    from src.velocity_matching import PositionAwareVelocityMatcher
    matcher = PositionAwareVelocityMatcher(pipe=pipe, device=device)

    results = matcher.analyze_position_influence(z0, e0, delta_e)

    out_path = output_dir / "position_ablation.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    logger.info(f"  Position 0 dominance: {results['position_0_dominance']:.2%}")
    logger.info(f"  U-shape ratio: {results['u_shape_ratio']:.2f}")
    logger.info(f"  Top-5 influential positions: "
                f"{sorted(range(len(results['raw_influences'])), key=lambda i: results['raw_influences'][i], reverse=True)[:5]}")
    return results


def run_convergence_comparison(pipe, z0, e0, eta_inv, device, output_dir,
                               num_steps=100):
    """
    Experiment 3: Position-aware vs uniform optimization convergence.

    Runs the same optimization twice:
    - With position-aware gradient scaling (VMAD's approach)
    - With uniform gradients (naive approach)
    Compares convergence speed and final loss.
    """
    logger.info("[Exp 3] Convergence comparison: position-aware vs uniform...")

    from src.velocity_matching import PositionAwareVelocityMatcher

    # Run with position-aware (VMAD)
    logger.info("  Running position-aware optimization...")
    matcher_pa = PositionAwareVelocityMatcher(
        pipe=pipe, T_m=1.0, num_opt_steps=num_steps,
        lr=1e-3, lambda_pos=0.01, position_aware=True, device=device,
    )
    result_pa = matcher_pa.optimize(z0=z0, e0=e0, eta_inv=eta_inv)

    # Run without position-aware (uniform)
    logger.info("  Running uniform optimization...")
    matcher_uniform = PositionAwareVelocityMatcher(
        pipe=pipe, T_m=1.0, num_opt_steps=num_steps,
        lr=1e-3, lambda_pos=0.0, position_aware=False, device=device,
    )
    result_uniform = matcher_uniform.optimize(z0=z0, e0=e0, eta_inv=eta_inv)

    # Compare
    comparison = {
        "position_aware": {
            "final_loss_vel": result_pa["final_loss_vel"],
            "delta_e_norm": result_pa["delta_e"].norm().item(),
            "loss_curve": [h["loss_vel"] for h in result_pa["loss_history"]],
            "position_energy": result_pa["position_energy_distribution"],
        },
        "uniform": {
            "final_loss_vel": result_uniform["final_loss_vel"],
            "delta_e_norm": result_uniform["delta_e"].norm().item(),
            "loss_curve": [h["loss_vel"] for h in result_uniform["loss_history"]],
            "position_energy": result_uniform["position_energy_distribution"],
        },
        "improvement": {
            "loss_reduction": (
                (result_uniform["final_loss_vel"] - result_pa["final_loss_vel"]) /
                (result_uniform["final_loss_vel"] + 1e-8)
            ),
        },
    }

    # Find step where position-aware reaches uniform's final loss
    uniform_final = result_uniform["final_loss_vel"]
    pa_curve = comparison["position_aware"]["loss_curve"]
    convergence_step = num_steps  # default: never reached
    for i, loss in enumerate(pa_curve):
        if loss <= uniform_final:
            convergence_step = i
            break
    comparison["improvement"]["convergence_speedup"] = num_steps / max(convergence_step, 1)
    comparison["improvement"]["steps_to_match_uniform"] = convergence_step

    out_path = output_dir / "convergence_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    logger.info(f"  Position-aware final loss: {result_pa['final_loss_vel']:.6f}")
    logger.info(f"  Uniform final loss: {result_uniform['final_loss_vel']:.6f}")
    logger.info(f"  Loss reduction: {comparison['improvement']['loss_reduction']:.1%}")
    logger.info(f"  Convergence speedup: {comparison['improvement']['convergence_speedup']:.1f}x")
    return comparison


def run_spectral_boundary(pipe, z0, e0, aug_embeddings, device, output_dir):
    """
    Experiment 4: Empirically estimate the spectral boundary T_m*.

    Measures where motion sensitivity drops below content sensitivity.
    """
    logger.info("[Exp 4] Spectral boundary estimation...")

    from src.spectral_analysis import SpectralBoundaryEstimator
    estimator = SpectralBoundaryEstimator(pipe, device=device, num_timesteps=20)

    result = estimator.estimate_boundary(
        z0=z0,
        embeddings_varied_content=aug_embeddings[:3] if aug_embeddings else [],
        embeddings_varied_motion=[e0],  # Placeholder; ideally use different Δe
    )

    out_path = output_dir / "spectral_boundary.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    logger.info(f"  Estimated T_m*: {result['T_m_star']:.3f}")
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # ── Load model ──
    from src.pipeline import VMADPipeline, VMADConfig
    from src.flow_matching import FlowMatchingInverter, encode_video_to_latents
    from src.video_utils import load_video, normalize_video

    config = VMADConfig()
    if args.model_path:
        config.t2v_path = args.model_path
    config.seed = args.seed

    pipeline = VMADPipeline(config)
    pipe = pipeline.pipe
    device = pipeline.device

    # ── Prepare input ──
    if args.video:
        videos = [(args.video, args.caption or "a video scene")]
    elif args.video_dir and args.caption_dir:
        videos = []
        for vp in sorted(args.video_dir.glob("*.mp4")):
            cp = args.caption_dir / f"{vp.stem}.txt"
            if cp.exists():
                videos.append((str(vp), cp.read_text(encoding="utf-8").strip()))
        if args.limit > 0:
            videos = videos[:args.limit]
    else:
        logger.error("Provide --video or --video-dir + --caption-dir")
        sys.exit(1)

    logger.info(f"Running experiment '{args.experiment}' on {len(videos)} video(s)")

    # ── Run experiment(s) per video ──
    all_results = []

    for vi, (video_path, caption) in enumerate(videos, 1):
        video_name = Path(video_path).stem
        video_out = args.output_dir / video_name
        video_out.mkdir(exist_ok=True)

        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{vi}/{len(videos)}] {video_name}: {caption[:50]}...")
        logger.info(f"{'─' * 50}")

        # Load and encode
        ref_video = load_video(video_path, num_frames=config.num_frames,
                               height=config.height, width=config.width, device=device)
        ref_norm = normalize_video(ref_video).unsqueeze(0)
        z0 = encode_video_to_latents(pipe, ref_norm, device)
        e0 = pipeline._encode_prompt(caption)

        # Inversion
        inverter = FlowMatchingInverter(pipe=pipe, num_inversion_steps=50, device=device)
        eta_inv = inverter.invert(z0, pipeline._encode_prompt(""))

        video_results = {"video": video_name, "caption": caption}

        # Experiment dispatch
        if args.experiment in ("full", "attention_only"):
            video_results["attention"] = run_attention_measurement(
                pipe, z0, e0, device, video_out
            )

        if args.experiment in ("full", "ablation", "convergence_comparison"):
            # Need a delta_e first
            from src.velocity_matching import PositionAwareVelocityMatcher
            logger.info("  [Prep] Quick Δe optimization (50 steps)...")
            matcher = PositionAwareVelocityMatcher(
                pipe=pipe, T_m=1.0, num_opt_steps=50, lr=1e-3,
                position_aware=True, device=device,
            )
            opt_result = matcher.optimize(z0=z0, e0=e0, eta_inv=eta_inv)
            delta_e = opt_result["delta_e"]

            if args.experiment in ("full", "ablation"):
                video_results["ablation"] = run_position_ablation(
                    pipe, z0, e0, delta_e, device, video_out
                )

        if args.experiment in ("full", "convergence_comparison"):
            video_results["convergence"] = run_convergence_comparison(
                pipe, z0, e0, eta_inv, device, video_out,
                num_steps=args.num_opt_steps,
            )

        if args.experiment in ("full", "spectral_boundary"):
            # Generate augmented embeddings for spectral boundary
            from src.content_augmentation import ContentAugmenter
            augmenter = ContentAugmenter(provider="mock", num_augmentations=3)
            aug_prompts = augmenter.augment(caption, n=3)
            aug_embeddings = [pipeline._encode_prompt(p) for p in aug_prompts]

            video_results["spectral"] = run_spectral_boundary(
                pipe, z0, e0, aug_embeddings, device, video_out
            )

        all_results.append(video_results)

        # Free memory
        del z0, e0, eta_inv, ref_video, ref_norm
        torch.cuda.empty_cache()

    # ── Aggregate results ──
    logger.info(f"\n{'=' * 60}")
    logger.info("Experiment Complete — Aggregate Results")
    logger.info(f"{'=' * 60}")

    if all_results and "attention" in all_results[0]:
        avg_ratio = sum(
            r["attention"]["summary"]["u_shape_ratio_early"]
            for r in all_results if "attention" in r
        ) / len(all_results)
        logger.info(f"  Mean U-shape ratio (t=0.15): {avg_ratio:.2f}x")

    if all_results and "ablation" in all_results[0]:
        avg_dominance = sum(
            r["ablation"]["position_0_dominance"]
            for r in all_results if "ablation" in r
        ) / len(all_results)
        logger.info(f"  Mean position-0 dominance: {avg_dominance:.2%}")

    if all_results and "convergence" in all_results[0]:
        avg_speedup = sum(
            r["convergence"]["improvement"]["convergence_speedup"]
            for r in all_results if "convergence" in r
        ) / len(all_results)
        logger.info(f"  Mean convergence speedup: {avg_speedup:.1f}x")

    # Save aggregate
    aggregate_path = args.output_dir / "experiment_results.json"
    aggregate_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"  Results: {aggregate_path}")


if __name__ == "__main__":
    main()
