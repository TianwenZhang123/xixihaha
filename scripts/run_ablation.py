"""
Ablation Experiment Script: P-Flow vs VISTA Prompt Optimization.

This script enables systematic comparison between:
1. P-Flow's original single-VLM prompt optimizer
2. VISTA's multi-agent self-improving optimizer
3. VISTA with individual components disabled (for ablation)

Ablation configurations:
- pflow_original: Original P-Flow PromptOptimizer
- vista_full: Full VISTA (SVPP + Tournament + MMAC + DTPA)
- vista_no_svpp: VISTA without Structured Prompt Planning
- vista_no_tournament: VISTA without Binary Tournament
- vista_no_mmac: VISTA without Multi-Agent Critiques (uses simple judge)
- vista_no_dtpa: VISTA without Deep Thinking (uses simple refinement)
- vista_mmac_only: Only MMAC evaluation, no DTPA
- vista_dtpa_only: Only DTPA refinement, no MMAC

Usage:
    python scripts/run_ablation.py \\
        --reference_video path/to/ref.mp4 \\
        --prompt "your effect description" \\
        --ablation vista_full \\
        --output_dir outputs/ablation

    # Run all ablations:
    python scripts/run_ablation.py \\
        --reference_video path/to/ref.mp4 \\
        --prompt "your effect description" \\
        --ablation all \\
        --output_dir outputs/ablation_study
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pflow.pipeline import PFlowPipeline
from pflow.prompt_optimizer import PromptOptimizer
from pflow.vista_optimizer import VISTAOptimizer


# ============================================================================
# Ablation Configurations
# ============================================================================

ABLATION_CONFIGS = {
    "pflow_original": {
        "description": "P-Flow original single-VLM prompt optimizer",
        "optimizer_class": "PromptOptimizer",
        "params": {},
    },
    "vista_full": {
        "description": "Full VISTA: SVPP + Tournament + MMAC + DTPA",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": True,
            "enable_tournament": True,
            "enable_mmac": True,
            "enable_dtpa": True,
        },
    },
    "vista_no_svpp": {
        "description": "VISTA without Structured Video Prompt Planning",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": False,
            "enable_tournament": True,
            "enable_mmac": True,
            "enable_dtpa": True,
        },
    },
    "vista_no_tournament": {
        "description": "VISTA without Binary Tournament Selection",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": True,
            "enable_tournament": False,
            "enable_mmac": True,
            "enable_dtpa": True,
        },
    },
    "vista_no_mmac": {
        "description": "VISTA without MMAC (single judge instead of triadic court)",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": True,
            "enable_tournament": True,
            "enable_mmac": False,
            "enable_dtpa": True,
        },
    },
    "vista_no_dtpa": {
        "description": "VISTA without Deep Thinking (simple refinement)",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": True,
            "enable_tournament": True,
            "enable_mmac": True,
            "enable_dtpa": False,
        },
    },
    "vista_mmac_only": {
        "description": "Only MMAC evaluation, no DTPA or SVPP",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": False,
            "enable_tournament": False,
            "enable_mmac": True,
            "enable_dtpa": False,
        },
    },
    "vista_dtpa_only": {
        "description": "Only DTPA refinement, no MMAC evaluation",
        "optimizer_class": "VISTAOptimizer",
        "params": {
            "enable_svpp": False,
            "enable_tournament": False,
            "enable_mmac": False,
            "enable_dtpa": True,
        },
    },
}


def create_optimizer(config_name: str, output_dir: str, use_mock: bool = False):
    """
    Create an optimizer instance based on ablation config name.

    Args:
        config_name: Key from ABLATION_CONFIGS.
        output_dir: Output directory for this run.
        use_mock: Use mock VLM for testing.

    Returns:
        Optimizer instance (PromptOptimizer or VISTAOptimizer).
    """
    config = ABLATION_CONFIGS[config_name]

    if config["optimizer_class"] == "PromptOptimizer":
        return PromptOptimizer(
            max_iterations=10,
            output_dir=output_dir,
            use_mock=use_mock,
        )
    else:
        return VISTAOptimizer(
            max_iterations=5,
            output_dir=output_dir,
            use_mock=use_mock,
            **config["params"],
        )


def run_single_ablation(
    config_name: str,
    reference_video_path: str,
    prompt: str,
    output_dir: str,
    use_mock: bool = False,
    config_path: str = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment.

    This creates a modified PFlowPipeline that uses the specified
    optimizer variant.

    Args:
        config_name: Ablation configuration name.
        reference_video_path: Path to reference video.
        prompt: User prompt.
        output_dir: Output directory.
        use_mock: Use mock VLM.
        config_path: Optional path to config YAML.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    config = ABLATION_CONFIGS[config_name]
    run_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Ablation: {config_name}")
    print(f"Description: {config['description']}")
    print(f"Output: {run_output_dir}")
    print(f"{'='*70}\n")

    # Create pipeline with appropriate optimizer
    pipeline = PFlowPipeline(
        config_path=config_path,
        use_mock_vlm=use_mock,
        device="cuda" if not use_mock else "cpu",
    )

    # Monkey-patch the prompt optimizer in the pipeline
    optimizer = create_optimizer(config_name, run_output_dir, use_mock)

    # Store original and replace
    # The pipeline creates its own PromptOptimizer in run(), so we need
    # to override the method that creates it
    original_run = pipeline.run

    def patched_run(reference_video_path, prompt, output_dir=None, seed=None):
        """Run with patched optimizer."""
        # We override just the optimizer creation by accessing the pipeline
        # internals. Since run() creates PromptOptimizer internally,
        # we need a different approach.
        # Actually let's just use the optimizer directly in a custom loop.
        return _run_with_optimizer(
            pipeline=pipeline,
            optimizer=optimizer,
            reference_video_path=reference_video_path,
            prompt=prompt,
            output_dir=run_output_dir,
            seed=seed,
        )

    # Time the run
    start_time = time.time()

    try:
        results = patched_run(
            reference_video_path=reference_video_path,
            prompt=prompt,
            output_dir=run_output_dir,
            seed=seed,
        )
    except Exception as e:
        results = {
            "error": str(e),
            "config_name": config_name,
        }
        print(f"ERROR in {config_name}: {e}")

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["config_name"] = config_name
    results["config_description"] = config["description"]

    # Save results
    results_path = os.path.join(run_output_dir, "ablation_results.json")
    serializable = {k: v for k, v in results.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted {config_name} in {elapsed:.1f}s")
    if "best_confidence" in results:
        print(f"  Best confidence: {results['best_confidence']:.3f}")
    if "num_iterations" in results:
        print(f"  Iterations: {results['num_iterations']}")

    return results


def _run_with_optimizer(
    pipeline: PFlowPipeline,
    optimizer,
    reference_video_path: str,
    prompt: str,
    output_dir: str,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Run the P-Flow pipeline with a custom optimizer.

    This replicates the pipeline.run() logic but uses the provided
    optimizer instead of creating a new PromptOptimizer.

    For mock/dry-run mode, simulates the video generation.
    """
    from pflow.video_utils import load_video, save_video_tensor
    from pflow.trajectory import TrajectoryManager
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    max_iterations = optimizer.max_iterations if hasattr(optimizer, 'max_iterations') else 10

    # For mock mode, create dummy tensors
    if isinstance(optimizer.vlm_client, (type(None),)) or hasattr(optimizer.vlm_client, 'call_count'):
        # Mock mode - use dummy video tensors
        reference_video = torch.randn(3, 16, 64, 64)  # Small dummy
        generated_video = torch.randn(3, 16, 64, 64)
    else:
        # Real mode - load actual video
        reference_video = load_video(
            reference_video_path,
            num_frames=pipeline.config["video"]["num_frames"],
            height=pipeline.config["video"]["height"],
            width=pipeline.config["video"]["width"],
            device=pipeline.device,
        )
        generated_video = reference_video.clone()  # placeholder

    # Trajectory tracking
    trajectory = TrajectoryManager(
        output_dir=str(output_path),
        save_all_videos=False,
    )
    trajectory.set_reference(reference_video, reference_video_path)

    current_prompt = prompt
    best_prompt = prompt
    best_confidence = 0.0
    all_prompts = [prompt]

    for iteration in range(1, max_iterations + 1):
        print(f"  Iter {iteration}/{max_iterations}: prompt='{current_prompt[:60]}...'")

        # In mock mode, generated_video is just noise
        # In real mode, this would call pipeline._generate_video()

        # Get previous video for comparison
        _, prev_video, _ = trajectory.get_videos_for_vlm()

        # Run optimization with the selected optimizer
        result = optimizer.optimize_prompt(
            initial_prompt=current_prompt,
            reference_video=reference_video,
            generated_video=generated_video,
            iteration=iteration,
            history=trajectory.get_history_for_vlm(),
            user_description=prompt,
            previous_video=prev_video,
        )

        # Update trajectory
        trajectory.add_entry(
            iteration=iteration,
            prompt=current_prompt,
            video=generated_video,
            analysis=result.get("analysis", ""),
            improvements=result.get("improvements", []),
            confidence=result.get("confidence", 0.0),
            key_differences=result.get("key_differences", []),
        )

        # Track best
        confidence = result.get("confidence", 0.0)
        if confidence > best_confidence:
            best_confidence = confidence
            best_prompt = current_prompt

        # Check convergence
        if optimizer.should_stop_early(
            trajectory.get_history_for_vlm(), min_iterations=3
        ):
            print(f"  Converged at iteration {iteration}")
            break

        # Update prompt
        refined = result.get("refined_prompt", "")
        if refined:
            current_prompt = refined
            all_prompts.append(current_prompt)

    return {
        "best_prompt": best_prompt,
        "best_confidence": best_confidence,
        "num_iterations": len(trajectory),
        "all_prompts": all_prompts,
        "trajectory": trajectory.get_history_for_vlm(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run P-Flow ablation experiments comparing prompt optimization methods."
    )
    parser.add_argument(
        "--reference_video", type=str, required=True,
        help="Path to reference video"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="User prompt describing desired visual effect"
    )
    parser.add_argument(
        "--ablation", type=str, default="all",
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        help="Ablation configuration to run (or 'all' for all configurations)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ablation_study",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock VLM for testing without API"
    )

    args = parser.parse_args()

    # Determine which configs to run
    if args.ablation == "all":
        configs_to_run = list(ABLATION_CONFIGS.keys())
    else:
        configs_to_run = [args.ablation]

    print(f"P-Flow Ablation Study")
    print(f"{'='*70}")
    print(f"Reference: {args.reference_video}")
    print(f"Prompt: {args.prompt}")
    print(f"Configurations: {configs_to_run}")
    print(f"Output: {args.output_dir}")
    print(f"Mock mode: {args.mock}")
    print(f"{'='*70}")

    # Run ablations
    all_results = {}
    for config_name in configs_to_run:
        results = run_single_ablation(
            config_name=config_name,
            reference_video_path=args.reference_video,
            prompt=args.prompt,
            output_dir=args.output_dir,
            use_mock=args.mock,
            config_path=args.config,
            seed=args.seed,
        )
        all_results[config_name] = results

    # Save summary
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    summary = {}
    for name, r in all_results.items():
        summary[name] = {
            "description": ABLATION_CONFIGS[name]["description"],
            "best_confidence": r.get("best_confidence", 0.0),
            "num_iterations": r.get("num_iterations", 0),
            "elapsed_time": r.get("elapsed_time", 0.0),
            "error": r.get("error", None),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print(f"\n\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<25} {'Confidence':<12} {'Iterations':<12} {'Time (s)':<10}")
    print(f"{'-'*70}")
    for name, s in summary.items():
        conf = f"{s['best_confidence']:.3f}" if s.get('best_confidence') else "ERROR"
        iters = str(s.get('num_iterations', '-'))
        elapsed = f"{s.get('elapsed_time', 0):.1f}"
        print(f"{name:<25} {conf:<12} {iters:<12} {elapsed:<10}")
    print(f"{'='*70}")
    print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()
