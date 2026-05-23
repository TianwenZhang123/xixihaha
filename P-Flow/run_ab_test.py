#!/usr/bin/env python3
"""
A/B Test: Compare V1 (original) vs V2 (optimized) prompt strategy.

Runs the same reference video through both prompt strategies,
then evaluates which produces better reproduction quality.

Usage:
    # Run A/B test on a single video
    python run_ab_test.py --video /path/to/reference.mp4 \
        --alpha 0.001 --i_max 5 --seed 42

    # Evaluate existing results (skip generation)
    python run_ab_test.py --eval_only \
        --dir_v1 /path/to/v1_output \
        --dir_v2 /path/to/v2_output
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import PFlowPipeline
from src.vlm_client import create_vlm_client
from src.vlm_client_v2 import create_vlm_client_v2


def run_with_vlm_version(
    video_path: str,
    vlm_version: str,
    output_dir: str,
    config_path: str = None,
    alpha: float = 0.001,
    i_max: int = 5,
    seed: int = 42,
    hint: str = "",
) -> dict:
    """
    Run pipeline with specified VLM version.
    
    Args:
        vlm_version: "v1" or "v2"
        
    Returns:
        Pipeline result dict
    """
    config_override = {"optimization": {"i_max": i_max}}

    pipeline = PFlowPipeline(
        config_path=config_path,
        config=config_override,
    )

    # Override VLM client based on version
    if vlm_version == "v2":
        pipeline._vlm_client = create_vlm_client_v2(pipeline.config["vlm"])
        # Use V2's generate_initial_prompt
        initial_prompt = pipeline._vlm_client.generate_initial_prompt(
            reference_video_path=video_path,
            user_hint=hint,
        )
    else:
        pipeline._vlm_client = create_vlm_client(pipeline.config["vlm"])
        initial_prompt = pipeline.generate_initial_prompt(
            reference_video_path=video_path,
            user_hint=hint,
        )

    print(f"\n{'='*60}")
    print(f"Running {vlm_version.upper()} strategy")
    print(f"Initial prompt ({len(initial_prompt.split())} words): {initial_prompt[:100]}...")
    print(f"{'='*60}\n")

    result = pipeline.run(
        reference_video_path=video_path,
        prompt=initial_prompt,
        output_dir=output_dir,
        seed=seed,
        video_description=hint,
        alpha_override=alpha,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="A/B Test: V1 vs V2 Prompt Strategy")
    parser.add_argument("--video", type=str, help="Reference video path")
    parser.add_argument("--alpha", type=float, default=0.001, help="Noise prior alpha")
    parser.add_argument("--i_max", type=int, default=5, help="Iterations per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hint", type=str, default="", help="Hint for initial prompt generation")
    parser.add_argument("--output_base", type=str, default="/root/autodl-tmp/outputs/ab_test",
                       help="Base output directory")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing results")
    parser.add_argument("--dir_v1", type=str, help="V1 output dir (for eval_only)")
    parser.add_argument("--dir_v2", type=str, help="V2 output dir (for eval_only)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.eval_only:
        # Evaluate existing results
        if not args.dir_v1 or not args.dir_v2:
            parser.error("--eval_only requires --dir_v1 and --dir_v2")
        
        from evaluation.eval_reproduction import compare_experiments
        compare_experiments(
            args.dir_v1, args.dir_v2,
            device=args.device,
            labels=("V1-original", "V2-optimized"),
        )
        return

    if not args.video:
        parser.error("--video is required (or use --eval_only)")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_base) / timestamp
    dir_v1 = str(base_dir / "v1_original")
    dir_v2 = str(base_dir / "v2_optimized")

    # Default config
    config_path = args.config
    if config_path is None:
        default_config = Path(__file__).parent / "configs" / "paper_default.yaml"
        if default_config.exists():
            config_path = str(default_config)

    # Run V1
    print("\n" + "=" * 80)
    print("PHASE 1: Running V1 (Original Prompt Strategy)")
    print("=" * 80)
    result_v1 = run_with_vlm_version(
        video_path=args.video,
        vlm_version="v1",
        output_dir=dir_v1,
        config_path=config_path,
        alpha=args.alpha,
        i_max=args.i_max,
        seed=args.seed,
        hint=args.hint,
    )

    # Run V2
    print("\n" + "=" * 80)
    print("PHASE 2: Running V2 (Optimized Prompt Strategy)")
    print("=" * 80)
    result_v2 = run_with_vlm_version(
        video_path=args.video,
        vlm_version="v2",
        output_dir=dir_v2,
        config_path=config_path,
        alpha=args.alpha,
        i_max=args.i_max,
        seed=args.seed,
        hint=args.hint,
    )

    # Evaluate both
    print("\n" + "=" * 80)
    print("PHASE 3: Evaluating Results")
    print("=" * 80)
    
    from evaluation.eval_reproduction import compare_experiments
    comparison = compare_experiments(
        dir_v1, dir_v2,
        device=args.device,
        labels=("V1-original", "V2-optimized"),
    )

    # Save comparison summary
    summary = {
        "timestamp": timestamp,
        "reference_video": args.video,
        "alpha": args.alpha,
        "i_max": args.i_max,
        "seed": args.seed,
        "v1_initial_prompt": result_v1["initial_prompt"],
        "v1_final_prompt": result_v1["final_prompt"],
        "v1_prompt_words": len(result_v1["final_prompt"].split()),
        "v2_initial_prompt": result_v2["initial_prompt"],
        "v2_final_prompt": result_v2["final_prompt"],
        "v2_prompt_words": len(result_v2["final_prompt"].split()),
        "comparison": comparison,
        "v1_time": result_v1["total_time"],
        "v2_time": result_v2["total_time"],
    }

    summary_path = base_dir / "ab_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"A/B TEST COMPLETE")
    print(f"{'='*80}")
    print(f"V1 output: {dir_v1}")
    print(f"V2 output: {dir_v2}")
    print(f"Summary: {summary_path}")
    print(f"\nV1 prompt ({result_v1.get('num_iterations')} iters, "
          f"{len(result_v1['final_prompt'].split())} words): {result_v1['final_prompt'][:80]}...")
    print(f"V2 prompt ({result_v2.get('num_iterations')} iters, "
          f"{len(result_v2['final_prompt'].split())} words): {result_v2['final_prompt'][:80]}...")


if __name__ == "__main__":
    main()
