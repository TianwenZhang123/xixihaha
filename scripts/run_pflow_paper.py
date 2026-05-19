"""
P-Flow Paper-Faithful Entry Point for AutoDL.

Runs the complete P-Flow algorithm (Algorithm 1) on AutoDL 4090:
- Local Wan 2.1-T2V-1.3B model
- Qwen3-VL-Flash via DashScope API
- Fixed i_max iterations (no early stopping)
- MovieGenBench dataset

Usage:
    # Set API key first:
    export DASHSCOPE_API_KEY="your-key-here"

    # Run by video index (auto-reads prompt from dataset txt):
    python scripts/run_pflow_paper.py --video_index 23 --seed 42

    # Run with explicit video + prompt:
    python scripts/run_pflow_paper.py \
        --reference_video /root/autodl-tmp/data/moviegen_bench/23.mp4 \
        --prompt "A cat waking up its sleeping owner demanding breakfast..." \
        --output_dir /root/autodl-tmp/outputs/test_023 \
        --seed 42

    # Mock mode (no GPU/API, for testing logic):
    python scripts/run_pflow_paper.py --video_index 0 --mock

    # Override iterations:
    python scripts/run_pflow_paper.py --video_index 23 --i_max 5
"""

import argparse
import csv
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="P-Flow Paper-Faithful: Test-Time Prompt Optimization (AutoDL)"
    )

    # Video/prompt source (either --video_index OR --reference_video + --prompt)
    parser.add_argument(
        "--video_index", type=int, default=None,
        help="Video index in MovieGenBench dataset (0-based). "
             "Auto-resolves video path and prompt from dataset files.",
    )
    parser.add_argument(
        "--dataset_dir", type=str,
        default="/root/autodl-tmp/data/moviegen_bench",
        help="Path to MovieGenBench dataset root.",
    )
    parser.add_argument(
        "--reference_video", type=str, default=None,
        help="Path to the reference video (overrides --video_index).",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Initial text prompt (overrides --video_index).",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir", type=str, default="/root/autodl-tmp/outputs/pflow_run",
        help="Directory to save all output files.",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model_path", type=str, default="/root/autodl-tmp/models/Wan2.1-T2V-1.3B",
        help="Local path to Wan 2.1-T2V-1.3B model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    # Algorithm parameters
    parser.add_argument(
        "--i_max", type=int, default=None,
        help="Override fixed iteration count (paper: 10, default: 3 for 1.3B).",
    )

    # VLM settings
    parser.add_argument(
        "--vlm_model", type=str, default=None,
        help="Override VLM model name (default: qwen-vl-max).",
    )
    parser.add_argument(
        "--vlm_api_key", type=str, default=None,
        help="DashScope API key (or set DASHSCOPE_API_KEY env var).",
    )

    # Noise prior parameters
    parser.add_argument("--alpha", type=float, default=None, help="Noise blending weight.")
    parser.add_argument("--rho_s", type=float, default=None, help="Spatial SVD ratio.")
    parser.add_argument("--rho_m", type=float, default=None, help="Temporal SVD ratio.")

    # Video parameters
    parser.add_argument("--height", type=int, default=None, help="Video height.")
    parser.add_argument("--width", type=int, default=None, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of video frames.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale.")

    # VLM structured input (paper Listing 1 placeholders)
    parser.add_argument(
        "--desired_effect", type=str, default="",
        help="Description of target visual effect for VLM.",
    )
    parser.add_argument(
        "--subject", type=str, default="",
        help="Main subject for VLM structured input.",
    )
    parser.add_argument(
        "--environment", type=str, default="",
        help="Scene environment for VLM structured input.",
    )

    # Mode flags
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock mode (no GPU/API, for testing logic).",
    )
    parser.add_argument(
        "--noise_prior_only", action="store_true",
        help="Only run noise prior enhancement (skip optimization loop).",
    )

    return parser.parse_args()


def build_config_overrides(args) -> dict:
    """Build config override dictionary from command line arguments."""
    overrides = {}

    if args.i_max is not None:
        overrides.setdefault("optimization", {})["i_max"] = args.i_max

    if args.alpha is not None:
        overrides.setdefault("noise_prior", {})["alpha"] = args.alpha
    if args.rho_s is not None:
        overrides.setdefault("noise_prior", {})["rho_s"] = args.rho_s
    if args.rho_m is not None:
        overrides.setdefault("noise_prior", {})["rho_m"] = args.rho_m

    if args.guidance_scale is not None:
        overrides.setdefault("video", {})["guidance_scale"] = args.guidance_scale
    if args.num_frames is not None:
        overrides.setdefault("video", {})["num_frames"] = args.num_frames
    if args.height is not None:
        overrides.setdefault("video", {})["height"] = args.height
    if args.width is not None:
        overrides.setdefault("video", {})["width"] = args.width

    if args.vlm_model is not None:
        overrides.setdefault("vlm", {})["model_name"] = args.vlm_model

    return overrides


def resolve_video_and_prompt(args):
    """
    Resolve reference_video path and prompt from arguments.
    
    Priority:
    1. If --reference_video and --prompt are both given, use them directly.
    2. If --video_index is given, auto-read from dataset.
    3. Otherwise, error.
    """
    if args.reference_video and args.prompt:
        return args.reference_video, args.prompt

    if args.video_index is not None:
        dataset_dir = args.dataset_dir
        
        # Resolve video path (videos are directly in dataset_dir)
        video_path = os.path.join(dataset_dir, f"{args.video_index}.mp4")
        if args.reference_video:
            video_path = args.reference_video
        
        # Read prompt from txt file (1 prompt per line, 0-indexed)
        prompts_file = os.path.join(dataset_dir, "MovieGenVideoBench.txt")
        if not os.path.exists(prompts_file):
            print(f"Error: Prompts file not found: {prompts_file}")
            sys.exit(1)
        
        with open(prompts_file, "r", encoding="utf-8") as f:
            all_prompts = [line.strip() for line in f.readlines()]
        
        if args.video_index < 0 or args.video_index >= len(all_prompts):
            print(f"Error: video_index {args.video_index} out of range [0, {len(all_prompts)-1}]")
            sys.exit(1)
        
        prompt = args.prompt if args.prompt else all_prompts[args.video_index]
        
        return video_path, prompt
    
    print("Error: Must provide either --video_index or both --reference_video and --prompt.")
    parser_help = "  Example: python scripts/run_pflow_paper.py --video_index 23 --seed 42"
    print(parser_help)
    sys.exit(1)


def main():
    args = parse_args()

    # Resolve video and prompt
    reference_video, prompt = resolve_video_and_prompt(args)

    # Validate input
    if not os.path.exists(reference_video):
        print(f"Error: Reference video not found: {reference_video}")
        sys.exit(1)

    # Auto-resolve output dir from video_index if not explicitly set
    if args.video_index is not None and args.output_dir == "/root/autodl-tmp/outputs/pflow_run":
        args.output_dir = f"/root/autodl-tmp/outputs/test_{args.video_index:03d}"

    # Build config overrides
    config_overrides = build_config_overrides(args)

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    # Import pipeline
    from pflow.pipeline import PFlowPipeline

    # Initialize pipeline
    print("=" * 60)
    print("Initializing P-Flow pipeline (paper-faithful)...")
    print(f"  Video: {reference_video}")
    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    pipeline = PFlowPipeline(
        model_path=args.model_path,
        config_path=config_path,
        config=config_overrides if config_overrides else None,
        device="cuda" if not args.mock else "cpu",
        use_mock_vlm=args.mock,
        vlm_api_key=args.vlm_api_key,
        vlm_model=args.vlm_model,
    )

    # Run
    if args.noise_prior_only:
        output_path = os.path.join(args.output_dir, "noise_prior_result.mp4")
        os.makedirs(args.output_dir, exist_ok=True)

        result_path = pipeline.run_noise_prior_only(
            reference_video_path=reference_video,
            prompt=prompt,
            output_path=output_path,
            seed=args.seed,
        )
        print(f"\nNoise prior result saved to: {result_path}")
    else:
        # Full P-Flow pipeline (Algorithm 1)
        results = pipeline.run(
            reference_video_path=reference_video,
            prompt=prompt,
            output_dir=args.output_dir,
            seed=args.seed,
            desired_visual_effect=args.desired_effect,
            subject=args.subject,
            environment=args.environment,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("RUN COMPLETE — Summary:")
        print("=" * 60)
        print(f"  Output dir: {results['output_dir']}")
        print(f"  Iterations: {results['num_iterations']} (fixed)")
        print(f"  Final prompt: {results['final_prompt'][:100]}...")
        print(f"  Video files: {len(results['video_paths'])}")
        for vp in results['video_paths']:
            print(f"    - {vp}")
        print(f"\n  Next step: Run VBench/FVD to select best video.")
        print("=" * 60)


if __name__ == "__main__":
    main()
