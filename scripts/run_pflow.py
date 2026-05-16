"""
Main Entry Point for P-Flow Pipeline.

Usage:
    python scripts/run_pflow.py \
        --reference_video path/to/reference.mp4 \
        --prompt "A cat walking through magical sparkles" \
        --output_dir outputs/ \
        --config config/default.yaml

    # Test mode (without GPU/API):
    python scripts/run_pflow.py \
        --reference_video path/to/reference.mp4 \
        --prompt "test prompt" \
        --mock \
        --output_dir outputs/
"""

import argparse
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
        description="P-Flow: Test-Time Prompt Optimization for Video Visual Effects"
    )
    
    # Required arguments
    parser.add_argument(
        "--reference_video",
        type=str,
        required=True,
        help="Path to the reference video containing the target visual effect.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial text prompt describing the desired video content.",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save all output files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.1-T2V-14B",
        help="HuggingFace model name for video generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override max optimization iterations from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device (cuda/cpu).",
    )
    parser.add_argument(
        "--vlm_api_key",
        type=str,
        default=None,
        help="API key for VLM relay (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--vlm_base_url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible API relay (e.g., https://api.linkapi.org/v1).",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default=None,
        help="Override VLM model name (e.g., 'gemini-2.0-flash', 'gemini-1.5-pro').",
    )
    
    # Mode flags
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock VLM (for testing without API access).",
    )
    parser.add_argument(
        "--noise_prior_only",
        action="store_true",
        help="Only run noise prior enhancement (skip prompt optimization).",
    )
    parser.add_argument(
        "--no_save_intermediate",
        action="store_true",
        help="Don't save intermediate generated videos.",
    )
    
    # Hyperparameter overrides
    parser.add_argument("--alpha", type=float, default=None, help="Noise blending weight.")
    parser.add_argument("--rho_s", type=float, default=None, help="Spatial SVD ratio.")
    parser.add_argument("--rho_m", type=float, default=None, help="Temporal SVD ratio.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of video frames.")
    parser.add_argument("--height", type=int, default=None, help="Video height.")
    parser.add_argument("--width", type=int, default=None, help="Video width.")
    
    return parser.parse_args()


def build_config_overrides(args) -> dict:
    """Build config override dictionary from command line arguments."""
    overrides = {}
    
    if args.max_iterations is not None:
        overrides.setdefault("prompt_optimization", {})["max_iterations"] = args.max_iterations
    if args.vlm_model is not None:
        overrides.setdefault("prompt_optimization", {})["vlm_model"] = args.vlm_model
    
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
    
    if args.no_save_intermediate:
        overrides.setdefault("output", {})["save_intermediate"] = False
    
    return overrides


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.reference_video):
        print(f"Error: Reference video not found: {args.reference_video}")
        sys.exit(1)
    
    # Build config overrides
    config_overrides = build_config_overrides(args)
    
    # Import pipeline
    from pflow.pipeline import PFlowPipeline
    
    # Initialize pipeline
    print("Initializing P-Flow pipeline...")
    pipeline = PFlowPipeline(
        model_name=args.model,
        config_path=args.config,
        config=config_overrides if config_overrides else None,
        device=args.device,
        use_mock_vlm=args.mock,
        vlm_api_key=args.vlm_api_key,
        vlm_base_url=args.vlm_base_url,
    )
    
    # Run pipeline
    if args.noise_prior_only:
        # Only noise prior enhancement (no prompt optimization)
        output_path = os.path.join(args.output_dir, "noise_prior_result.mp4")
        os.makedirs(args.output_dir, exist_ok=True)
        
        result_path = pipeline.run_noise_prior_only(
            reference_video_path=args.reference_video,
            prompt=args.prompt,
            output_path=output_path,
            seed=args.seed,
        )
        print(f"\nResult saved to: {result_path}")
    else:
        # Full P-Flow pipeline
        results = pipeline.run(
            reference_video_path=args.reference_video,
            prompt=args.prompt,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        
        # Print summary
        print("\n" + "=" * 40)
        print("RESULTS SUMMARY")
        print("=" * 40)
        print(f"Best video: {results['best_video_path']}")
        print(f"Best prompt: {results['best_prompt']}")
        print(f"Best confidence: {results['best_confidence']:.3f}")
        print(f"Total iterations: {results['num_iterations']}")
        
        # Save full results
        results_file = os.path.join(args.output_dir, "final_results.json")
        # Remove non-serializable items
        results_serializable = {k: v for k, v in results.items() if k != "trajectory"}
        results_serializable["trajectory_summary"] = {
            "num_entries": len(results.get("trajectory", [])),
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
