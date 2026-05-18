"""
P-Flow Pipeline with Wan 2.7 API Backend.

This script runs P-Flow using DashScope's Wan 2.7 API for video generation
instead of a local model. This provides:
- Higher quality output (Wan 2.7 is much larger than local 1.3B)
- No GPU memory requirement for video generation
- Still uses VLM (gemini-2.0-flash via LinkAPI) for prompt optimization

Architecture:
    Reference Video → Noise Prior (local) → Enhanced Noise
    Prompt → VLM Optimization Loop:
        - Generate video via Wan 2.7 API (cloud)
        - Analyze with gemini-2.0-flash (LinkAPI relay)
        - Refine prompt
    → Output best video

Note: Noise prior enhancement is skipped in API mode because we cannot
inject custom latents into the DashScope API. The API mode relies purely
on the test-time prompt optimization loop.

Usage:
    # Full P-Flow with Wan 2.7 API
    python scripts/run_pflow_api.py \
        --reference_video path/to/reference.mp4 \
        --prompt "Golden particles floating upward with glowing trails" \
        --output_dir outputs/wan27_run1 \
        --dashscope_api_key sk-xxxxx \
        --seed 42

    # Quick test (mock mode)
    python scripts/run_pflow_api.py \
        --reference_video path/to/reference.mp4 \
        --prompt "test" \
        --output_dir outputs/mock_api \
        --mock

    # Compare with local Wan 2.1:
    # Step 1: Run local 2.1
    python scripts/run_pflow.py --reference_video ref.mp4 --prompt "..." --output_dir outputs/wan21
    # Step 2: Run API 2.7
    python scripts/run_pflow_api.py --reference_video ref.mp4 --prompt "..." --output_dir outputs/wan27
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="P-Flow with Wan 2.7 API backend (DashScope)"
    )
    
    # Required
    parser.add_argument(
        "--reference_video", type=str, required=True,
        help="Path to the reference video."
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing desired visual effect."
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/wan27_api",
        help="Output directory."
    )
    
    # Wan 2.7 API settings
    parser.add_argument(
        "--dashscope_api_key", type=str, default=None,
        help="DashScope API key (or set DASHSCOPE_API_KEY env var)."
    )
    parser.add_argument(
        "--wan_model", type=str, default="wan2.7-t2v",
        choices=["wan2.7-t2v", "wan2.6-t2v", "wan2.5-t2v"],
        help="Wan model version for video generation."
    )
    parser.add_argument(
        "--video_size", type=str, default="1280*720",
        choices=["960*480", "1280*720", "1920*1080"],
        help="Output video resolution."
    )
    parser.add_argument(
        "--video_duration", type=int, default=5,
        choices=[5, 10, 15],
        help="Video duration in seconds."
    )
    
    # VLM settings (for prompt optimization)
    parser.add_argument(
        "--vlm_api_key", type=str, default=None,
        help="API key for VLM relay (or set OPENAI_API_KEY env var)."
    )
    parser.add_argument(
        "--vlm_base_url", type=str, default="https://api.linkapi.org/v1",
        help="Base URL for VLM relay."
    )
    parser.add_argument(
        "--vlm_model", type=str, default="gemini-2.0-flash",
        help="VLM model for prompt optimization."
    )
    
    # Optimization settings
    parser.add_argument(
        "--max_iterations", type=int, default=5,
        help="Max prompt optimization iterations (fewer for API to save cost)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    
    # Flags
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock mode (no API calls, for testing)."
    )
    parser.add_argument(
        "--single_shot", action="store_true",
        help="Generate one video without optimization loop (cheapest)."
    )
    parser.add_argument(
        "--prompt_extend", action="store_true", default=True,
        help="Enable DashScope smart prompt rewriting."
    )
    parser.add_argument(
        "--no_prompt_extend", action="store_true",
        help="Disable DashScope smart prompt rewriting."
    )
    parser.add_argument(
        "--multi_shot", action="store_true",
        help="Enable multi-shot (multi-scene) video generation."
    )
    
    return parser.parse_args()


def run_single_shot(wan_client, prompt, output_dir, seed, prompt_extend, multi_shot):
    """Generate a single video without optimization loop."""
    output_path = os.path.join(output_dir, "generated.mp4")
    
    print("\n[Single Shot] Generating video...")
    result = wan_client.generate_video(
        prompt=prompt,
        output_path=output_path,
        prompt_extend=prompt_extend,
        seed=seed,
        shot_type="multi" if multi_shot else None,
    )
    
    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"  Video: {result['video_path']}")
    print(f"  Time: {result['elapsed_time']:.0f}s")
    print(f"  Task ID: {result['task_id']}")
    print(f"{'='*60}")
    
    return result


def run_optimization_loop(
    wan_client,
    vlm_client,
    reference_video_path,
    prompt,
    output_dir,
    max_iterations,
    seed,
    prompt_extend,
):
    """
    Run P-Flow prompt optimization loop with Wan 2.7 API.
    
    Flow:
    1. Generate video with current prompt via Wan 2.7 API
    2. Send [reference | generated] to VLM for analysis
    3. VLM suggests refined prompt
    4. Repeat until convergence or max iterations
    """
    from pflow.prompt_optimizer import PromptOptimizer
    from pflow.trajectory import TrajectoryManager
    from pflow.video_utils import load_video
    import torch
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("P-Flow API Mode: Test-Time Prompt Optimization")
    print("=" * 60)
    print(f"  Video backend: Wan 2.7 API (DashScope)")
    print(f"  VLM backend: {vlm_client.model_name if hasattr(vlm_client, 'model_name') else 'mock'}")
    print(f"  Reference: {reference_video_path}")
    print(f"  Initial prompt: {prompt}")
    print(f"  Max iterations: {max_iterations}")
    print("=" * 60)
    
    # Load reference video for VLM comparison
    print("\n[Step 1] Loading reference video for VLM analysis...")
    reference_video = load_video(
        reference_video_path,
        num_frames=16,  # Fewer frames needed for VLM analysis only
        height=480,
        width=832,
    )
    
    # Initialize components
    trajectory = TrajectoryManager(
        output_dir=str(output_path),
        save_all_videos=True,
    )
    trajectory.set_reference(reference_video, reference_video_path)
    
    prompt_optimizer = PromptOptimizer(
        vlm_client=vlm_client,
        max_iterations=max_iterations,
        output_dir=str(output_path),
    )
    
    # Optimization loop
    print(f"\n[Step 2] Starting optimization loop...")
    
    current_prompt = prompt
    best_video_path = None
    best_prompt = prompt
    best_confidence = 0.0
    all_results = []
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─'*50}")
        print(f"  Iteration {iteration}/{max_iterations}")
        print(f"  Prompt: {current_prompt[:100]}...")
        print(f"{'─'*50}")
        
        # Generate video via API
        iter_video_path = str(output_path / f"generated_iter_{iteration:03d}.mp4")
        
        try:
            gen_result = wan_client.generate_video(
                prompt=current_prompt,
                output_path=iter_video_path,
                prompt_extend=prompt_extend,
                seed=seed + iteration,  # Different seed per iteration
            )
            all_results.append(gen_result)
        except Exception as e:
            print(f"  ERROR: Video generation failed: {e}")
            break
        
        # Load generated video for VLM analysis
        try:
            generated_video = load_video(
                iter_video_path,
                num_frames=16,
                height=480,
                width=832,
            )
        except Exception as e:
            print(f"  WARNING: Could not load generated video for analysis: {e}")
            generated_video = torch.randn(3, 16, 480, 832)  # Fallback
        
        # Send to VLM for analysis
        print(f"  Analyzing with VLM...")
        _, prev_video, _ = trajectory.get_videos_for_vlm()
        
        optimization_result = prompt_optimizer.optimize_prompt(
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
            analysis=optimization_result.get("analysis", ""),
            improvements=optimization_result.get("improvements", []),
            confidence=optimization_result.get("confidence", 0.0),
            key_differences=optimization_result.get("key_differences", []),
        )
        
        # Track best
        confidence = optimization_result.get("confidence", 0.0)
        print(f"  Confidence: {confidence:.3f}")
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_video_path = iter_video_path
            best_prompt = current_prompt
        
        # Check convergence
        if prompt_optimizer.should_stop_early(
            trajectory.get_history_for_vlm(), min_iterations=2
        ):
            print(f"\n  ✓ Converged at iteration {iteration}!")
            break
        
        # Update prompt
        refined = optimization_result.get("refined_prompt", "")
        if refined:
            current_prompt = refined
    
    # Copy best video to final output
    if best_video_path and os.path.exists(best_video_path):
        import shutil
        final_path = str(output_path / "best_result.mp4")
        shutil.copy2(best_video_path, final_path)
    else:
        final_path = best_video_path
    
    # Save results
    results = {
        "best_video_path": final_path,
        "best_prompt": best_prompt,
        "best_confidence": best_confidence,
        "num_iterations": len(trajectory),
        "all_prompts": [prompt] + [e.prompt for e in trajectory.entries],
        "api_results": [
            {k: v for k, v in r.items() if k != "usage"}
            for r in all_results
        ],
        "total_api_time": sum(r.get("elapsed_time", 0) for r in all_results),
    }
    
    results_path = str(output_path / "final_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save prompts history
    prompts_path = str(output_path / "prompts_history.json")
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump({
            "initial_prompt": prompt,
            "best_prompt": best_prompt,
            "all_prompts": results["all_prompts"],
            "best_confidence": best_confidence,
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("P-Flow API Optimization Complete!")
    print(f"  Best confidence: {best_confidence:.3f}")
    print(f"  Total iterations: {len(trajectory)}")
    print(f"  Best prompt: {best_prompt[:100]}...")
    print(f"  Best video: {final_path}")
    print(f"  Total API time: {results['total_api_time']:.0f}s")
    print("=" * 60)
    
    return results


def main():
    args = parse_args()
    
    # Validate
    if not os.path.exists(args.reference_video):
        print(f"Error: Reference video not found: {args.reference_video}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompt_extend = args.prompt_extend and not args.no_prompt_extend
    
    # Initialize Wan API client
    if args.mock:
        from pflow.wan_api_client import MockWanAPIClient
        wan_client = MockWanAPIClient(model=args.wan_model)
    else:
        from pflow.wan_api_client import WanAPIClient
        wan_client = WanAPIClient(
            api_key=args.dashscope_api_key,
            model=args.wan_model,
            size=args.video_size,
            duration=args.video_duration,
        )
        
        # Test connection
        print("Testing DashScope API connection...")
        if not wan_client.test_connection():
            print("WARNING: API connection test failed. Continuing anyway...")
    
    # Single shot mode (no optimization)
    if args.single_shot:
        run_single_shot(
            wan_client=wan_client,
            prompt=args.prompt,
            output_dir=args.output_dir,
            seed=args.seed,
            prompt_extend=prompt_extend,
            multi_shot=args.multi_shot,
        )
        return
    
    # Full optimization mode
    # Initialize VLM client
    if args.mock:
        from pflow.vlm_client import MockVLMClient
        vlm_client = MockVLMClient()
    else:
        from pflow.vlm_client import VLMClient
        vlm_client = VLMClient(
            model_name=args.vlm_model,
            api_key=args.vlm_api_key,
            base_url=args.vlm_base_url,
        )
    
    run_optimization_loop(
        wan_client=wan_client,
        vlm_client=vlm_client,
        reference_video_path=args.reference_video,
        prompt=args.prompt,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        seed=args.seed,
        prompt_extend=prompt_extend,
    )


if __name__ == "__main__":
    main()
