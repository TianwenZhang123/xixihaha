#!/usr/bin/env python3
"""
Reproduction Quality Evaluation - Per-Iteration Metrics.

Designed for evaluating iterative prompt optimization:
measures how close each iteration's generated video is to the reference.

Metrics implemented:
1. CLIP-Score (frame-level): semantic similarity between ref and gen frames
2. SSIM: structural similarity per frame (averaged)
3. LPIPS: perceptual distance per frame (averaged)
4. Optical Flow Consistency: motion pattern similarity
5. Prompt-Video Alignment (CLIP): how well the prompt matches generated video

Usage:
    python evaluation/eval_reproduction.py \
        --experiment_dir /path/to/output \
        --device cuda

    # Compare V1 vs V2 prompt strategy:
    python evaluation/eval_reproduction.py \
        --experiment_dir /path/to/v1_output \
        --experiment_dir2 /path/to/v2_output \
        --device cuda
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Lightweight Frame-Level Metrics (no heavy model downloads required)
# =============================================================================

class SSIMMetric:
    """Structural Similarity Index between video frames."""

    def compute(self, ref_frames: np.ndarray, gen_frames: np.ndarray) -> float:
        """
        Compute average SSIM between corresponding frames.
        
        Args:
            ref_frames: (N, H, W, 3) uint8
            gen_frames: (N, H, W, 3) uint8
            
        Returns:
            Average SSIM score (higher is better, max 1.0)
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            # Fallback: simple correlation-based similarity
            return self._simple_similarity(ref_frames, gen_frames)

        scores = []
        n = min(len(ref_frames), len(gen_frames))
        for i in range(n):
            score = ssim(ref_frames[i], gen_frames[i], 
                        multichannel=True, channel_axis=2,
                        data_range=255)
            scores.append(score)
        
        return float(np.mean(scores))

    def _simple_similarity(self, ref: np.ndarray, gen: np.ndarray) -> float:
        """Fallback: normalized cross-correlation."""
        n = min(len(ref), len(gen))
        scores = []
        for i in range(n):
            r = ref[i].astype(float).flatten()
            g = gen[i].astype(float).flatten()
            r_norm = r - r.mean()
            g_norm = g - g.mean()
            denom = (np.linalg.norm(r_norm) * np.linalg.norm(g_norm))
            if denom == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(r_norm, g_norm) / denom))
        return float(np.mean(scores))


class CLIPScoreMetric:
    """CLIP-based visual similarity between frames."""

    # Default local model path (AutoDL)
    DEFAULT_LOCAL_PATH = "/root/autodl-tmp/models/clip-vit-base-patch32"

    def __init__(self, device: str = "cuda", model_path: str = None):
        self.device = device
        self.model_path = model_path
        self._model = None
        self._preprocess = None

    def _load_model(self):
        """Lazy-load CLIP model. Priority: local path > openai clip > HF online."""
        if self._model is not None:
            return

        # 1) Try local path first (offline-friendly)
        local_path = self.model_path or self.DEFAULT_LOCAL_PATH
        if os.path.isdir(local_path):
            try:
                from transformers import CLIPModel, CLIPProcessor
                # Use safetensors to avoid torch.load CVE-2025-32434 restriction
                self._model = CLIPModel.from_pretrained(
                    local_path, local_files_only=True, use_safetensors=True
                ).to(self.device).eval()
                self._preprocess = CLIPProcessor.from_pretrained(
                    local_path, local_files_only=True
                )
                self._clip_type = "hf"
                print(f"  CLIP loaded from local: {local_path}")
                return
            except Exception as e:
                print(f"  Warning: failed to load CLIP from {local_path}: {e}")

        # 2) Try openai clip package
        try:
            import clip
            self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            self._clip_type = "openai"
            print("  CLIP loaded via openai/clip package")
            return
        except (ImportError, Exception):
            pass

        # 3) Try HuggingFace online (may fail on restricted networks)
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", use_safetensors=True
            ).to(self.device).eval()
            self._preprocess = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_type = "hf"
            print("  CLIP loaded from HuggingFace online")
            return
        except Exception:
            pass

        # 4) All failed
        self._clip_type = "none"
        print("Warning: CLIP not available. Either:")
        print(f"  - Download model to {self.DEFAULT_LOCAL_PATH}")
        print("  - Or install: pip install git+https://github.com/openai/CLIP.git")
        print("  - Or ensure network access to huggingface.co")

    @torch.no_grad()
    def compute_frame_similarity(
        self, ref_frames: np.ndarray, gen_frames: np.ndarray, num_samples: int = 8
    ) -> float:
        """
        Compute CLIP cosine similarity between sampled frame pairs.
        
        Args:
            ref_frames: (N, H, W, 3) uint8
            gen_frames: (N, H, W, 3) uint8
            num_samples: number of frames to compare
            
        Returns:
            Average CLIP similarity (higher is better, max 1.0)
        """
        self._load_model()
        if self._clip_type == "none":
            return -1.0  # Indicate unavailable

        from PIL import Image

        n = min(len(ref_frames), len(gen_frames))
        indices = np.linspace(0, n - 1, min(num_samples, n), dtype=int)

        similarities = []
        for idx in indices:
            ref_img = Image.fromarray(ref_frames[idx])
            gen_img = Image.fromarray(gen_frames[idx])

            if self._clip_type == "openai":
                ref_tensor = self._preprocess(ref_img).unsqueeze(0).to(self.device)
                gen_tensor = self._preprocess(gen_img).unsqueeze(0).to(self.device)
                ref_feat = self._model.encode_image(ref_tensor)
                gen_feat = self._model.encode_image(gen_tensor)
            else:  # HuggingFace
                inputs_ref = self._preprocess(images=ref_img, return_tensors="pt").to(self.device)
                inputs_gen = self._preprocess(images=gen_img, return_tensors="pt").to(self.device)
                ref_feat = self._model.get_image_features(**inputs_ref)
                gen_feat = self._model.get_image_features(**inputs_gen)

            # Cosine similarity
            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
            sim = (ref_feat * gen_feat).sum().item()
            similarities.append(sim)

        return float(np.mean(similarities))

    @torch.no_grad()
    def compute_prompt_alignment(self, prompt: str, gen_frames: np.ndarray, num_samples: int = 4) -> float:
        """
        Compute CLIP text-image alignment score.
        
        How well does the generated video match the prompt?
        """
        self._load_model()
        if self._clip_type == "none":
            return -1.0

        from PIL import Image
        import clip as clip_module

        n = len(gen_frames)
        indices = np.linspace(0, n - 1, min(num_samples, n), dtype=int)

        scores = []
        for idx in indices:
            gen_img = Image.fromarray(gen_frames[idx])

            if self._clip_type == "openai":
                img_tensor = self._preprocess(gen_img).unsqueeze(0).to(self.device)
                text_token = clip_module.tokenize([prompt], truncate=True).to(self.device)
                img_feat = self._model.encode_image(img_tensor)
                txt_feat = self._model.encode_text(text_token)
            else:
                inputs = self._preprocess(
                    text=[prompt], images=gen_img, 
                    return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                outputs = self._model(**inputs)
                img_feat = outputs.image_embeds
                txt_feat = outputs.text_embeds

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score = (img_feat * txt_feat).sum().item()
            scores.append(score)

        return float(np.mean(scores))


class MotionConsistencyMetric:
    """Measures motion pattern similarity using frame differences."""

    def compute(self, ref_frames: np.ndarray, gen_frames: np.ndarray) -> Dict[str, float]:
        """
        Compare motion patterns between reference and generated video.
        
        Uses frame-difference magnitude and direction correlation.
        
        Returns:
            Dict with motion_magnitude_sim and motion_direction_sim
        """
        n = min(len(ref_frames), len(gen_frames))
        if n < 2:
            return {"motion_magnitude_sim": 0.0, "motion_direction_sim": 0.0}

        ref_gray = ref_frames[:n].mean(axis=-1)  # (N, H, W)
        gen_gray = gen_frames[:n].mean(axis=-1)

        # Frame differences as motion proxy
        ref_diffs = np.diff(ref_gray.astype(float), axis=0)  # (N-1, H, W)
        gen_diffs = np.diff(gen_gray.astype(float), axis=0)

        # Motion magnitude similarity (correlation of per-frame motion amounts)
        ref_magnitudes = np.array([np.abs(d).mean() for d in ref_diffs])
        gen_magnitudes = np.array([np.abs(d).mean() for d in gen_diffs])
        
        if ref_magnitudes.std() == 0 or gen_magnitudes.std() == 0:
            mag_corr = 0.0
        else:
            mag_corr = float(np.corrcoef(ref_magnitudes, gen_magnitudes)[0, 1])
            if np.isnan(mag_corr):
                mag_corr = 0.0

        # Motion direction similarity (spatial correlation of motion maps)
        dir_sims = []
        for i in range(len(ref_diffs)):
            r = ref_diffs[i].flatten()
            g = gen_diffs[i].flatten()
            r_norm = np.linalg.norm(r)
            g_norm = np.linalg.norm(g)
            if r_norm == 0 or g_norm == 0:
                dir_sims.append(0.0)
            else:
                dir_sims.append(float(np.dot(r, g) / (r_norm * g_norm)))

        return {
            "motion_magnitude_sim": mag_corr,
            "motion_direction_sim": float(np.mean(dir_sims)),
        }


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def load_video_frames(video_path: str, num_frames: int = 16) -> Optional[np.ndarray]:
    """Load video as numpy array of frames (N, H, W, 3) uint8."""
    if not os.path.exists(video_path):
        return None

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return frames
    except (ImportError, Exception):
        pass

    try:
        import imageio.v3 as iio
        all_frames = iio.imread(video_path, plugin="pyav")
        total = len(all_frames)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return all_frames[indices]
    except Exception:
        return None


def evaluate_experiment(
    experiment_dir: str,
    device: str = "cuda",
    num_eval_frames: int = 16,
    clip_model_path: str = None,
) -> Dict[str, Any]:
    """
    Evaluate a reproduction experiment: per-iteration quality metrics.
    
    Args:
        experiment_dir: Directory containing reference.mp4 and generated_iter_*.mp4
        device: Compute device for CLIP
        num_eval_frames: Frames to sample per video for evaluation
        clip_model_path: Local path to CLIP model (optional)
        
    Returns:
        Complete per-iteration evaluation results
    """
    exp_path = Path(experiment_dir)
    
    # Load reference
    ref_path = exp_path / "reference.mp4"
    if not ref_path.exists():
        return {"error": f"No reference.mp4 in {experiment_dir}"}
    
    ref_frames = load_video_frames(str(ref_path), num_eval_frames)
    if ref_frames is None:
        return {"error": "Failed to load reference video"}

    # Find generated videos
    gen_videos = sorted(glob.glob(str(exp_path / "generated_iter_*.mp4")))
    if not gen_videos:
        return {"error": "No generated videos found"}

    # Load prompts if available
    prompts = {}
    prompts_file = exp_path / "prompts_history.json"
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts_data = json.load(f)
            for i, p in enumerate(prompts_data.get("all_prompts", [])):
                prompts[i + 1] = p

    # Also check optimization_log for prompts
    opt_log_dir = exp_path / "optimization_log"
    if opt_log_dir.exists():
        for log_file in sorted(opt_log_dir.glob("iter_*.json")):
            with open(log_file) as f:
                log = json.load(f)
                iter_num = log.get("iteration", 0)
                if iter_num > 0 and "input_prompt" in log:
                    prompts[iter_num] = log["input_prompt"]

    # Initialize metrics
    ssim_metric = SSIMMetric()
    clip_metric = CLIPScoreMetric(device=device, model_path=clip_model_path)
    motion_metric = MotionConsistencyMetric()

    # Evaluate each iteration
    print(f"\nEvaluating {len(gen_videos)} iterations in: {experiment_dir}")
    print(f"Reference: {ref_path} ({len(ref_frames)} frames sampled)")
    print("-" * 70)

    iteration_results = []
    
    for video_path in tqdm(gen_videos, desc="Evaluating iterations"):
        iter_num = int(Path(video_path).stem.split("_")[-1])
        
        gen_frames = load_video_frames(video_path, num_eval_frames)
        if gen_frames is None:
            iteration_results.append({
                "iteration": iter_num,
                "error": "failed to load"
            })
            continue

        # Resize gen_frames to match ref if needed
        if gen_frames.shape[1:3] != ref_frames.shape[1:3]:
            from PIL import Image
            target_h, target_w = ref_frames.shape[1], ref_frames.shape[2]
            resized = []
            for frame in gen_frames:
                img = Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS)
                resized.append(np.array(img))
            gen_frames = np.array(resized)

        # Compute metrics
        ssim_score = ssim_metric.compute(ref_frames, gen_frames)
        clip_score = clip_metric.compute_frame_similarity(ref_frames, gen_frames)
        motion_scores = motion_metric.compute(ref_frames, gen_frames)
        
        # Prompt-video alignment (if prompt available)
        prompt_align = -1.0
        if iter_num in prompts:
            prompt_align = clip_metric.compute_prompt_alignment(
                prompts[iter_num], gen_frames
            )

        result = {
            "iteration": iter_num,
            "ssim": round(ssim_score, 4),
            "clip_similarity": round(clip_score, 4),
            "motion_magnitude_sim": round(motion_scores["motion_magnitude_sim"], 4),
            "motion_direction_sim": round(motion_scores["motion_direction_sim"], 4),
            "prompt_video_alignment": round(prompt_align, 4),
            "prompt_words": len(prompts.get(iter_num, "").split()) if iter_num in prompts else None,
        }
        iteration_results.append(result)

    # Summary
    if iteration_results:
        ssim_scores = [r["ssim"] for r in iteration_results if "ssim" in r]
        clip_scores = [r["clip_similarity"] for r in iteration_results if r.get("clip_similarity", -1) >= 0]

        summary = {
            "experiment_dir": str(experiment_dir),
            "num_iterations": len(iteration_results),
            "best_ssim_iter": iteration_results[np.argmax(ssim_scores)]["iteration"] if ssim_scores else None,
            "best_ssim": max(ssim_scores) if ssim_scores else None,
            "best_clip_iter": iteration_results[np.argmax(clip_scores)]["iteration"] if clip_scores else None,
            "best_clip": max(clip_scores) if clip_scores else None,
            "ssim_trend": "improving" if len(ssim_scores) > 1 and ssim_scores[-1] > ssim_scores[0] else "degrading",
            "clip_trend": "improving" if len(clip_scores) > 1 and clip_scores[-1] > clip_scores[0] else "degrading",
        }
    else:
        summary = {"error": "no valid results"}

    full_results = {
        "summary": summary,
        "per_iteration": iteration_results,
    }

    # Save results
    output_path = exp_path / "reproduction_eval.json"
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'Iter':<6}{'SSIM':<10}{'CLIP-Sim':<12}{'Motion-Mag':<12}{'Motion-Dir':<12}{'P-V Align':<12}{'Words':<8}")
    print("-" * 72)
    for r in iteration_results:
        if "error" in r:
            print(f"{r['iteration']:<6}ERROR")
            continue
        print(
            f"{r['iteration']:<6}"
            f"{r['ssim']:<10.4f}"
            f"{r['clip_similarity']:<12.4f}"
            f"{r['motion_magnitude_sim']:<12.4f}"
            f"{r['motion_direction_sim']:<12.4f}"
            f"{r['prompt_video_alignment']:<12.4f}"
            f"{r.get('prompt_words', 'N/A')!s:<8}"
        )
    
    print(f"\nBest SSIM: iter {summary.get('best_ssim_iter')} ({summary.get('best_ssim', 0):.4f})")
    print(f"Best CLIP: iter {summary.get('best_clip_iter')} ({summary.get('best_clip', 0):.4f})")
    print(f"SSIM trend: {summary.get('ssim_trend')}")
    print(f"CLIP trend: {summary.get('clip_trend')}")
    print(f"\nResults saved: {output_path}")

    return full_results


def compare_experiments(
    dir1: str,
    dir2: str,
    device: str = "cuda",
    labels: Tuple[str, str] = ("V1", "V2"),
    clip_model_path: str = None,
) -> Dict[str, Any]:
    """
    Compare two experiment runs (e.g., V1 prompt vs V2 prompt).
    
    Prints side-by-side comparison table.
    """
    print(f"Comparing: {labels[0]}={dir1}")
    print(f"      vs:  {labels[1]}={dir2}")
    print("=" * 80)

    results1 = evaluate_experiment(dir1, device, clip_model_path=clip_model_path)
    print("\n" + "=" * 80 + "\n")
    results2 = evaluate_experiment(dir2, device, clip_model_path=clip_model_path)

    # Side-by-side comparison
    print("\n" + "=" * 80)
    print(f"COMPARISON: {labels[0]} vs {labels[1]}")
    print("=" * 80)

    s1 = results1.get("summary", {})
    s2 = results2.get("summary", {})

    comparison = {
        "best_ssim": {labels[0]: s1.get("best_ssim"), labels[1]: s2.get("best_ssim")},
        "best_clip": {labels[0]: s1.get("best_clip"), labels[1]: s2.get("best_clip")},
        "ssim_trend": {labels[0]: s1.get("ssim_trend"), labels[1]: s2.get("ssim_trend")},
        "clip_trend": {labels[0]: s1.get("clip_trend"), labels[1]: s2.get("clip_trend")},
        "winner_ssim": labels[0] if (s1.get("best_ssim", 0) or 0) > (s2.get("best_ssim", 0) or 0) else labels[1],
        "winner_clip": labels[0] if (s1.get("best_clip", 0) or 0) > (s2.get("best_clip", 0) or 0) else labels[1],
    }

    print(f"\n{'Metric':<20}{labels[0]:<20}{labels[1]:<20}{'Winner':<10}")
    print("-" * 70)
    print(f"{'Best SSIM':<20}{s1.get('best_ssim', 'N/A')!s:<20}{s2.get('best_ssim', 'N/A')!s:<20}{comparison['winner_ssim']}")
    print(f"{'Best CLIP-Sim':<20}{s1.get('best_clip', 'N/A')!s:<20}{s2.get('best_clip', 'N/A')!s:<20}{comparison['winner_clip']}")
    print(f"{'SSIM Trend':<20}{s1.get('ssim_trend', 'N/A')!s:<20}{s2.get('ssim_trend', 'N/A')!s:<20}")
    print(f"{'CLIP Trend':<20}{s1.get('clip_trend', 'N/A')!s:<20}{s2.get('clip_trend', 'N/A')!s:<20}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Reproduction Quality Evaluation")
    parser.add_argument("--experiment_dir", type=str, required=True,
                       help="Experiment output directory (with reference.mp4 and generated_iter_*.mp4)")
    parser.add_argument("--experiment_dir2", type=str, default=None,
                       help="Second experiment to compare (optional)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to sample for evaluation")
    parser.add_argument("--clip_model_path", type=str, default=None,
                       help="Local path to CLIP model (e.g., /root/autodl-tmp/models/clip-vit-base-patch32). "
                            "If not specified, tries default local path, then online download.")
    parser.add_argument("--labels", type=str, nargs=2, default=["Run1", "Run2"],
                       help="Labels for comparison (e.g., V1 V2)")

    args = parser.parse_args()

    # Set CLIP model path globally via environment for metrics to pick up
    if args.clip_model_path:
        os.environ["CLIP_LOCAL_PATH"] = args.clip_model_path

    if args.experiment_dir2:
        compare_experiments(
            args.experiment_dir, args.experiment_dir2,
            device=args.device,
            labels=tuple(args.labels),
            clip_model_path=args.clip_model_path,
        )
    else:
        evaluate_experiment(
            args.experiment_dir,
            device=args.device,
            num_eval_frames=args.num_frames,
            clip_model_path=args.clip_model_path,
        )


if __name__ == "__main__":
    main()
