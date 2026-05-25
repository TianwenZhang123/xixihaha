"""
P-Flow Enhancement Module: Iterative Prompt Optimization + Noise Prior.

This module provides the "ours" method that improves upon the baseline:
    Baseline: VLM caption → Wan generate (one-shot)
    P-Flow:   VLM caption → [Noise Prior + Iterative VLM Refinement] → Wan generate (best of N)

The enhancement is PLUGGABLE — it takes a baseline caption as input and
produces an optimized prompt + noise-guided generation.

Key components:
    1. Noise Prior: Flow Matching Inversion → SVD Filtering → Blended Noise
    2. Iterative Prompt Optimization: VLM compares ref vs generated → refines prompt
    3. Best Selection: Pick the iteration with highest CLIP similarity

Usage:
    baseline_pipeline = BaselinePipeline(config)
    enhancer = PFlowEnhancer(config, baseline_pipeline)
    result = enhancer.run_single(video_path, output_dir, sample_id, initial_caption)
"""

import os
import json
import time
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch
import numpy as np

from .distributed import setup_single_gpu, cleanup_gpu_memory
from .noise_prior import NoisePriorEnhancement
from .prompt_optimizer import PromptOptimizer
from .trajectory import TrajectoryManager
from .video_utils import (
    load_video, save_video_tensor, normalize_video, denormalize_video
)
from .flow_matching import encode_video_to_latents

logger = logging.getLogger(__name__)

# Same negative prompt for consistency
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, work, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, "
    "deformed, blurry, watermark"
)


class PFlowEnhancer:
    """
    P-Flow Enhancement: Noise Prior + Iterative Prompt Optimization.

    This is the "ours" method. It takes a baseline caption and enhances it
    through iterative VLM-guided refinement with noise prior guidance.

    Can be used as a drop-in enhancement on top of BaselinePipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        wan_pipe=None,
        vlm_client=None,
    ):
        """
        Args:
            config: Full configuration dict.
            wan_pipe: Pre-loaded Wan pipeline (shared with baseline).
            vlm_client: Pre-loaded VLM client (for iterative refinement).
        """
        self.config = config
        self.device = setup_single_gpu()
        self.dtype = getattr(torch, config.get("model", {}).get("dtype", "bfloat16"))

        self._pipe = wan_pipe
        self._vlm_client = vlm_client

        # Enhancement parameters
        noise_cfg = config.get("noise_prior", {})
        self.alpha = noise_cfg.get("alpha", 0.001)
        self.rho_s = noise_cfg.get("rho_s", 0.1)
        self.rho_m = noise_cfg.get("rho_m", 0.9)
        self.inversion_steps = noise_cfg.get("inversion_steps", 50)

        opt_cfg = config.get("optimization", {})
        self.i_max = opt_cfg.get("i_max", 10)

    @property
    def pipe(self):
        """Access the Wan pipeline (must be set externally or loaded)."""
        if self._pipe is None:
            from .distributed import load_model_single_gpu
            model_path = self.config["model"]["t2v_path"]
            self._pipe = load_model_single_gpu(
                model_path=model_path,
                dtype=self.dtype,
                model_type="t2v",
            )
        return self._pipe

    @pipe.setter
    def pipe(self, value):
        self._pipe = value

    @property
    def vlm_client(self):
        """Access the VLM client for iterative refinement."""
        if self._vlm_client is None:
            from .vlm_client import create_vlm_client
            self._vlm_client = create_vlm_client(self.config["vlm"])
        return self._vlm_client

    @vlm_client.setter
    def vlm_client(self, value):
        self._vlm_client = value

    @torch.no_grad()
    def run_single(
        self,
        video_path: str,
        output_dir: str,
        sample_id: int = 0,
        seed: int = 42,
        initial_caption: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run P-Flow enhancement for a single video.

        Steps:
            1. Load reference video → encode to latent
            2. Compute noise prior (inversion + SVD) — ONCE
            3. Iterative loop (i_max iterations):
               a. Blend noise prior with random noise
               b. Generate video with current prompt
               c. VLM compares ref vs generated → refine prompt
            4. Select best iteration (highest CLIP-sim or last)
            5. Save results

        Args:
            video_path: Path to reference video.
            output_dir: Output directory for this sample.
            sample_id: Sample identifier.
            seed: Random seed.
            initial_caption: Starting prompt (from baseline captioning).

        Returns:
            Dict with all iteration results and best video path.
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_cfg = self.config["video"]
        generator = torch.Generator(device=self.device).manual_seed(seed + sample_id)
        torch.manual_seed(seed + sample_id)

        logger.info(f"[P-Flow] Processing sample {sample_id}: {video_path}")
        logger.info(f"  Iterations: {self.i_max}, alpha: {self.alpha}")

        # Step 1: Load and encode reference video
        reference_video = load_video(
            video_path,
            num_frames=video_cfg["num_frames"],
            height=video_cfg["height"],
            width=video_cfg["width"],
            device=self.device,
        )

        ref_normalized = normalize_video(reference_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_normalized, self.device)

        # Step 2: Compute Noise Prior (ONCE)
        logger.info("  Computing noise prior (inversion + SVD)...")
        noise_enhancer = NoisePriorEnhancement(
            pipe=self.pipe,
            alpha=self.alpha,
            rho_s=self.rho_s,
            rho_m=self.rho_m,
            num_inversion_steps=self.inversion_steps,
            device=self.device,
        )

        prompt_embeds = self._encode_prompt(initial_caption or "a video scene")
        eta_temporal = noise_enhancer.compute_temporal_prior(
            video_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=prompt_embeds,
        )
        logger.info(f"  eta_temporal: mean={eta_temporal.mean():.4f}, std={eta_temporal.std():.4f}")

        # Step 3: Initialize optimization components
        trajectory = TrajectoryManager(
            output_dir=str(output_path),
            save_all_videos=True,
        )
        trajectory.set_reference(reference_video)

        prompt_optimizer = PromptOptimizer(
            vlm_client=self.vlm_client,
            max_iterations=self.i_max,
            output_dir=str(output_path),
        )

        # Step 4: Iterative Optimization Loop
        current_prompt = initial_caption or "a video scene"
        last_prompt = ""
        all_results = []

        for iteration in range(1, self.i_max + 1):
            iter_start = time.time()
            logger.info(f"  --- Iteration {iteration}/{self.i_max} ---")
            logger.info(f"    Prompt: {current_prompt[:80]}...")

            # 4a: Blend noise (motion prior + random exploration)
            eta = noise_enhancer.blend_noise(eta_temporal, generator)

            # 4b: Generate video with noise prior
            generated_video = self._generate_video(
                prompt=current_prompt,
                latents=eta,
                generator=generator,
            )

            # Save iteration video
            iter_video_path = str(output_path / f"iter_{iteration:03d}.mp4")
            save_video_tensor(generated_video, iter_video_path, fps=video_cfg["fps"])

            # 4c-4d: VLM analysis and prompt refinement
            prev_video = trajectory.get_previous_video()
            optimization_result = prompt_optimizer.optimize_prompt(
                current_prompt=current_prompt,
                reference_video=reference_video,
                generated_video=generated_video,
                iteration=iteration,
                desired_visual_effect=current_prompt,
                last_text_prompt=last_prompt,
                previous_video=prev_video,
                history=trajectory.get_text_history(),
            )

            # Update trajectory
            trajectory.add_entry(
                iteration=iteration,
                prompt=current_prompt,
                video=generated_video,
                video_path=iter_video_path,
                analysis=optimization_result.get("analysis", {}),
                refined_prompt=optimization_result.get("refined_prompt", ""),
            )

            iter_time = time.time() - iter_start
            all_results.append({
                "iteration": iteration,
                "prompt": current_prompt,
                "video_path": iter_video_path,
                "time_seconds": iter_time,
            })
            logger.info(f"    Iteration time: {iter_time:.1f}s")

            # Update prompt for next iteration
            last_prompt = current_prompt
            refined = optimization_result.get("refined_prompt", "")
            if refined and refined.strip():
                current_prompt = refined

        # Step 5: Select best iteration
        # Strategy: use last iteration (or could use CLIP-sim selection)
        best_iter = self.i_max
        best_video_path = str(output_path / f"iter_{best_iter:03d}.mp4")

        # Copy best to final output
        final_output = str(output_path / f"{sample_id}.mp4")
        import shutil
        shutil.copy2(best_video_path, final_output)

        total_time = time.time() - start_time
        logger.info(f"[P-Flow] Done. Total time: {total_time:.1f}s. Best: iter {best_iter}")

        # Save metadata
        metadata = {
            "sample_id": sample_id,
            "video_path": video_path,
            "initial_caption": initial_caption,
            "final_prompt": current_prompt,
            "best_iteration": best_iter,
            "total_iterations": self.i_max,
            "alpha": self.alpha,
            "total_time_seconds": total_time,
            "iterations": all_results,
            "method": "pflow",
        }
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata

    def _generate_video(
        self,
        prompt: str,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate video using Wan 2.1-1.3B with optional noise prior."""
        video_cfg = self.config["video"]

        kwargs = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": video_cfg["height"],
            "width": video_cfg["width"],
            "num_frames": video_cfg["num_frames"],
            "guidance_scale": video_cfg["guidance_scale"],
            "num_inference_steps": video_cfg["num_inference_steps"],
            "generator": generator,
            "output_type": "pt",
        }

        if latents is not None:
            kwargs["latents"] = latents

        output = self.pipe(**kwargs)

        # Handle output format
        if hasattr(output, "frames"):
            video = output.frames
            if isinstance(video, list):
                import torchvision.transforms as T
                transform = T.ToTensor()
                frames = [transform(f) for f in video[0]]
                video = torch.stack(frames, dim=1)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:
                    video = video[0]
                    if video.shape[0] == video_cfg["num_frames"]:
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]

        if video.min() < 0:
            video = denormalize_video(video)
        video = video.clamp(0, 1)

        return video

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings for noise prior computation."""
        import inspect

        if hasattr(self.pipe, "encode_prompt"):
            sig = inspect.signature(self.pipe.encode_prompt)
            params = sig.parameters
            kwargs = {"prompt": prompt}
            if "device" in params:
                kwargs["device"] = self.device
            if "num_videos_per_prompt" in params:
                kwargs["num_videos_per_prompt"] = 1
            if "do_classifier_free_guidance" in params:
                kwargs["do_classifier_free_guidance"] = False

            result = self.pipe.encode_prompt(**kwargs)
            if isinstance(result, tuple):
                return result[0]
            return result
        else:
            text_inputs = self.pipe.tokenizer(
                prompt, padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            return self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
