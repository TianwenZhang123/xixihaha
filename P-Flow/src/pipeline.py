"""
Main P-Flow Pipeline - Full Paper Reproduction (Algorithm 1).

Paper: arXiv:2603.22091
Target: Wan 2.1-14B on single A800 (80GB) with CPU offload

Algorithm 1:
    Input: Reference video V_ref, user prompt P_user, video model G
    Output: All generated videos V_1...V_{i_max}

    1. Encode V_ref to latent space
    2. Flow matching inversion -> eta_inv
    3. SVD filtering -> eta_temporal
    4. For i = 1 to i_max (fixed, NO early stopping):
       a. Blend: eta = sqrt(alpha)*eta_temporal + sqrt(1-alpha)*eta_new
       b. Generate V_i = G(P_i, eta)
       c. Create vertical composite [V_ref | V_{i-1} | V_i]
       d. VLM analysis -> refined prompt P_{i+1}
       e. Update trajectory
    5. Return all {V_i, P_i, A_i} for offline evaluation

Hardware: Single A800-80GB with enable_model_cpu_offload()
VLM: DashScope Qwen-VL (qwen-vl-max)
Processing: One video at a time (sequential)
"""

import os
import json
import time
import yaml
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from .distributed import load_model_single_gpu, setup_single_gpu, cleanup_gpu_memory
from .noise_prior import NoisePriorEnhancement
from .prompt_optimizer import PromptOptimizer
from .trajectory import TrajectoryManager
from .vlm_client import create_vlm_client, MockVLMClient
from .video_utils import load_video, save_video_tensor, normalize_video, denormalize_video
from .flow_matching import encode_video_to_latents

logger = logging.getLogger(__name__)


class PFlowPipeline:
    """
    P-Flow: Training-Free Visual Effects Customization via Test-Time Prompt Optimization.

    Full reproduction with:
    - Wan 2.1-14B (T2V or I2V) on single A800 with CPU offload
    - DashScope Qwen-VL (qwen-vl-max)
    - Paper-exact parameters (alpha=0.001, rho_s=0.1, rho_m=0.9, i_max=10)
    - Both T2V and I2V generation modes
    - Sequential processing: one video at a time
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_mock_vlm: bool = False,
    ):
        """
        Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML config file.
            config: Config dict (overrides config_path).
            use_mock_vlm: Use mock VLM for testing.
        """
        self.config = self._load_config(config_path, config)
        self.use_mock_vlm = use_mock_vlm

        # Lazy-loaded components
        self._pipe = None
        self._vlm_client = None

        # Setup single-GPU environment
        self.device = setup_single_gpu()
        self.dtype = getattr(torch, self.config["model"]["dtype"])

    @property
    def pipe(self):
        """Lazy-load the video generation pipeline."""
        if self._pipe is None:
            self._pipe = self._load_model()
        return self._pipe

    @property
    def vlm_client(self):
        """Lazy-load the VLM client."""
        if self._vlm_client is None:
            if self.use_mock_vlm:
                self._vlm_client = MockVLMClient()
            else:
                self._vlm_client = create_vlm_client(self.config["vlm"])
        return self._vlm_client

    def _load_config(self, config_path, config) -> Dict[str, Any]:
        """Load configuration with defaults."""
        default_config = {
            "model": {
                "t2v_path": "/root/autodl-tmp/models/Wan2.1-T2V-14B-Diffusers",
                "i2v_path": "/root/autodl-tmp/models/Wan2.1-I2V-14B-Diffusers",
                "dtype": "bfloat16",
            },
            "video": {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
            },
            "noise_prior": {
                "alpha": 0.001,
                "rho_s": 0.1,
                "rho_m": 0.9,
                "inversion_steps": 50,
                "guidance_scale": 1.0,
            },
            "optimization": {
                "i_max": 10,
            },
            "vlm": {
                "provider": "dashscope",
                "model_name": "qwen-vl-max",
                "api_key_env": "DASHSCOPE_API_KEY",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "trajectory": {
                "max_videos_to_vlm": 3,
                "keep_all_text_history": True,
            },
            "output": {
                "base_dir": "/data/outputs/pflow",
                "save_all_iterations": True,
                "save_composites": True,
            },
        }

        if config is not None:
            self._deep_merge(default_config, config)
            return default_config

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            self._deep_merge(default_config, file_config)

        return default_config

    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _load_model(self):
        """Load Wan 2.1-14B on single A800 with CPU offload."""
        model_path = self.config["model"]["t2v_path"]

        if self.device == "cpu":
            logger.warning("Loading model on CPU (testing only)")
            from diffusers import WanPipeline
            return WanPipeline.from_pretrained(model_path, torch_dtype=self.dtype)

        return load_model_single_gpu(
            model_path=model_path,
            dtype=self.dtype,
            model_type="t2v",
        )

    @torch.no_grad()
    def run(
        self,
        reference_video_path: str,
        prompt: str,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        desired_visual_effect: str = "",
        subject: str = "",
        environment: str = "",
        mode: str = "t2v",
        reference_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete P-Flow pipeline (Algorithm 1).

        Fixed i_max iterations. No early stopping.
        Best video selected offline by evaluation metrics.

        Args:
            reference_video_path: Path to reference video.
            prompt: User's text prompt.
            output_dir: Output directory.
            seed: Random seed.
            desired_visual_effect: Target effect description.
            subject: Main subject.
            environment: Scene environment.
            mode: "t2v" for text-to-video, "i2v" for image-to-video.
            reference_image_path: First frame for I2V mode.

        Returns:
            Dict with all iteration results.
        """
        # Setup
        output_dir = output_dir or self.config["output"]["base_dir"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None

        i_max = self.config["optimization"]["i_max"]
        alpha = self.config["noise_prior"]["alpha"]

        logger.info("=" * 60)
        logger.info("P-Flow: Test-Time Prompt Optimization (Full Reproduction)")
        logger.info("=" * 60)
        logger.info(f"Reference video: {reference_video_path}")
        logger.info(f"Initial prompt: {prompt}")
        logger.info(f"Iterations: {i_max} (fixed, NO early stopping)")
        logger.info(f"Mode: {mode}")
        logger.info(f"Model: Wan 2.1-14B (single GPU + CPU offload)")
        logger.info(f"VLM: {self.config['vlm']['model_name']}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        total_start = time.time()

        # Step 1: Load and encode reference video
        logger.info("[Step 1] Loading reference video...")
        reference_video = load_video(
            reference_video_path,
            num_frames=self.config["video"]["num_frames"],
            height=self.config["video"]["height"],
            width=self.config["video"]["width"],
            device=self.device,
        )

        ref_save_path = str(output_path / "reference.mp4")
        save_video_tensor(reference_video, ref_save_path, fps=self.config["video"]["fps"])

        # Encode to latent space
        logger.info("[Step 1] Encoding to latent space...")
        ref_normalized = normalize_video(reference_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_normalized, self.device)

        # Step 2: Noise Prior Enhancement (computed ONCE)
        logger.info("[Step 2] Computing Noise Prior Enhancement...")
        noise_enhancer = NoisePriorEnhancement(
            pipe=self.pipe,
            alpha=alpha,
            rho_s=self.config["noise_prior"]["rho_s"],
            rho_m=self.config["noise_prior"]["rho_m"],
            num_inversion_steps=self.config["noise_prior"]["inversion_steps"],
            device=self.device,
            use_efficient_svd=(self.config["video"]["height"] * self.config["video"]["width"] > 480 * 832),
        )

        prompt_embeds_inv = self._encode_prompt(prompt)
        eta_temporal = noise_enhancer.compute_temporal_prior(
            video_latents=ref_latents,
            prompt_embeds=prompt_embeds_inv,
            negative_prompt_embeds=prompt_embeds_inv,
        )
        logger.info(f"  eta_temporal: mean={eta_temporal.mean():.4f}, std={eta_temporal.std():.4f}")

        # Step 3: Initialize components
        logger.info("[Step 3] Initializing optimization...")
        trajectory = TrajectoryManager(
            output_dir=str(output_path),
            save_all_videos=self.config["output"]["save_all_iterations"],
        )
        trajectory.set_reference(reference_video, ref_save_path)

        prompt_optimizer = PromptOptimizer(
            vlm_client=self.vlm_client,
            max_iterations=i_max,
            output_dir=str(output_path),
        )

        # Step 4: Fixed Iteration Loop
        logger.info(f"[Step 4] Optimization loop ({i_max} iterations)...")

        current_prompt = prompt
        last_prompt = ""
        timing_stats = []

        for iteration in range(1, i_max + 1):
            iter_start = time.time()

            logger.info(f"\n--- Iteration {iteration}/{i_max} ---")
            logger.info(f"  Prompt: {current_prompt[:100]}...")

            # 4a: Blend noise (Eq. 7)
            eta = noise_enhancer.blend_noise(eta_temporal, generator)

            # 4b: Generate video
            gen_start = time.time()
            generated_video = self._generate_video(
                prompt=current_prompt,
                latents=eta,
                generator=generator,
                mode=mode,
                reference_image_path=reference_image_path,
            )
            gen_time = time.time() - gen_start
            logger.info(f"  Generation time: {gen_time:.1f}s")

            # Save generated video
            iter_video_path = str(output_path / f"generated_iter_{iteration:03d}.mp4")
            save_video_tensor(generated_video, iter_video_path, fps=self.config["video"]["fps"])

            # 4c-4d: VLM analysis and prompt refinement
            vlm_start = time.time()
            prev_video = trajectory.get_previous_video()
            optimization_result = prompt_optimizer.optimize_prompt(
                current_prompt=current_prompt,
                reference_video=reference_video,
                generated_video=generated_video,
                iteration=iteration,
                desired_visual_effect=desired_visual_effect or prompt,
                subject=subject,
                environment=environment,
                last_text_prompt=last_prompt,
                previous_video=prev_video,
                history=trajectory.get_text_history(),
            )
            vlm_time = time.time() - vlm_start
            logger.info(f"  VLM time: {vlm_time:.1f}s")

            # 4e: Update trajectory
            trajectory.add_entry(
                iteration=iteration,
                prompt=current_prompt,
                video=generated_video,
                video_path=iter_video_path,
                analysis=optimization_result.get("analysis", {}),
                refined_prompt=optimization_result.get("refined_prompt", ""),
            )

            # Timing stats
            iter_time = time.time() - iter_start
            timing_stats.append({
                "iteration": iteration,
                "total_time": iter_time,
                "generation_time": gen_time,
                "vlm_time": vlm_time,
            })
            logger.info(f"  Iteration total: {iter_time:.1f}s")

            # Update prompt (no convergence check)
            last_prompt = current_prompt
            refined = optimization_result.get("refined_prompt", "")
            if refined and refined.strip():
                current_prompt = refined

        # Step 5: Save all results
        total_time = time.time() - total_start
        logger.info(f"\n[Step 5] Complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

        # Save trajectory
        trajectory_data = trajectory.get_full_trajectory()
        with open(output_path / "full_trajectory.json", "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        # Save timing and metadata
        metadata = {
            "initial_prompt": prompt,
            "final_prompt": current_prompt,
            "all_prompts": trajectory.get_all_prompts(),
            "total_iterations": i_max,
            "total_time_seconds": total_time,
            "timing_stats": timing_stats,
            "config": self.config,
            "note": "Best video selected offline by FID-VID/FVD metrics",
        }
        with open(output_path / "experiment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"  Videos saved: {output_path}/generated_iter_*.mp4")
        logger.info(f"  Run evaluation scripts to select best iteration.")

        return {
            "output_dir": str(output_path),
            "initial_prompt": prompt,
            "final_prompt": current_prompt,
            "all_prompts": trajectory.get_all_prompts(),
            "num_iterations": i_max,
            "total_time": total_time,
            "trajectory": trajectory_data,
            "video_paths": [
                str(output_path / f"generated_iter_{i:03d}.mp4")
                for i in range(1, i_max + 1)
            ],
        }

    def _generate_video(
        self,
        prompt: str,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        mode: str = "t2v",
        reference_image_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate video using Wan 2.1-14B.

        Supports both T2V and I2V modes.
        """
        kwargs = {
            "prompt": prompt,
            "height": self.config["video"]["height"],
            "width": self.config["video"]["width"],
            "num_frames": self.config["video"]["num_frames"],
            "guidance_scale": self.config["video"]["guidance_scale"],
            "num_inference_steps": self.config["video"]["num_inference_steps"],
            "generator": generator,
            "output_type": "pt",
        }

        if latents is not None:
            kwargs["latents"] = latents

        if mode == "i2v" and reference_image_path:
            from PIL import Image
            ref_image = Image.open(reference_image_path).convert("RGB")
            kwargs["image"] = ref_image

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
                    if video.shape[0] == self.config["video"]["num_frames"]:
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]

        if video.min() < 0:
            video = denormalize_video(video)
        video = video.clamp(0, 1)

        return video

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
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

    def run_noise_prior_only(
        self,
        reference_video_path: str,
        prompt: str,
        output_path: str = "output_noise_prior.mp4",
        seed: Optional[int] = None,
    ) -> str:
        """
        Run only noise prior enhancement (no prompt optimization).
        Useful for ablation experiments.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        reference_video = load_video(
            reference_video_path,
            num_frames=self.config["video"]["num_frames"],
            height=self.config["video"]["height"],
            width=self.config["video"]["width"],
            device=self.device,
        )

        ref_normalized = normalize_video(reference_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_normalized, self.device)

        noise_enhancer = NoisePriorEnhancement(
            pipe=self.pipe,
            alpha=self.config["noise_prior"]["alpha"],
            rho_s=self.config["noise_prior"]["rho_s"],
            rho_m=self.config["noise_prior"]["rho_m"],
            num_inversion_steps=self.config["noise_prior"]["inversion_steps"],
            device=self.device,
        )

        prompt_embeds = self._encode_prompt(prompt)
        enhanced_noise = noise_enhancer.enhance(
            video_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=prompt_embeds,
            generator=generator,
        )

        video = self._generate_video(prompt, enhanced_noise, generator)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_video_tensor(video, output_path, fps=self.config["video"]["fps"])

        return output_path
