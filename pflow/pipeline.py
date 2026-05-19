"""
Main P-Flow Pipeline (Paper-Faithful Implementation).

Implements Algorithm 1 from Appendix A of the paper (arXiv:2603.22091):
1. Load and encode reference video
2. Perform Noise Prior Enhancement (Flow Inversion → SVD Filter → Blend)
3. Fixed i_max=10 iterations of Test-Time Prompt Optimization
4. Output ALL generated videos (best selected offline by VBench/FVD)

Key differences from previous implementation:
- NO confidence score
- NO early stopping (fixed i_max iterations)
- Vertical composite layout (top/middle/bottom)
- Paper VLM output format: {analysis: {4 sub-fields}, refined_prompt}
- Local Wan 2.1-T2V-1.3B model on AutoDL

Reference: Sections 3.1-3.6 and Algorithm 1 of the paper.
"""

import os
import json
import time
import yaml
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from .noise_prior import NoisePriorEnhancement
from .prompt_optimizer import PromptOptimizer
from .trajectory import TrajectoryManager
from .vlm_client import VLMClient, MockVLMClient
from .video_utils import (
    load_video,
    save_video_tensor,
    normalize_video,
    denormalize_video,
)
from .flow_matching import encode_video_to_latents, decode_latents_to_video


class PFlowPipeline:
    """
    P-Flow: Training-Free Framework for Customizing Dynamic Visual Effects
    via Test-Time Prompt Optimization.

    Algorithm 1 (Paper-Faithful):

    Input: Reference video V_ref, user prompt P_user, video model G
    Output: All generated videos V_1...V_{i_max} for offline evaluation

    1. Encode V_ref to latent space
    2. Perform flow matching inversion → η_inv
    3. Apply SVD filtering → η_temporal
    4. Blend with random noise → η (noise prior)
    5. Initialize prompt P_0 = P_user
    6. For i = 1 to i_max (FIXED, no early stopping):
       a. Generate V_i = G(P_i, η)
       b. Create vertical composite [V_ref | V_{i-1} | V_i]
       c. Send to VLM with history (Listing 1 format)
       d. Get refined prompt P_{i+1} and analysis A_i
       e. Update trajectory with {V_i, P_i, A_i}
    7. Return all {V_i, P_i, A_i} for offline best selection
    """

    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/models/Wan2.1-T2V-1.3B",
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_mock_vlm: bool = False,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_model: Optional[str] = None,
    ):
        """
        Initialize the P-Flow pipeline.

        Args:
            model_path: Local path to Wan 2.1-T2V-1.3B model weights.
            config_path: Path to YAML config file.
            config: Config dictionary (overrides config_path).
            device: Computation device.
            dtype: Model dtype.
            use_mock_vlm: Use mock VLM for testing.
            vlm_api_key: DashScope API key for VLM (Qwen3-VL-Flash).
            vlm_base_url: DashScope API base URL.
            vlm_model: VLM model name override.
        """
        self.device = device
        self.dtype = dtype
        self.use_mock_vlm = use_mock_vlm

        # Load configuration
        self.config = self._load_config(config_path, config)

        # Initialize video generation pipeline (lazy loading)
        self._pipe = None
        self._model_path = model_path

        # Initialize VLM client
        if use_mock_vlm:
            self.vlm_client = MockVLMClient()
        else:
            self.vlm_client = VLMClient(
                model_name=vlm_model or self.config["vlm"]["model_name"],
                api_key=vlm_api_key,
                base_url=vlm_base_url,
                temperature=self.config["vlm"]["temperature"],
                max_tokens=self.config["vlm"]["max_tokens"],
            )

    @property
    def pipe(self):
        """Lazy-load the video generation pipeline."""
        if self._pipe is None:
            self._pipe = self._load_model()
        return self._pipe

    def _load_config(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        default_config = {
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
            },
            "optimization": {
                "i_max": 3,  # Fixed iterations, NO early stopping (paper uses 10, reduced for 1.3B)
            },
            "vlm": {
                "model_name": "qwen-vl-max",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "trajectory": {
                "max_videos_to_vlm": 3,  # V_ref, V_{i-1}, V_i
            },
            "output": {
                "save_all_videos": True,
                "save_composites": True,
                "output_dir": "/root/autodl-tmp/outputs",
            },
        }

        if config is not None:
            self._deep_merge(default_config, config)
            return default_config

        if config_path is not None and os.path.exists(config_path):
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            self._deep_merge(default_config, file_config)

        return default_config

    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _load_model(self):
        """
        Load the Wan 2.1-T2V-1.3B model from local path.

        Returns:
            Loaded diffusers pipeline.
        """
        print(f"Loading model from: {self._model_path}...")

        try:
            from diffusers import WanPipeline

            pipe = WanPipeline.from_pretrained(
                self._model_path,
                torch_dtype=self.dtype,
            )
            pipe = pipe.to(self.device)

            # Enable memory optimizations for 4090
            pipe.enable_model_cpu_offload()

        except ImportError:
            from diffusers import AutoPipelineForText2Video

            pipe = AutoPipelineForText2Video.from_pretrained(
                self._model_path,
                torch_dtype=self.dtype,
            )
            pipe = pipe.to(self.device)

        print("Model loaded successfully.")
        return pipe

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
    ) -> Dict[str, Any]:
        """
        Run the complete P-Flow pipeline (Algorithm 1).

        IMPORTANT: Runs FIXED i_max iterations. No early stopping.
        Best video is selected offline (not by confidence).

        Args:
            reference_video_path: Path to the reference video.
            prompt: User's text prompt describing desired effect.
            output_dir: Output directory (overrides config).
            seed: Random seed for reproducibility.
            desired_visual_effect: Description of target effect for VLM.
            subject: Main subject for VLM structured input.
            environment: Scene environment for VLM structured input.

        Returns:
            Dictionary with all iteration results for offline evaluation.
        """
        # Setup
        output_dir = output_dir or self.config["output"]["output_dir"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None

        i_max = self.config["optimization"]["i_max"]

        print("=" * 60)
        print("P-Flow: Test-Time Prompt Optimization (Paper-Faithful)")
        print("=" * 60)
        print(f"Reference video: {reference_video_path}")
        print(f"Initial prompt: {prompt}")
        print(f"Fixed iterations: {i_max} (NO early stopping)")
        print(f"Model: {self._model_path}")
        print(f"Output dir: {output_dir}")
        print("=" * 60)

        # Step 1: Load and encode reference video
        print("\n[Step 1] Loading reference video...")
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
        print("[Step 1] Encoding reference video to latent space...")
        ref_normalized = normalize_video(reference_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_normalized, self.device)

        # Step 2: Noise Prior Enhancement
        print("\n[Step 2] Computing Noise Prior Enhancement...")
        noise_enhancer = NoisePriorEnhancement(
            pipe=self.pipe,
            alpha=self.config["noise_prior"]["alpha"],
            rho_s=self.config["noise_prior"]["rho_s"],
            rho_m=self.config["noise_prior"]["rho_m"],
            num_inversion_steps=self.config["noise_prior"]["inversion_steps"],
            device=self.device,
        )

        prompt_embeds_inv = self._encode_prompt("")
        enhanced_noise = noise_enhancer.enhance(
            video_latents=ref_latents,
            prompt_embeds=prompt_embeds_inv,
            negative_prompt_embeds=prompt_embeds_inv,
            generator=generator,
        )
        print(f"  Noise prior shape: {enhanced_noise.shape}")
        print(f"  Noise prior stats: mean={enhanced_noise.mean():.4f}, std={enhanced_noise.std():.4f}")

        # Step 3: Initialize trajectory and prompt optimizer
        print("\n[Step 3] Initializing optimization components...")
        trajectory = TrajectoryManager(
            output_dir=str(output_path),
            save_all_videos=self.config["output"]["save_all_videos"],
        )
        trajectory.set_reference(reference_video, ref_save_path)

        prompt_optimizer = PromptOptimizer(
            vlm_client=self.vlm_client,
            max_iterations=i_max,
            output_dir=str(output_path),
        )

        # Step 4: FIXED Iteration Loop (NO early stopping)
        print(f"\n[Step 4] Starting optimization loop ({i_max} fixed iterations)...")

        current_prompt = prompt
        last_prompt = ""

        for iteration in range(1, i_max + 1):
            iter_start_time = time.time()

            print(f"\n{'='*50}")
            print(f"  Iteration {iteration}/{i_max}")
            print(f"  Prompt: {current_prompt[:100]}...")
            print(f"{'='*50}")

            # 4a: Generate video with current prompt and enhanced noise
            print(f"  [4a] Generating video...")
            generated_video = self._generate_video(
                prompt=current_prompt,
                latents=enhanced_noise,
                generator=generator,
            )

            # Save generated video
            iter_video_path = str(output_path / f"generated_iter_{iteration:03d}.mp4")
            save_video_tensor(generated_video, iter_video_path, fps=self.config["video"]["fps"])

            # 4b-4d: Send to VLM for analysis and get refined prompt
            print(f"  [4b-4d] Analyzing with VLM...")

            # Get previous video for 3-panel vertical composite
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

            # 4e: Update trajectory
            trajectory.add_entry(
                iteration=iteration,
                prompt=current_prompt,
                video=generated_video,
                video_path=iter_video_path,
                analysis=optimization_result.get("analysis", {}),
                refined_prompt=optimization_result.get("refined_prompt", ""),
            )

            # Print analysis summary
            analysis = optimization_result.get("analysis", {})
            if isinstance(analysis, dict):
                comparison = analysis.get("comparison", "")[:150]
            else:
                comparison = str(analysis)[:150]
            print(f"  Comparison: {comparison}")

            iter_time = time.time() - iter_start_time
            print(f"  Iteration time: {iter_time:.1f}s")

            # Update prompt for next iteration (NO convergence check)
            last_prompt = current_prompt
            refined_prompt = optimization_result.get("refined_prompt", "")
            if refined_prompt and refined_prompt.strip():
                current_prompt = refined_prompt
            else:
                print("  Warning: VLM did not return refined prompt, keeping current.")

        # Step 5: Save all results (best selected offline)
        print("\n[Step 5] Saving final results...")
        print("  NOTE: Paper selects best video offline using VBench/FVD metrics.")
        print("  All generated videos are saved for offline evaluation.")

        # Save complete trajectory
        trajectory_data = trajectory.get_full_trajectory()
        trajectory_path = str(output_path / "full_trajectory.json")
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        # Save prompts history
        all_prompts = trajectory.get_all_prompts()
        prompts_path = str(output_path / "prompts_history.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump({
                "initial_prompt": prompt,
                "final_prompt": current_prompt,
                "all_prompts": all_prompts,
                "total_iterations": i_max,
                "note": "Best video selected offline by VBench/FVD (not by confidence)",
            }, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("P-Flow Optimization Complete!")
        print(f"  Total iterations: {i_max} (fixed)")
        print(f"  Videos saved: {output_path}/generated_iter_*.mp4")
        print(f"  Final prompt: {current_prompt[:100]}...")
        print(f"  Trajectory: {trajectory_path}")
        print(f"  NOTE: Run offline evaluation (VBench/FVD) to select best video.")
        print("=" * 60)

        return {
            "output_dir": str(output_path),
            "initial_prompt": prompt,
            "final_prompt": current_prompt,
            "all_prompts": all_prompts,
            "num_iterations": i_max,
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
    ) -> torch.Tensor:
        """
        Generate a video using the local Wan 2.1-T2V-1.3B model.

        Args:
            prompt: Text prompt for generation.
            latents: Enhanced noise prior (from noise prior enhancement).
            generator: Random generator.

        Returns:
            Generated video tensor (C, F, H, W) in [0, 1].
        """
        output = self.pipe(
            prompt=prompt,
            height=self.config["video"]["height"],
            width=self.config["video"]["width"],
            num_frames=self.config["video"]["num_frames"],
            guidance_scale=self.config["video"]["guidance_scale"],
            num_inference_steps=self.config["video"]["num_inference_steps"],
            latents=latents,
            generator=generator,
            output_type="pt",
        )

        # Handle different output formats from diffusers
        if hasattr(output, "frames"):
            video = output.frames
            if isinstance(video, list):
                import torchvision.transforms as T
                transform = T.ToTensor()
                frames = [transform(f) for f in video[0]]
                video = torch.stack(frames, dim=1)  # (C, F, H, W)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:  # (B, F, C, H, W) or (B, C, F, H, W)
                    video = video[0]
                    if video.shape[0] == self.config["video"]["num_frames"]:
                        video = video.permute(1, 0, 2, 3)  # (F, C, H, W) -> (C, F, H, W)
        else:
            video = output[0]

        # Ensure [0, 1] range
        if video.min() < 0:
            video = denormalize_video(video)
        video = video.clamp(0, 1)

        return video

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt to embeddings (compatible with Wan Pipeline)."""
        import inspect

        if hasattr(self.pipe, "encode_prompt"):
            # Inspect the signature to pass only supported parameters
            sig = inspect.signature(self.pipe.encode_prompt)
            params = sig.parameters

            kwargs = {"prompt": prompt}

            # Wan Pipeline uses 'device' but some versions don't
            if "device" in params:
                kwargs["device"] = self.device
            # SD-style pipelines use num_images_per_prompt
            if "num_images_per_prompt" in params:
                kwargs["num_images_per_prompt"] = 1
            # Wan Pipeline uses num_videos_per_prompt
            if "num_videos_per_prompt" in params:
                kwargs["num_videos_per_prompt"] = 1
            # Some pipelines accept do_classifier_free_guidance
            if "do_classifier_free_guidance" in params:
                kwargs["do_classifier_free_guidance"] = False

            prompt_embeds = self.pipe.encode_prompt(**kwargs)
            if isinstance(prompt_embeds, tuple):
                prompt_embeds = prompt_embeds[0]
        else:
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            prompt_embeds = self.pipe.text_encoder(text_input_ids)[0]

        return prompt_embeds

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

        prompt_embeds = self._encode_prompt("")
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
