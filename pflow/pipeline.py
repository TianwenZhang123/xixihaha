"""
Main P-Flow Pipeline.

This module integrates all components into the complete P-Flow framework:
1. Load and encode reference video
2. Perform Noise Prior Enhancement
3. Iterative Test-Time Prompt Optimization loop
4. Output final optimized video

Implements Algorithm 1 from Appendix A of the paper.

Reference: Sections 3.1-3.6 of the paper.
"""

import os
import json
import time
import yaml
from typing import Optional, Dict, List, Any, Union
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
    
    Algorithm (from paper's Algorithm 1):
    
    Input: Reference video V_ref, user prompt P_user, video model G
    Output: Optimized video V* matching V_ref's visual effects
    
    1. Encode V_ref to latent space
    2. Perform flow matching inversion → η_inv
    3. Apply SVD filtering → η_temporal
    4. Blend with random noise → η (noise prior)
    5. Initialize prompt P_0 = P_user
    6. For i = 1 to i_max:
       a. Generate V_i = G(P_i, η)
       b. Create composite [V_ref | V_i]
       c. Send to VLM with history
       d. Get refined prompt P_{i+1} and analysis A_i
       e. Update trajectory with {V_i, P_i, A_i}
       f. If converged: break
    7. Return V* = V_best (highest confidence)
    """
    
    def __init__(
        self,
        model_name: str = "Wan-AI/Wan2.1-T2V-14B",
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_mock_vlm: bool = False,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
    ):
        """
        Initialize the P-Flow pipeline.
        
        Args:
            model_name: HuggingFace model name for Wan 2.1.
            config_path: Path to YAML config file.
            config: Config dictionary (overrides config_path).
            device: Computation device.
            dtype: Model dtype (bfloat16 recommended for Wan 2.1).
            use_mock_vlm: Use mock VLM for testing.
            vlm_api_key: API key for VLM relay service.
            vlm_base_url: Base URL for OpenAI-compatible API relay.
        """
        self.device = device
        self.dtype = dtype
        self.use_mock_vlm = use_mock_vlm
        
        # Load configuration
        self.config = self._load_config(config_path, config)
        
        # Initialize video generation pipeline (lazy loading)
        self._pipe = None
        self._model_name = model_name
        
        # Initialize VLM client (via OpenAI-compatible relay)
        if use_mock_vlm:
            self.vlm_client = MockVLMClient()
        else:
            self.vlm_client = VLMClient(
                model_name=self.config["prompt_optimization"]["vlm_model"],
                api_key=vlm_api_key,
                base_url=vlm_base_url,
                temperature=self.config["prompt_optimization"]["temperature"],
                max_tokens=self.config["prompt_optimization"]["max_tokens"],
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
            "prompt_optimization": {
                "max_iterations": 10,
                "vlm_model": "gemini-2.0-flash",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "trajectory": {
                "max_videos_to_vlm": 3,
                "keep_all_text": True,
            },
            "output": {
                "save_intermediate": True,
                "save_prompts": True,
                "output_dir": "outputs",
            },
        }
        
        if config is not None:
            # Deep merge with defaults
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
        Load the Wan 2.1 video generation model.
        
        Returns:
            Loaded diffusers pipeline.
        """
        print(f"Loading model: {self._model_name}...")
        
        try:
            from diffusers import WanPipeline
            
            pipe = WanPipeline.from_pretrained(
                self._model_name,
                torch_dtype=self.dtype,
            )
            pipe = pipe.to(self.device)
            
            # Enable memory optimizations
            pipe.enable_model_cpu_offload()
            
        except ImportError:
            # Fallback: try AutoPipelineForText2Video
            from diffusers import AutoPipelineForText2Video
            
            pipe = AutoPipelineForText2Video.from_pretrained(
                self._model_name,
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
    ) -> Dict[str, Any]:
        """
        Run the complete P-Flow pipeline.
        
        This is the main entry point implementing Algorithm 1.
        
        Args:
            reference_video_path: Path to the reference video.
            prompt: User's text prompt describing desired effect.
            output_dir: Output directory (overrides config).
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with:
                - 'best_video_path': Path to best generated video
                - 'best_prompt': Best prompt found
                - 'all_prompts': List of all prompts tried
                - 'trajectory': Full optimization trajectory
                - 'num_iterations': Total iterations run
        """
        # Setup
        output_dir = output_dir or self.config["output"]["output_dir"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        max_iterations = self.config["prompt_optimization"]["max_iterations"]
        
        print("=" * 60)
        print("P-Flow: Test-Time Prompt Optimization Pipeline")
        print("=" * 60)
        print(f"Reference video: {reference_video_path}")
        print(f"Initial prompt: {prompt}")
        print(f"Max iterations: {max_iterations}")
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
        
        # Save reference video copy
        ref_save_path = str(output_path / "reference.mp4")
        save_video_tensor(reference_video, ref_save_path, fps=self.config["video"]["fps"])
        
        # Encode to latent space
        print("[Step 1] Encoding reference video to latent space...")
        ref_normalized = normalize_video(reference_video).unsqueeze(0)  # Add batch dim
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
        
        # Get text embeddings for inversion (use empty prompt for unconditional)
        prompt_embeds_inv = self._encode_prompt("")
        negative_prompt_embeds = self._encode_prompt("")
        
        # Compute enhanced noise prior
        enhanced_noise = noise_enhancer.enhance(
            video_latents=ref_latents,
            prompt_embeds=prompt_embeds_inv,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
        )
        print(f"  Noise prior shape: {enhanced_noise.shape}")
        print(f"  Noise prior stats: mean={enhanced_noise.mean():.4f}, std={enhanced_noise.std():.4f}")
        
        # Step 3: Initialize trajectory and prompt optimizer
        print("\n[Step 3] Initializing optimization components...")
        trajectory = TrajectoryManager(
            output_dir=str(output_path),
            save_all_videos=self.config["output"]["save_intermediate"],
        )
        trajectory.set_reference(reference_video, ref_save_path)
        
        prompt_optimizer = PromptOptimizer(
            vlm_client=self.vlm_client,
            max_iterations=max_iterations,
            output_dir=str(output_path),
        )
        
        # Step 4: Iterative Optimization Loop
        print(f"\n[Step 4] Starting optimization loop (max {max_iterations} iterations)...")
        
        current_prompt = prompt
        best_video = None
        best_prompt = prompt
        best_confidence = 0.0
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            print(f"  Prompt: {current_prompt[:100]}...")
            
            # 4a: Generate video with current prompt and enhanced noise
            print(f"  Generating video...")
            generated_video = self._generate_video(
                prompt=current_prompt,
                latents=enhanced_noise,
                generator=generator,
            )
            
            # Save intermediate video
            if self.config["output"]["save_intermediate"]:
                iter_video_path = str(output_path / f"generated_iter_{iteration:03d}.mp4")
                save_video_tensor(generated_video, iter_video_path, fps=self.config["video"]["fps"])
            
            # 4b-4d: Send to VLM for analysis and get refined prompt
            print(f"  Analyzing with VLM...")
            
            # Get previous video for 3-panel comparison
            _, prev_video, _ = trajectory.get_videos_for_vlm()
            
            # Run prompt optimization
            optimization_result = prompt_optimizer.optimize_prompt(
                initial_prompt=current_prompt,
                reference_video=reference_video,
                generated_video=generated_video,
                iteration=iteration,
                history=trajectory.get_history_for_vlm(),
                user_description=prompt,
                previous_video=prev_video,
            )
            
            # 4e: Update trajectory
            trajectory.add_entry(
                iteration=iteration,
                prompt=current_prompt,
                video=generated_video,
                analysis=optimization_result.get("analysis", ""),
                improvements=optimization_result.get("improvements", []),
                confidence=optimization_result.get("confidence", 0.0),
                key_differences=optimization_result.get("key_differences", []),
            )
            
            # Track best result
            confidence = optimization_result.get("confidence", 0.0)
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Key differences: {optimization_result.get('key_differences', [])[:3]}")
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_video = generated_video
                best_prompt = current_prompt
            
            # 4f: Check convergence
            if prompt_optimizer.should_stop_early(
                trajectory.get_history_for_vlm(),
                min_iterations=3,
            ):
                print(f"\n  ✓ Converged at iteration {iteration}!")
                break
            
            # Update prompt for next iteration
            refined_prompt = optimization_result.get("refined_prompt", "")
            if refined_prompt:
                current_prompt = refined_prompt
            else:
                print("  Warning: VLM did not return refined prompt, keeping current.")
        
        # Step 5: Save final results
        print("\n[Step 5] Saving final results...")
        
        # Save best video
        best_video_path = str(output_path / "best_result.mp4")
        if best_video is not None:
            save_video_tensor(best_video, best_video_path, fps=self.config["video"]["fps"])
        
        # Save all prompts
        all_prompts = [e.prompt for e in trajectory.entries]
        prompts_path = str(output_path / "prompts_history.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump({
                "initial_prompt": prompt,
                "best_prompt": best_prompt,
                "all_prompts": all_prompts,
                "best_confidence": best_confidence,
            }, f, indent=2, ensure_ascii=False)
        
        # Save convergence info
        convergence_info = trajectory.get_convergence_info()
        convergence_path = str(output_path / "convergence.json")
        with open(convergence_path, "w", encoding="utf-8") as f:
            json.dump(convergence_info, f, indent=2)
        
        print("\n" + "=" * 60)
        print("P-Flow Optimization Complete!")
        print(f"  Best confidence: {best_confidence:.3f}")
        print(f"  Total iterations: {len(trajectory)}")
        print(f"  Best prompt: {best_prompt[:100]}...")
        print(f"  Output: {best_video_path}")
        print("=" * 60)
        
        return {
            "best_video_path": best_video_path,
            "best_prompt": best_prompt,
            "best_confidence": best_confidence,
            "all_prompts": all_prompts,
            "trajectory": trajectory.get_history_for_vlm(),
            "num_iterations": len(trajectory),
            "convergence": convergence_info,
        }
    
    def _generate_video(
        self,
        prompt: str,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate a video using the Wan 2.1 model.
        
        Args:
            prompt: Text prompt for generation.
            latents: Optional initial noise (from noise prior enhancement).
            generator: Random generator.
            
        Returns:
            Generated video tensor (C, F, H, W) in [0, 1].
        """
        # Generate with the pipeline
        output = self.pipe(
            prompt=prompt,
            height=self.config["video"]["height"],
            width=self.config["video"]["width"],
            num_frames=self.config["video"]["num_frames"],
            guidance_scale=self.config["video"]["guidance_scale"],
            num_inference_steps=self.config["video"]["num_inference_steps"],
            latents=latents,
            generator=generator,
            output_type="pt",  # Return as PyTorch tensor
        )
        
        # The output format depends on the pipeline version
        if hasattr(output, "frames"):
            # Standard diffusers output
            video = output.frames
            if isinstance(video, list):
                # List of PIL images - convert to tensor
                import torchvision.transforms as T
                transform = T.ToTensor()
                frames = [transform(f) for f in video[0]]
                video = torch.stack(frames, dim=1)  # (C, F, H, W)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:  # (B, F, C, H, W) or (B, C, F, H, W)
                    video = video[0]
                    if video.shape[0] == self.config["video"]["num_frames"]:
                        # (F, C, H, W) -> (C, F, H, W)
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]
        
        # Ensure [0, 1] range
        if video.min() < 0:
            video = denormalize_video(video)
        video = video.clamp(0, 1)
        
        return video
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encode a text prompt to embeddings using the pipeline's text encoder.
        
        Args:
            prompt: Text string to encode.
            
        Returns:
            Prompt embeddings tensor.
        """
        # Use the pipeline's built-in text encoding
        if hasattr(self.pipe, "encode_prompt"):
            # Newer diffusers API
            prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            if isinstance(prompt_embeds, tuple):
                prompt_embeds = prompt_embeds[0]
        else:
            # Manual encoding
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
        Run only the noise prior enhancement (no prompt optimization).
        Useful for testing/ablation.
        
        Args:
            reference_video_path: Path to reference video.
            prompt: Generation prompt.
            output_path: Output video path.
            seed: Random seed.
            
        Returns:
            Path to generated video.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Load reference
        reference_video = load_video(
            reference_video_path,
            num_frames=self.config["video"]["num_frames"],
            height=self.config["video"]["height"],
            width=self.config["video"]["width"],
            device=self.device,
        )
        
        # Encode
        ref_normalized = normalize_video(reference_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_normalized, self.device)
        
        # Noise enhancement
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
        
        # Generate
        video = self._generate_video(prompt, enhanced_noise, generator)
        save_video_tensor(video, output_path, fps=self.config["video"]["fps"])
        
        return output_path
