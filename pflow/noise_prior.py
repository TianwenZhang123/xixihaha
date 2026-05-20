"""
Noise Prior Enhancement Module for P-Flow.

This module implements the complete noise prior enhancement pipeline:
1. Flow matching inversion of reference video → η_inv
2. SVD spatial filtering (remove content, keep structure)
3. SVD temporal retention (preserve motion dynamics)
4. Noise blending with fresh random noise

Final equation (Eq. 10):
    η = √α · η_temporal + √(1-α) · η_new
    
where α = 0.001 (very small, mostly random noise but with motion hint).

Reference: Section 3.3 of the paper.
"""

import torch
from typing import Optional, Tuple

from .flow_matching import FlowMatchingInverter, encode_video_to_latents
from .svd_filter import SVDFilter


class NoisePriorEnhancement:
    """
    Complete Noise Prior Enhancement pipeline.
    
    Pipeline flow:
        V_ref → VAE Encode → x_1 → Flow Inversion → η_inv
        η_inv → SVD Spatial Filter → η_spatial
        η_spatial → SVD Temporal Retain → η_temporal
        η_temporal + η_new → Blending → η (final noise prior)
    """
    
    def __init__(
        self,
        pipe,
        alpha: float = 0.001,
        rho_s: float = 0.1,
        rho_m: float = 0.9,
        num_inversion_steps: int = 50,
        device: str = "cuda",
        use_efficient_svd: bool = False,
    ):
        """
        Args:
            pipe: Video generation pipeline (Wan 2.1).
            alpha: Noise blending weight (Eq. 10). Default 0.001.
            rho_s: Spatial SVD retention ratio. Default 0.1.
            rho_m: Temporal SVD retention ratio. Default 0.9.
            num_inversion_steps: Steps for flow matching inversion.
            device: Computation device.
            use_efficient_svd: Whether to use randomized SVD for memory efficiency.
        """
        self.pipe = pipe
        self.alpha = alpha
        self.rho_s = rho_s
        self.rho_m = rho_m
        self.device = device
        self.use_efficient_svd = use_efficient_svd
        
        # Initialize sub-modules
        self.inverter = FlowMatchingInverter(
            pipe=pipe,
            num_inversion_steps=num_inversion_steps,
            guidance_scale=1.0,  # No guidance during inversion
            device=device,
        )
        self.svd_filter = SVDFilter(rho_s=rho_s, rho_m=rho_m)
        
    @torch.no_grad()
    def compute_temporal_prior(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the temporal noise prior η_temporal (Algorithm 1, lines 2-3).
        
        This performs:
            1. Flow Matching Inversion: x_1 → η_inv
            2. Two-stage SVD Projection: η_inv → η_temporal
        
        The blending with random noise happens PER ITERATION in the main loop,
        NOT here. This ensures each iteration has fresh exploration noise.
        
        Args:
            video_latents: Reference video encoded to latent space (B, C, F, H, W).
            prompt_embeds: Text embeddings (P_0) for conditioning the inversion.
            negative_prompt_embeds: Negative text embeddings.
            
        Returns:
            Temporal noise prior η_temporal of shape (B, C, F, H, W).
        """
        # Step 1: Flow Matching Inversion (Algorithm 1, line 2)
        # x_1 (video latents) → x_0 (inverted noise η_inv)
        eta_inv = self.inverter.invert(
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # Step 2: Two-stage SVD Projection (Algorithm 1, line 3)
        # η_inv → η_temporal
        if self.use_efficient_svd:
            eta_temporal = self.svd_filter.filter_efficient(eta_inv)
        else:
            eta_temporal = self.svd_filter.filter(eta_inv)
        
        return eta_temporal

    @torch.no_grad()
    def enhance(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Perform the complete noise prior enhancement (legacy interface).
        
        NOTE: For paper-faithful usage, prefer compute_temporal_prior() + per-iteration
        blending in the main loop. This method performs all steps in one call.
        
        Args:
            video_latents: Reference video encoded to latent space (B, C, F, H, W).
            prompt_embeds: Text embeddings for the reference description.
            negative_prompt_embeds: Negative text embeddings.
            generator: Random number generator for reproducibility.
            
        Returns:
            Enhanced noise prior η of shape (B, C, F, H, W).
        """
        # Compute temporal prior
        eta_temporal = self.compute_temporal_prior(
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # Noise Blending (Eq. 7)
        # η = √α · η_temporal + √(1-α) · η_new
        eta_new = torch.randn(
            eta_temporal.shape, dtype=eta_temporal.dtype,
            device=eta_temporal.device, generator=generator
        )
        
        eta_enhanced = (
            torch.sqrt(torch.tensor(self.alpha, device=self.device)) * eta_temporal
            + torch.sqrt(torch.tensor(1.0 - self.alpha, device=self.device)) * eta_new
        )
        
        return eta_enhanced
    
    @torch.no_grad()
    def enhance_from_video(
        self,
        video_tensor: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Convenience method: perform enhancement starting from raw video tensor.
        
        Args:
            video_tensor: Reference video (B, C, F, H, W) in [-1, 1] or [0, 1].
            prompt_embeds: Text embeddings.
            negative_prompt_embeds: Negative text embeddings.
            generator: Random generator.
            
        Returns:
            Enhanced noise prior η.
        """
        # Normalize to [-1, 1] if needed
        if video_tensor.min() >= 0:
            video_tensor = video_tensor * 2.0 - 1.0
            
        # Encode to latent space
        video_latents = encode_video_to_latents(self.pipe, video_tensor, self.device)
        
        # Run enhancement pipeline
        return self.enhance(
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
        )
    
    def get_noise_statistics(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute statistics at each stage for analysis/debugging.
        
        Returns dict with norms and statistics at each filtering stage.
        """
        from .svd_filter import compute_svd_statistics
        
        # Inversion
        eta_inv = self.inverter.invert(
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        stats = {
            "eta_inv_norm": eta_inv.norm().item(),
            "eta_inv_mean": eta_inv.mean().item(),
            "eta_inv_std": eta_inv.std().item(),
        }
        
        # SVD statistics on inverted noise
        if eta_inv.dim() == 5:
            svd_stats = compute_svd_statistics(eta_inv[0])
        else:
            svd_stats = compute_svd_statistics(eta_inv)
        stats.update(svd_stats)
        
        # After filtering
        eta_temporal = self.svd_filter.filter(eta_inv)
        stats["eta_temporal_norm"] = eta_temporal.norm().item()
        stats["eta_temporal_mean"] = eta_temporal.mean().item()
        stats["eta_temporal_std"] = eta_temporal.std().item()
        
        return stats
