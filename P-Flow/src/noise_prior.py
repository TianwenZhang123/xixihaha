"""
Noise Prior Enhancement for P-Flow (Wan2.1-1.3B).

Complete pipeline (Section 3.3):
    V_ref → VAE Encode → x_1 → Flow Inversion → η_inv
    η_inv → SVD Spatial Filter → η_spatial
    η_spatial → SVD Temporal Retain → η_temporal
    η_temporal + η_new → Blending → η (final noise prior)

Final equation (Eq. 7):
    η = √α · η_temporal + √(1-α) · η_new

where α = 0.001 (mostly random, with subtle motion hint).

Important: compute_temporal_prior() is called ONCE.
The blending with η_new happens PER ITERATION in the main loop.
"""

import torch
from typing import Optional
import logging

from .flow_matching import FlowMatchingInverter, encode_video_to_latents
from .svd_filter import SVDFilter

logger = logging.getLogger(__name__)


class NoisePriorEnhancement:
    """
    Complete Noise Prior Enhancement pipeline for Wan 2.1-1.3B.

    Pipeline:
        V_ref → VAE → x_1 → Inversion → η_inv → SVD → η_temporal
        Per-iteration: η = √α · η_temporal + √(1-α) · η_new
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
            pipe: Wan 2.1-1.3B pipeline.
            alpha: Blending weight (Eq. 7). Paper: 0.001.
            rho_s: Spatial SVD threshold. Paper: 0.1.
            rho_m: Temporal SVD threshold. Paper: 0.9.
            num_inversion_steps: ODE steps for inversion.
            device: Primary compute device.
            use_efficient_svd: Use randomized SVD for memory efficiency.
        """
        self.pipe = pipe
        self.alpha = alpha
        self.rho_s = rho_s
        self.rho_m = rho_m
        self.device = device
        self.use_efficient_svd = use_efficient_svd

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
        Compute η_temporal (Algorithm 1, lines 2-3). Called ONCE.

        Steps:
            1. Flow Matching Inversion: x_1 → η_inv
            2. Two-stage SVD: η_inv → η_temporal

        The per-iteration blending (Eq. 7) happens in the main loop.

        Args:
            video_latents: Reference video in latent space (B, C, F, H, W).
            prompt_embeds: Text embeddings for P_0.
            negative_prompt_embeds: Negative embeddings.

        Returns:
            η_temporal (B, C, F, H, W).
        """
        logger.info("Computing noise prior (inversion + SVD filtering)...")

        # Step 1: Flow Matching Inversion
        eta_inv = self.inverter.invert(
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        logger.info(f"  η_inv: shape={eta_inv.shape}, mean={eta_inv.mean():.4f}, std={eta_inv.std():.4f}")

        # Step 2: Two-stage SVD Projection
        if self.use_efficient_svd:
            eta_temporal = self.svd_filter.filter_efficient(eta_inv)
        else:
            eta_temporal = self.svd_filter.filter(eta_inv)

        logger.info(f"  η_temporal: shape={eta_temporal.shape}, mean={eta_temporal.mean():.4f}, std={eta_temporal.std():.4f}")

        return eta_temporal

    @torch.no_grad()
    def blend_noise(
        self,
        eta_temporal: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Blend temporal prior with fresh random noise (Eq. 7).

        η = √α · η_temporal + √(1-α) · η_new

        Called once PER ITERATION to provide exploration while preserving motion.

        Args:
            eta_temporal: Temporal noise prior.
            generator: Random generator for reproducibility.

        Returns:
            Blended noise η.
        """
        eta_new = torch.randn(
            eta_temporal.shape, dtype=eta_temporal.dtype,
            device=eta_temporal.device, generator=generator
        )

        eta = (
            torch.sqrt(torch.tensor(self.alpha, device=self.device)) * eta_temporal
            + torch.sqrt(torch.tensor(1.0 - self.alpha, device=self.device)) * eta_new
        )

        return eta

    @torch.no_grad()
    def enhance(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Complete noise prior enhancement in one call (legacy interface).
        For paper-faithful usage, prefer compute_temporal_prior() + blend_noise().
        """
        eta_temporal = self.compute_temporal_prior(
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        return self.blend_noise(eta_temporal, generator)
