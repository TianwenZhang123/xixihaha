"""
Lightweight Velocity Field Matching for P-Flow.

Simplified from VMAD's Position-Aware Velocity Matcher:
    - Removed content disentanglement (not needed for reproduction mode)
    - Removed position-aware gradient scaling (simplify for speed)
    - Reduced default optimization steps (30 vs 100)
    - Reuses P-Flow's conditional inversion noise (higher quality than VMAD's unconditional)

Core Principle:
    Given reference video latent z₀ and inverted noise η_inv, find Δe such that:
        v_θ(x_t, t, e₀ + Δe) ≈ v* = z₀ - η_inv  for all t ∈ [0, T_m]

    This makes the model's ODE trajectory pass through the reference video's latent,
    capturing motion dynamics that text alone cannot express.

Computational Cost:
    30 steps × (1 forward + 1 backward) ≈ ~90 equivalent DiT forward passes.
    Combined with P-Flow's inversion (50) + generation (30) = ~170 total.
    This is ~2x baseline P-Flow, vs VMAD's ~11x.

Reference:
    - Reenact Anything (SIGGRAPH 2025): Motion-textual inversion
    - SiD-DiT (Apple 2025): Velocity distillation in flow matching
"""

import logging
import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VelocityMatcher:
    """
    Lightweight Velocity Field Matching for P-Flow.

    Optimizes Δe to align the model's velocity field with the ground-truth
    trajectory defined by (z₀, η_inv). No bells and whistles — just the
    core velocity matching loss with cosine-annealed Adam.

    Usage:
        matcher = VelocityMatcher(pipe=pipe, device="cuda")
        result = matcher.optimize(z0=z0, e0=e0, eta_inv=eta_inv)
        delta_e = result["delta_e"]  # shape: (1, L, D)
    """

    def __init__(
        self,
        pipe,
        T_m: float = 1.0,
        num_opt_steps: int = 30,
        lr: float = 1e-3,
        device: str = "cuda",
    ):
        """
        Args:
            pipe: Diffusers pipeline (with transformer or unet attribute).
            T_m: Time range upper bound for optimization.
                 1.0 = full reproduction, 0.3 = motion-only (for transfer).
            num_opt_steps: Number of optimization iterations (default 30,
                          sufficient when starting from good conditional inversion).
            lr: Peak learning rate for Adam optimizer.
            device: Compute device.
        """
        self.pipe = pipe
        self.T_m = T_m
        self.num_opt_steps = num_opt_steps
        self.lr = lr
        self.device = device

    def _get_model(self):
        """Get the denoising model (transformer or unet)."""
        if hasattr(self.pipe, "transformer"):
            return self.pipe.transformer
        elif hasattr(self.pipe, "unet"):
            return self.pipe.unet
        raise ValueError("Pipeline has neither transformer nor unet")

    def _model_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Single model forward pass (with grad for Δe)."""
        model = self._get_model()

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=latents.device, dtype=latents.dtype)

        model_output = model(
            hidden_states=latents,
            timestep=timestep.expand(latents.shape[0]),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )

        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def optimize(
        self,
        z0: torch.Tensor,
        e0: torch.Tensor,
        eta_inv: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Execute velocity field matching optimization.

        Args:
            z0: Target video latent (B, C, F, H, W).
            e0: Caption text embedding (B, L, D).
            eta_inv: Inverted noise from P-Flow's conditional inversion (B, C, F, H, W).

        Returns:
            Dictionary containing:
                - delta_e: Optimized embedding residual (B, L, D)
                - loss_history: Per-step loss values
                - final_loss: Final velocity matching loss
        """
        model = self._get_model()

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad_(False)

        # Decide mode based on latent size
        latent_frames = z0.shape[2] if z0.dim() == 5 else 0
        if latent_frames >= 13:
            model.train()
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            logger.info(f"    Model frozen (train + grad ckpt, latent_frames={latent_frames})")
        else:
            model.eval()
            logger.info(f"    Model frozen (eval mode, latent_frames={latent_frames})")

        torch.cuda.empty_cache()

        # Initialize Δe = zeros
        delta_e = torch.zeros_like(e0, requires_grad=True)
        optimizer = torch.optim.Adam([delta_e], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_opt_steps, eta_min=self.lr * 0.1
        )

        # Detach constant tensors
        z0 = z0.detach()
        eta_inv = eta_inv.detach()
        e0 = e0.detach()

        # Target velocity (constant throughout optimization)
        v_star = z0 - eta_inv

        loss_history = []
        final_loss = 0.0

        logger.info(
            f"  [VelocityMatch] Lightweight optimization: "
            f"steps={self.num_opt_steps}, T_m={self.T_m}, lr={self.lr}"
        )

        for step in range(self.num_opt_steps):
            optimizer.zero_grad()

            # Sample timestep t ~ U(0, T_m)
            t_norm = torch.rand(1, device=self.device, dtype=z0.dtype) * self.T_m
            t_model = t_norm * 1000.0  # Wan2.1 uses [0, 1000] timestep range

            # Construct x_t = (1-t)·η_inv + t·z₀
            x_t = (1 - t_norm) * eta_inv + t_norm * z0

            # Forward pass with current Δe
            e_current = e0 + delta_e
            v_pred = self._model_forward(x_t, t_model, e_current)

            # Velocity matching loss: ||v_pred - v*||²
            loss = ((v_pred - v_star) ** 2).mean()

            # Backward + update
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append({
                "step": step,
                "loss": loss_val,
                "lr": scheduler.get_last_lr()[0],
            })
            final_loss = loss_val

            if step % 10 == 0 or step == self.num_opt_steps - 1:
                logger.info(
                    f"    step {step:3d}/{self.num_opt_steps}: "
                    f"L_vel={loss_val:.6f}, "
                    f"||Δe||={delta_e.norm().item():.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Restore model state
        if latent_frames >= 13:
            if hasattr(model, "disable_gradient_checkpointing"):
                model.disable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        model.eval()

        return {
            "delta_e": delta_e.detach(),
            "loss_history": loss_history,
            "final_loss": final_loss,
        }
