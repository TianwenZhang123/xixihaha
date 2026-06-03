"""
Lightweight Velocity Field Matching for P-Flow.

Simplified from VMAD's Position-Aware Velocity Matcher:
    - Removed content disentanglement (not needed for reproduction mode)
    - Position-aware gradient scaling: optional (--position_aware flag, default off)
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
    trajectory defined by (z₀, η_inv). Supports optional position-aware
    gradient scaling to concentrate optimization effort at high-influence
    positions (attention sinks in DiT cross-attention).

    Usage:
        matcher = VelocityMatcher(pipe=pipe, device="cuda")
        result = matcher.optimize(z0=z0, e0=e0, eta_inv=eta_inv)
        delta_e = result["delta_e"]  # shape: (1, L, D)

        # With position-aware gradient scaling:
        matcher = VelocityMatcher(pipe=pipe, position_aware=True, device="cuda")
        result = matcher.optimize(z0=z0, e0=e0, eta_inv=eta_inv)
    """

    def __init__(
        self,
        pipe,
        T_m: float = 1.0,
        num_opt_steps: int = 30,
        lr: float = 1e-3,
        position_aware: bool = False,
        lambda_pos: float = 0.01,
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
            position_aware: Enable position-aware gradient scaling.
                If True, amplifies gradients at high-influence positions (position 0
                and last position in T5 embedding) based on empirically observed
                U-shape attention weight distribution in DiT cross-attention.
                Default False to preserve existing behavior.
            lambda_pos: Weight for position-aware regularization loss.
                Only used when position_aware=True. Default 0.01.
            device: Compute device.
        """
        self.pipe = pipe
        self.T_m = T_m
        self.num_opt_steps = num_opt_steps
        self.lr = lr
        self.position_aware = position_aware
        self.lambda_pos = lambda_pos
        self.device = device

    def _initialize_position_weights(self, seq_len: int) -> torch.Tensor:
        """
        Initialize position-aware weight profile based on U-shape attention pattern.

        DiT cross-attention with T5 relative position bias exhibits:
        - Position 0: highest attention weight (10-15× interior) — "attention sink"
        - Last position: moderately elevated (U-shape tail)
        - Interior: relatively uniform, slight U-shape decay toward center

        Returns an INVERSE weighting for regularization:
        - High-influence positions (pos 0) → low regularization weight → more freedom
        - Low-influence positions (center) → higher regularization → discourage waste

        Args:
            seq_len: Text embedding sequence length

        Returns:
            Position weight tensor of shape (seq_len,)
        """
        weights = torch.ones(seq_len, device=self.device)

        # Position 0: attention sink → low regularization = more optimization freedom
        weights[0] = 0.1

        # Last position: moderate elevation
        if seq_len > 1:
            weights[-1] = 0.3

        # Interior: U-shape (higher at edges, lower at center)
        if seq_len > 2:
            center = seq_len // 2
            for i in range(1, seq_len - 1):
                dist_from_edge = min(i, seq_len - 1 - i)
                weights[i] = 0.5 + 0.5 * (dist_from_edge / center)

        return weights

    def _compute_position_regularization(
        self, delta_e: torch.Tensor, position_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute position-aware regularization loss.

        Penalizes Δe magnitude at low-influence positions while allowing
        high-influence positions more freedom.

        L_pos = Σ_j w_j · ||Δe[:,j,:]||²
        """
        # delta_e shape: (B, L, D) or (L, D)
        if delta_e.dim() == 3:
            per_position_energy = (delta_e ** 2).mean(dim=(0, 2))  # (L,)
        else:
            per_position_energy = (delta_e ** 2).mean(dim=-1)  # (L,)
        loss_pos = (position_weights * per_position_energy).mean()
        return loss_pos

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

        # Initialize position weights (only used when position_aware=True)
        position_weights = None
        if self.position_aware:
            seq_len = e0.shape[1]
            position_weights = self._initialize_position_weights(seq_len)

        loss_history = []
        final_loss = 0.0

        logger.info(
            f"  [VelocityMatch] Lightweight optimization: "
            f"steps={self.num_opt_steps}, T_m={self.T_m}, lr={self.lr}, "
            f"position_aware={self.position_aware}"
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
            loss_vel = ((v_pred - v_star) ** 2).mean()

            # Position-aware regularization (optional)
            loss_pos = torch.tensor(0.0, device=self.device)
            if self.position_aware and position_weights is not None and self.lambda_pos > 0:
                loss_pos = self._compute_position_regularization(
                    delta_e.unsqueeze(0) if delta_e.dim() == 2 else delta_e,
                    position_weights,
                )

            # Total loss
            if self.position_aware and self.lambda_pos > 0:
                loss = loss_vel + self.lambda_pos * loss_pos
            else:
                loss = loss_vel

            # Backward
            loss.backward()

            # Position-aware gradient scaling (amplify high-influence positions)
            if self.position_aware and delta_e.grad is not None and position_weights is not None:
                grad_scale = 1.0 / (position_weights + 0.1)
                grad_scale = grad_scale / grad_scale.mean()  # Normalize
                if delta_e.grad.dim() == 3:
                    delta_e.grad.data *= grad_scale.unsqueeze(0).unsqueeze(-1)
                elif delta_e.grad.dim() == 2:
                    delta_e.grad.data *= grad_scale.unsqueeze(-1)

            # Update
            optimizer.step()
            scheduler.step()

            loss_val = loss_vel.item()
            loss_history.append({
                "step": step,
                "loss": loss_val,
                "loss_pos": loss_pos.item() if self.position_aware else 0.0,
                "lr": scheduler.get_last_lr()[0],
            })
            final_loss = loss_val

            if step % 10 == 0 or step == self.num_opt_steps - 1:
                logger.info(
                    f"    step {step:3d}/{self.num_opt_steps}: "
                    f"L_vel={loss_val:.6f}, "
                    f"{'L_pos=' + f'{loss_pos.item():.6f}, ' if self.position_aware else ''}"
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
