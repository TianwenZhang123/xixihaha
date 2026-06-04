"""
Lightweight Velocity Field Matching for P-Flow (v2 — Enhanced + Optimized).

Key improvements over v1:
    1. Stratified Multi-Timestep Sampling: Sample K timesteps per step via stratified
       uniform partition → reduce gradient variance by √K without extra memory.
       (Inspired by: Adaptive Non-Uniform Timestep Sampling, CVPR 2025)

    2. Padding-Aware Gradient Mask: Zero-out gradients on padding token positions
       so that Δe optimization concentrates on semantically meaningful tokens.
       (Motivated by: Padding Tone, NAACL 2025 — padding tokens in T2I models
       can carry unintended "tonal" information through cross-attention)

    3. Motion-Aware Loss Weighting (Latent Temporal Discrepancy):
       Weight the MSE loss by per-pixel temporal variance of v*, so that
       high-motion regions (large |Δv* across frames|) get higher penalty.
       (Reference: Latent Temporal Discrepancy as Motion Prior, 2025)

    4. Norm-Aware Adaptive Injection (optional, for pipeline-side use):
       Clip ||Δe|| or adapt es so that es × ||Δe|| stays in optimal range.

Performance optimizations (v2.1):
    - Batched K-timestep forward: All K x_t packed into a single batch forward,
      reducing kernel launch overhead and enabling better GPU utilization.
    - Early stopping: If loss converges (relative change < threshold for N steps),
      optimization terminates early to save compute.
    - Adaptive K scheduling: Start with K=1 for fast directional convergence,
      switch to full K for refinement (optional).

Core Principle (unchanged):
    Given reference video latent z₀ and inverted noise η_inv, find Δe such that:
        v_θ(x_t, t, e₀ + Δe) ≈ v* = z₀ - η_inv  for all t ∈ [0, T_m]

Computational Cost (optimized):
    With early stopping + batched forward, typically 15-25 steps × 1 batched forward
    ≈ ~30-50 equivalent passes (down from ~240 in v2 without optimization).

Reference:
    - Reenact Anything (SIGGRAPH 2025): Motion-textual inversion, inflated embedding
    - Motion Inversion (SIGGRAPH 2025): Frame-to-frame debiasing for motion embeddings
    - SiD-DiT (Apple 2025): Velocity distillation in flow matching
    - Adaptive Non-Uniform Timestep Sampling (CVPR 2025): Stratified / loss-aware sampling
    - Latent Temporal Discrepancy (2025): Motion-prior loss weighting
    - Padding Tone (NAACL 2025): Padding tokens influence T2I cross-attention
"""

import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


class VelocityMatcher:
    """
    Lightweight Velocity Field Matching for P-Flow (v2.1 — optimized).

    Optimizes Δe to align the model's velocity field with the ground-truth
    trajectory defined by (z₀, η_inv).

    Usage:
        matcher = VelocityMatcher(pipe=pipe, device="cuda")
        result = matcher.optimize(z0=z0, e0=e0, eta_inv=eta_inv, token_length=180)
        delta_e = result["delta_e"]  # shape: (1, L, D)
    """

    def __init__(
        self,
        pipe,
        T_m: float = 1.0,
        num_opt_steps: int = 30,
        lr: float = 1e-3,
        num_timesteps_per_step: int = 4,
        motion_weight_strength: float = 1.0,
        early_stop_patience: int = 5,
        early_stop_threshold: float = 1e-4,
        use_batched_forward: bool = True,
        use_adaptive_K: bool = True,
        adaptive_K_warmup: int = 8,
        use_amp: bool = True,
        use_warm_start: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            pipe: Diffusers pipeline (with transformer or unet attribute).
            T_m: Time range upper bound for optimization.
                 1.0 = full reproduction, 0.3 = motion-only (for transfer).
            num_opt_steps: Maximum optimization iterations (default 30).
            lr: Peak learning rate for Adam optimizer.
            num_timesteps_per_step: Number of stratified timesteps per optimization
                step (K). Higher K → lower gradient variance, slight memory increase.
                K=4 gives ~2x variance reduction vs K=1.
            motion_weight_strength: Controls how much to emphasize motion regions.
                0.0 = uniform (no motion weighting), 1.0 = full LTD weighting.
            early_stop_patience: Stop if loss doesn't improve for this many steps.
            early_stop_threshold: Minimum relative improvement to count as progress.
            use_batched_forward: If True, batch all K timesteps into one forward pass
                (faster but uses K× memory for intermediate activations).
            use_adaptive_K: If True, use K=1 for the first `adaptive_K_warmup` steps
                then switch to full K. Reduces early compute by ~(K-1)/K in warmup.
            adaptive_K_warmup: Number of warmup steps to use K=1 before switching
                to full K. Only used when use_adaptive_K=True.
            use_amp: If True, use torch.cuda.amp autocast for forward passes.
                Gives ~30-50% speedup on Ampere+ GPUs with minimal precision loss.
            use_warm_start: If True, initialize Δe from a direction estimated by
                the target velocity field, instead of zeros. Saves ~3-5 warmup steps.
            device: Compute device.
        """
        self.pipe = pipe
        self.T_m = T_m
        self.num_opt_steps = num_opt_steps
        self.lr = lr
        self.num_timesteps_per_step = num_timesteps_per_step
        self.motion_weight_strength = motion_weight_strength
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        self.use_batched_forward = use_batched_forward
        self.use_adaptive_K = use_adaptive_K
        self.adaptive_K_warmup = adaptive_K_warmup
        self.use_amp = use_amp
        self.use_warm_start = use_warm_start
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
        """Single model forward pass (with grad for Δe), optionally with AMP."""
        model = self._get_model()

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=latents.device, dtype=latents.dtype)

        if self.use_amp and latents.is_cuda:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_output = model(
                    hidden_states=latents,
                    timestep=timestep.expand(latents.shape[0]),
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
        else:
            model_output = model(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0]),
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )

        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _model_forward_batched(
        self,
        latents_batch: torch.Tensor,
        timesteps_batch: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward pass for K timesteps simultaneously, optionally with AMP.

        Args:
            latents_batch: (K, C, F, H, W) — K different x_t stacked
            timesteps_batch: (K,) — K different timesteps
            encoder_hidden_states: (1, L, D) — same embedding for all K
                Will be expanded to (K, L, D).

        Returns:
            v_pred_batch: (K, C, F, H, W)
        """
        model = self._get_model()
        K = latents_batch.shape[0]

        timesteps_batch = timesteps_batch.to(
            device=latents_batch.device, dtype=latents_batch.dtype
        )

        # Expand embedding to batch size K
        e_expanded = encoder_hidden_states.expand(K, -1, -1)

        if self.use_amp and latents_batch.is_cuda:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_output = model(
                    hidden_states=latents_batch,
                    timestep=timesteps_batch,
                    encoder_hidden_states=e_expanded,
                    return_dict=False,
                )
        else:
            model_output = model(
                hidden_states=latents_batch,
                timestep=timesteps_batch,
                encoder_hidden_states=e_expanded,
                return_dict=False,
            )

        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _compute_motion_weight(self, v_star: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel motion weight based on Latent Temporal Discrepancy (LTD).

        For video latents (B, C, F, H, W), motion is measured as temporal variance
        of v* across frames. High temporal variance = high motion = higher weight.

        Formula: w = 1 + strength * log(1 + σ_temporal²(v*) / μ)
        where μ is the mean temporal variance (normalization constant).

        Returns:
            weight: (B, C, F, H, W) or scalar 1.0 if not video.
        """
        if self.motion_weight_strength <= 0:
            return 1.0

        if v_star.dim() != 5:
            return 1.0

        # v_star shape: (B, C, F, H, W)
        # Temporal variance: variance across frame dimension
        temporal_var = v_star.var(dim=2, keepdim=True)  # (B, C, 1, H, W)
        # Broadcast to full shape
        temporal_var = temporal_var.expand_as(v_star)

        # Normalize by mean variance to get relative importance
        mu = temporal_var.mean().clamp(min=1e-8)

        # Log-scaled weight (prevents extreme values)
        weight = 1.0 + self.motion_weight_strength * torch.log1p(temporal_var / mu)

        # Normalize so mean weight = 1 (preserves loss magnitude)
        weight = weight / weight.mean()

        return weight.detach()

    def _sample_stratified_timesteps(self, K: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Stratified uniform sampling of K timesteps in [0, T_m].

        Divides [0, T_m] into K equal bins and samples one point per bin.
        This guarantees better coverage of the time range vs pure random sampling.

        Returns:
            t_norms: (K,) tensor of sampled normalized timesteps.
        """
        bin_width = self.T_m / K
        # Sample one uniform point within each bin
        bin_offsets = torch.rand(K, device=self.device, dtype=dtype) * bin_width
        bin_starts = torch.linspace(0, self.T_m - bin_width, K, device=self.device, dtype=dtype)
        t_norms = bin_starts + bin_offsets
        return t_norms

    def optimize(
        self,
        z0: torch.Tensor,
        e0: torch.Tensor,
        eta_inv: torch.Tensor,
        token_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute velocity field matching optimization (v2.1 — optimized).

        Performance improvements over v2:
        - Batched forward: K timesteps processed in single model call
        - Early stopping: Converged optimization exits early
        - Reduced overhead: Less Python loop overhead, better GPU utilization

        Args:
            z0: Target video latent (B, C, F, H, W).
            e0: Caption text embedding (B, L, D).
            eta_inv: Inverted noise from P-Flow's conditional inversion (B, C, F, H, W).
            token_length: Actual number of meaningful tokens in the caption
                (excluding padding). If None, no padding mask is applied.

        Returns:
            Dictionary containing:
                - delta_e: Optimized embedding residual (B, L, D)
                - loss_history: Per-step loss values
                - final_loss: Final velocity matching loss
                - steps_taken: Actual number of steps (may be < num_opt_steps)
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

        # Build padding mask for gradient zeroing
        # Shape: (1, L, 1) — broadcast over hidden dim
        seq_len = e0.shape[1]
        if token_length is not None and token_length < seq_len:
            padding_mask = torch.zeros(1, seq_len, 1, device=self.device, dtype=e0.dtype)
            padding_mask[:, :token_length, :] = 1.0
            logger.info(
                f"    [PaddingMask] Active tokens: {token_length}/{seq_len} "
                f"({token_length/seq_len*100:.1f}%)"
            )
        else:
            padding_mask = None

        # Detach constant tensors
        z0 = z0.detach()
        eta_inv = eta_inv.detach()
        e0 = e0.detach()

        # Target velocity (constant throughout optimization)
        v_star = z0 - eta_inv

        # ═══ Warm-start Δe initialization ═══
        # Instead of starting from zeros, project v_star direction into embedding space
        # via a single-step "pseudo-gradient" to give Δe a meaningful initial direction.
        if self.use_warm_start:
            delta_e = self._compute_warm_start(z0, eta_inv, v_star, e0)
            delta_e = delta_e.requires_grad_(True)
            logger.info(f"    [WarmStart] Δe initialized with ||Δe||={delta_e.norm().item():.4f}")
        else:
            delta_e = torch.zeros_like(e0, requires_grad=True)

        # Create optimizer AFTER delta_e is initialized (may be warm-started)
        optimizer = torch.optim.Adam([delta_e], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_opt_steps, eta_min=self.lr * 0.1
        )

        # Pre-compute motion weight (constant, detached)
        motion_weight = self._compute_motion_weight(v_star)
        if isinstance(motion_weight, torch.Tensor):
            logger.info(
                f"    [MotionWeight] Temporal variance range: "
                f"min={motion_weight.min().item():.3f}, max={motion_weight.max().item():.3f}"
            )

        loss_history = []
        final_loss = 0.0

        K_full = self.num_timesteps_per_step

        # Determine if batched forward is feasible
        # For large video latents, batching K copies may OOM → fall back to sequential
        can_batch = self.use_batched_forward
        if can_batch and z0.dim() == 5:
            # Estimate: K copies of latent + activations
            latent_bytes = z0.numel() * z0.element_size()
            if K_full * latent_bytes > 4 * 1024**3:  # > 4GB for latents alone
                can_batch = False
                logger.info(f"    [Batched] Disabled: K={K_full} × latent too large, using sequential")

        # Early stopping state
        best_loss = float("inf")
        patience_counter = 0
        steps_taken = 0

        logger.info(
            f"  [VelocityMatch] Optimizing: "
            f"steps={self.num_opt_steps}, T_m={self.T_m}, lr={self.lr}, "
            f"K={K_full}, motion_weight={self.motion_weight_strength:.1f}, "
            f"padding_mask={'ON' if padding_mask is not None else 'OFF'}, "
            f"batched={'ON' if can_batch else 'OFF'}, "
            f"adaptive_K={'ON' if self.use_adaptive_K else 'OFF'}, "
            f"amp={'ON' if self.use_amp else 'OFF'}, "
            f"warm_start={'ON' if self.use_warm_start else 'OFF'}, "
            f"early_stop=(patience={self.early_stop_patience}, thr={self.early_stop_threshold})"
        )

        for step in range(self.num_opt_steps):
            optimizer.zero_grad()

            # ═══ Adaptive K Scheduling ═══
            # Use K=1 in warmup (fast directional convergence), then K_full for refinement
            if self.use_adaptive_K and step < self.adaptive_K_warmup:
                K_step = 1
            else:
                K_step = K_full

            # ═══ Stratified Multi-Timestep Sampling ═══
            t_norms = self._sample_stratified_timesteps(K_step, dtype=z0.dtype)

            # Compute loss — batched or sequential
            e_current = e0 + delta_e

            if can_batch and K_step > 1:
                loss = self._compute_loss_batched(
                    z0, eta_inv, v_star, e_current, t_norms, motion_weight
                )
            else:
                loss = self._compute_loss_sequential(
                    z0, eta_inv, v_star, e_current, t_norms, motion_weight
                )

            # Backward
            loss.backward()

            # ═══ Padding-Aware Gradient Mask ═══
            if padding_mask is not None and delta_e.grad is not None:
                delta_e.grad.data.mul_(padding_mask)

            # Update
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append({
                "step": step,
                "loss": loss_val,
                "lr": scheduler.get_last_lr()[0],
            })
            final_loss = loss_val
            steps_taken = step + 1

            # ═══ Early Stopping Check ═══
            if loss_val < best_loss * (1 - self.early_stop_threshold):
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience and step >= 10:
                # Don't early-stop in the first 10 steps (warmup)
                logger.info(
                    f"    Early stop at step {step}: loss={loss_val:.6f}, "
                    f"best={best_loss:.6f}, patience exhausted"
                )
                break

            # Logging
            if step % 10 == 0 or step == self.num_opt_steps - 1:
                delta_norm = delta_e.norm().item()
                if padding_mask is not None:
                    effective_norm = (delta_e * padding_mask).norm().item()
                    logger.info(
                        f"    step {step:3d}/{self.num_opt_steps}: "
                        f"L_vel={loss_val:.6f}, "
                        f"||Δe||={delta_norm:.4f} (effective={effective_norm:.4f}), "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                else:
                    logger.info(
                        f"    step {step:3d}/{self.num_opt_steps}: "
                        f"L_vel={loss_val:.6f}, "
                        f"||Δe||={delta_norm:.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        # Restore model state
        if latent_frames >= 13:
            if hasattr(model, "disable_gradient_checkpointing"):
                model.disable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        model.eval()

        logger.info(
            f"  [VelocityMatch] Complete: {steps_taken}/{self.num_opt_steps} steps, "
            f"final_loss={final_loss:.6f}"
        )

        return {
            "delta_e": delta_e.detach(),
            "loss_history": loss_history,
            "final_loss": final_loss,
            "steps_taken": steps_taken,
        }

    def _compute_loss_batched(
        self,
        z0: torch.Tensor,
        eta_inv: torch.Tensor,
        v_star: torch.Tensor,
        e_current: torch.Tensor,
        t_norms: torch.Tensor,
        motion_weight,
    ) -> torch.Tensor:
        """
        Compute velocity matching loss with all K timesteps in a single batched forward.

        This is significantly faster than sequential when GPU memory allows, because:
        - Single kernel launch for the transformer
        - Better GPU utilization (larger batch = better parallelism)
        - Reduced Python overhead

        The backward pass is also batched (single backward over K outputs).
        """
        K = t_norms.shape[0]

        # Construct K different x_t: x_t_k = (1 - t_k) * eta_inv + t_k * z0
        # Shapes: t_norms is (K,), z0 is (1, C, F, H, W)
        t_expanded = t_norms.view(K, 1, 1, 1, 1)  # (K, 1, 1, 1, 1)
        x_t_batch = (1 - t_expanded) * eta_inv + t_expanded * z0  # (K, C, F, H, W)

        # Timesteps for model (scaled to [0, 1000])
        t_model_batch = t_norms * 1000.0  # (K,)

        # Batched forward
        v_pred_batch = self._model_forward_batched(x_t_batch, t_model_batch, e_current)
        # v_pred_batch shape: (K, C, F, H, W)

        # Expand v_star to match batch: (1, C, F, H, W) → (K, C, F, H, W)
        v_star_expanded = v_star.expand(K, -1, -1, -1, -1)

        # Compute residual
        residual_sq = (v_pred_batch - v_star_expanded) ** 2

        # Apply motion weight
        if isinstance(motion_weight, torch.Tensor):
            # motion_weight is (1, C, F, H, W), expand to (K, C, F, H, W)
            mw_expanded = motion_weight.expand(K, -1, -1, -1, -1)
            loss = (residual_sq * mw_expanded).mean()
        else:
            loss = residual_sq.mean()

        return loss

    def _compute_loss_sequential(
        self,
        z0: torch.Tensor,
        eta_inv: torch.Tensor,
        v_star: torch.Tensor,
        e_current: torch.Tensor,
        t_norms: torch.Tensor,
        motion_weight,
    ) -> torch.Tensor:
        """
        Compute velocity matching loss with K timesteps sequentially.

        Fallback for when batched forward would OOM (large video latents).
        Still accumulates loss in a single computation graph for one backward pass.
        """
        K = t_norms.shape[0]
        total_loss = torch.tensor(0.0, device=self.device, dtype=z0.dtype)

        for k in range(K):
            t_norm = t_norms[k]
            t_model = t_norm * 1000.0

            # Construct x_t = (1-t)·η_inv + t·z₀
            x_t = (1 - t_norm) * eta_inv + t_norm * z0

            # Forward pass with current Δe
            v_pred = self._model_forward(x_t, t_model, e_current)

            # Motion-Aware Loss Weighting
            residual_sq = (v_pred - v_star) ** 2
            if isinstance(motion_weight, torch.Tensor):
                weighted_loss = (residual_sq * motion_weight).mean()
            else:
                weighted_loss = residual_sq.mean()

            total_loss = total_loss + weighted_loss

        # Average over K timesteps
        return total_loss / K

    def _compute_warm_start(
        self,
        z0: torch.Tensor,
        eta_inv: torch.Tensor,
        v_star: torch.Tensor,
        e0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a warm-start initialization for Δe using a single-step
        pseudo-gradient approach.

        Strategy:
            1. Do one forward pass at t=0.5 (midpoint) with current e0
            2. Compute residual: r = v_pred - v_star
            3. Use the model's cross-attention Jacobian approximation:
               Δe ≈ -lr_init * J^T @ r (where J^T is approximated via
               a single backward through the frozen model)

        This gives Δe a meaningful initial direction aligned with the
        velocity matching objective, saving 3-5 zero-exploration steps.

        The result is scaled to have ||Δe|| ≈ lr * 0.5 to avoid overshooting.

        Cost: 1 forward + 1 backward = 2 equivalent passes (amortized by
        saving ~5 steps × K=1 = 5 passes in warmup phase).
        """
        # Use midpoint t=0.5 as representative timestep
        t_mid = torch.tensor(0.5, device=self.device, dtype=z0.dtype)
        x_t_mid = (1 - t_mid) * eta_inv + t_mid * z0
        t_model = t_mid * 1000.0

        # Temporarily create a grad-enabled e0 copy to get the gradient direction
        e0_temp = e0.clone().requires_grad_(True)

        # Need grad enabled for backward through model
        with torch.enable_grad():
            if self.use_amp and x_t_mid.is_cuda:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    v_pred = self._get_model()(
                        hidden_states=x_t_mid,
                        timestep=t_model.unsqueeze(0).to(dtype=x_t_mid.dtype),
                        encoder_hidden_states=e0_temp,
                        return_dict=False,
                    )
            else:
                v_pred = self._get_model()(
                    hidden_states=x_t_mid,
                    timestep=t_model.unsqueeze(0).to(dtype=x_t_mid.dtype),
                    encoder_hidden_states=e0_temp,
                    return_dict=False,
                )

            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]

            # Compute MSE loss and backprop to get gradient w.r.t. embedding
            loss = ((v_pred - v_star) ** 2).mean()
            loss.backward()

        # The gradient of loss w.r.t. e0 gives us the direction to move
        grad_e = e0_temp.grad  # (B, L, D)

        if grad_e is None:
            # Fallback: model doesn't propagate grad to encoder_hidden_states
            logger.info("    [WarmStart] No gradient available, falling back to zeros")
            return torch.zeros_like(e0)

        # Scale: Δe_init = -grad normalized to have small magnitude
        # We want ||Δe_init|| ≈ lr * scale_factor (conservative start)
        grad_norm = grad_e.norm().clamp(min=1e-8)
        scale = self.lr * 0.5  # Half a learning rate step
        delta_e_init = -grad_e * (scale / grad_norm)

        return delta_e_init.detach()
