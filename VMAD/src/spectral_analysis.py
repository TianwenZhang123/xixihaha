"""
Spectral Analysis Utilities for Motion-Content Separation Validation.

This module provides tools for analyzing and validating the spectral
properties of the VMAD framework, including:

1. Spectral Boundary Estimation: Empirically determine T_m* where
   motion and content influence on the velocity field cross over.

2. Frequency-Domain Visualization: Analyze the spectral distribution
   of inverted noise, filtered noise, and motion embeddings.

3. Position Bias Analysis: Measure and visualize the U-shape attention
   weight distribution in DiT cross-attention.

4. Information-Theoretic Metrics: Compute mutual information estimates
   between motion/content and the three-layer representation.

These tools support the theoretical claims in the paper and provide
empirical validation for the spectral autoregression hypothesis.

References:
    - Dieleman (2024): Spectral autoregression in diffusion
    - Spectral Progressive Diffusion (2025): Frequency-aware generation
    - Wu et al. (ICML 2025): Position bias in transformers
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpectralBoundaryEstimator:
    """
    Estimates the spectral boundary T_m* between motion-dominated and
    content-dominated regions of the velocity field.

    Methodology:
        1. Fix motion (same reference video), vary content prompts
           → Measure velocity field variance at each timestep t
        2. Fix content (same prompt), vary motion (different references)
           → Measure velocity field variance at each timestep t
        3. The crossover point where motion variance drops below content
           variance defines T_m*

    This validates Proposition 1 (Spectral Separation Theorem):
        For t < T_m*: velocity field is motion-dominated
        For t > T_m*: velocity field is content-dominated
    """

    def __init__(self, pipe, device: str = "cuda", num_timesteps: int = 20):
        """
        Args:
            pipe: Diffusers pipeline
            device: Compute device
            num_timesteps: Number of timesteps to evaluate
        """
        self.pipe = pipe
        self.device = device
        self.num_timesteps = num_timesteps

    def _get_model(self):
        if hasattr(self.pipe, "transformer"):
            return self.pipe.transformer
        return self.pipe.unet

    @torch.no_grad()
    def estimate_boundary(
        self,
        z0: torch.Tensor,
        embeddings_varied_content: List[torch.Tensor],
        embeddings_varied_motion: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Estimate T_m* by measuring velocity field sensitivity.

        Args:
            z0: Reference video latent (B, C, F, H, W)
            embeddings_varied_content: List of embeddings with same motion,
                                       different content descriptions
            embeddings_varied_motion: List of embeddings with same content,
                                      different motion Δe applied

        Returns:
            Dictionary with:
                - T_m_star: Estimated spectral boundary
                - motion_sensitivity: Per-timestep motion sensitivity curve
                - content_sensitivity: Per-timestep content sensitivity curve
                - crossover_timestep: Where curves cross
        """
        model = self._get_model()
        model.eval()

        timesteps = torch.linspace(0, 1, self.num_timesteps, device=self.device)
        motion_sensitivity = []
        content_sensitivity = []

        for t in timesteps:
            eta = torch.randn_like(z0)
            x_t = (1 - t) * eta + t * z0
            t_tensor = t.unsqueeze(0).to(dtype=z0.dtype)

            # Content sensitivity: variance across different content embeddings
            v_content_preds = []
            for emb in embeddings_varied_content:
                v = model(
                    hidden_states=x_t,
                    timestep=t_tensor.expand(x_t.shape[0]),
                    encoder_hidden_states=emb,
                    return_dict=False,
                )
                v = v[0] if isinstance(v, tuple) else v
                v_content_preds.append(v)

            if v_content_preds:
                v_stack = torch.stack(v_content_preds)
                content_var = v_stack.var(dim=0).mean().item()
                content_sensitivity.append(content_var)
            else:
                content_sensitivity.append(0.0)

            # Motion sensitivity: variance across different motion embeddings
            v_motion_preds = []
            for emb in embeddings_varied_motion:
                v = model(
                    hidden_states=x_t,
                    timestep=t_tensor.expand(x_t.shape[0]),
                    encoder_hidden_states=emb,
                    return_dict=False,
                )
                v = v[0] if isinstance(v, tuple) else v
                v_motion_preds.append(v)

            if v_motion_preds:
                v_stack = torch.stack(v_motion_preds)
                motion_var = v_stack.var(dim=0).mean().item()
                motion_sensitivity.append(motion_var)
            else:
                motion_sensitivity.append(0.0)

        # Find crossover point
        T_m_star = 0.3  # Default
        for i in range(len(timesteps) - 1):
            if (motion_sensitivity[i] > content_sensitivity[i] and
                    motion_sensitivity[i + 1] <= content_sensitivity[i + 1]):
                # Linear interpolation for precise crossover
                t1, t2 = timesteps[i].item(), timesteps[i + 1].item()
                m1, m2 = motion_sensitivity[i], motion_sensitivity[i + 1]
                c1, c2 = content_sensitivity[i], content_sensitivity[i + 1]
                # Solve: m1 + (m2-m1)*α = c1 + (c2-c1)*α
                denom = (c2 - c1) - (m2 - m1)
                if abs(denom) > 1e-8:
                    alpha = (m1 - c1) / denom
                    T_m_star = t1 + alpha * (t2 - t1)
                else:
                    T_m_star = (t1 + t2) / 2
                break

        return {
            "T_m_star": T_m_star,
            "motion_sensitivity": motion_sensitivity,
            "content_sensitivity": content_sensitivity,
            "timesteps": timesteps.cpu().tolist(),
            "crossover_timestep": T_m_star,
        }


class PositionBiasAnalyzer:
    """
    Analyzes the position bias (U-shape attention weight) in DiT cross-attention.

    Validates the empirical finding that position 0 receives 10-15× more
    attention weight than interior positions, and quantifies the impact
    on motion encoding.

    This connects to Wu et al. (ICML 2025) who prove position bias emergence
    in multi-layer attention from a graph-theoretic perspective.
    """

    def __init__(self, pipe, device: str = "cuda"):
        self.pipe = pipe
        self.device = device

    @torch.no_grad()
    def measure_attention_distribution(
        self,
        z0: torch.Tensor,
        embedding: torch.Tensor,
        timestep: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Measure cross-attention weight distribution over text positions.

        Hooks into the DiT transformer's cross-attention layers to capture
        the attention weight matrix, then analyzes the per-position distribution.

        Args:
            z0: Video latent (B, C, F, H, W)
            embedding: Text embedding (B, L, D)
            timestep: Timestep at which to measure

        Returns:
            Dictionary with attention statistics per position
        """
        model = self.pipe.transformer if hasattr(self.pipe, "transformer") else self.pipe.unet

        # Storage for attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            """Hook to capture cross-attention weights."""
            # This is model-specific; for Wan2.1 DiT we look for attention patterns
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                attention_weights.append(module.attn_weights.detach())

        # Register hooks on attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'cross_attn' in name.lower() or 'attn2' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        t = torch.tensor([timestep], device=self.device, dtype=z0.dtype)
        eta = torch.randn_like(z0)
        x_t = (1 - timestep) * eta + timestep * z0

        model(
            hidden_states=x_t,
            timestep=t.expand(x_t.shape[0]),
            encoder_hidden_states=embedding,
            return_dict=False,
        )

        # Remove hooks
        for h in hooks:
            h.remove()

        # Analyze attention weights
        if attention_weights:
            # Average across layers and heads
            avg_attn = torch.stack(attention_weights).mean(dim=0)
            # Sum over query positions to get per-key-position weight
            position_weights = avg_attn.sum(dim=-2).mean(dim=0)  # (L,)

            # Normalize
            position_weights = position_weights / position_weights.sum()

            return {
                "position_weights": position_weights.cpu().tolist(),
                "position_0_weight": position_weights[0].item(),
                "interior_mean_weight": position_weights[1:-1].mean().item() if len(position_weights) > 2 else 0,
                "u_shape_ratio": (position_weights[0].item() / (position_weights[1:-1].mean().item() + 1e-8)) if len(position_weights) > 2 else 0,
                "num_layers_captured": len(attention_weights),
            }
        else:
            logger.warning("No attention weights captured. Model may not expose cross-attention.")
            return {
                "position_weights": [],
                "position_0_weight": 0,
                "interior_mean_weight": 0,
                "u_shape_ratio": 0,
                "num_layers_captured": 0,
            }

    def compute_position_influence_map(
        self,
        z0: torch.Tensor,
        e0: torch.Tensor,
        delta_e: torch.Tensor,
        num_timesteps: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute per-position influence on velocity field across timesteps.

        For each position j and timestep t, measures:
            influence(j, t) = ||v_θ(x_t, t, e₀+Δe) - v_θ(x_t, t, e₀+Δe_{\\j})||²

        where Δe_{\\j} is Δe with position j zeroed out.

        This produces a 2D influence map that should show:
        - Position 0 has high influence across all timesteps (attention sink)
        - Early timesteps (t < T_m) show higher overall influence (motion region)
        """
        model = self.pipe.transformer if hasattr(self.pipe, "transformer") else self.pipe.unet
        model.eval()

        seq_len = delta_e.shape[-2] if delta_e.dim() == 3 else delta_e.shape[0]
        timesteps = torch.linspace(0.05, 0.5, num_timesteps, device=self.device)

        influence_map = torch.zeros(num_timesteps, seq_len)

        with torch.no_grad():
            for ti, t in enumerate(timesteps):
                eta = torch.randn_like(z0)
                x_t = (1 - t) * eta + t * z0
                t_tensor = t.unsqueeze(0).to(dtype=z0.dtype)

                # Reference velocity
                e_full = e0 + delta_e
                if e_full.dim() == 2:
                    e_full = e_full.unsqueeze(0)

                v_ref = model(
                    hidden_states=x_t,
                    timestep=t_tensor.expand(x_t.shape[0]),
                    encoder_hidden_states=e_full,
                    return_dict=False,
                )
                v_ref = v_ref[0] if isinstance(v_ref, tuple) else v_ref

                # Per-position ablation
                for j in range(seq_len):
                    delta_e_ablated = delta_e.clone()
                    if delta_e_ablated.dim() == 3:
                        delta_e_ablated[:, j, :] = 0
                    else:
                        delta_e_ablated[j, :] = 0

                    e_ablated = e0 + delta_e_ablated
                    if e_ablated.dim() == 2:
                        e_ablated = e_ablated.unsqueeze(0)

                    v_ablated = model(
                        hidden_states=x_t,
                        timestep=t_tensor.expand(x_t.shape[0]),
                        encoder_hidden_states=e_ablated,
                        return_dict=False,
                    )
                    v_ablated = v_ablated[0] if isinstance(v_ablated, tuple) else v_ablated

                    influence_map[ti, j] = ((v_ref - v_ablated) ** 2).mean().item()

        return {
            "influence_map": influence_map.tolist(),
            "timesteps": timesteps.cpu().tolist(),
            "position_indices": list(range(seq_len)),
            "position_0_total_influence": influence_map[:, 0].sum().item(),
            "total_influence": influence_map.sum().item(),
            "position_0_fraction": influence_map[:, 0].sum().item() / (influence_map.sum().item() + 1e-8),
        }


class ThreeLayerInformationAnalyzer:
    """
    Analyzes the information content of the three-layer motion representation.

    Validates the progressive refinement hypothesis:
    - Layer 1 (text): Low-bitrate, high-interpretability (coarse motion semantics)
    - Layer 2 (embedding Δe): Medium-bitrate (acceleration, rhythm, phase)
    - Layer 3 (noise prior η): High-bitrate (precise spatial trajectories)

    Each layer should encode RESIDUAL information not captured by previous layers.
    This is validated by measuring the incremental velocity field improvement
    when adding each layer.
    """

    def __init__(self, pipe, device: str = "cuda"):
        self.pipe = pipe
        self.device = device

    @torch.no_grad()
    def analyze_layer_contributions(
        self,
        z0: torch.Tensor,
        e0: torch.Tensor,
        delta_e: torch.Tensor,
        eta_motion: torch.Tensor,
        motion_text_embedding: Optional[torch.Tensor] = None,
        alpha: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Measure the incremental contribution of each representation layer.

        Computes velocity field deviation from target for:
        1. Baseline (no motion info)
        2. + Text only (Layer 1)
        3. + Embedding Δe (Layer 2)
        4. + Noise prior (Layer 3)
        5. All three combined

        Args:
            z0: Target video latent
            e0: Base embedding (content only)
            delta_e: Motion embedding residual
            eta_motion: Motion noise prior
            motion_text_embedding: Embedding of motion text description
            alpha: Noise blending weight

        Returns:
            Per-layer contribution metrics
        """
        model = self.pipe.transformer if hasattr(self.pipe, "transformer") else self.pipe.unet
        model.eval()

        # Target velocity
        eta_random = torch.randn_like(z0)
        v_target = z0 - eta_random  # Ideal velocity

        t = torch.tensor(0.15, device=self.device, dtype=z0.dtype)
        x_t = (1 - 0.15) * eta_random + 0.15 * z0
        t_tensor = t.unsqueeze(0)

        def compute_vel_error(embedding, latents=None):
            if embedding.dim() == 2:
                embedding = embedding.unsqueeze(0)
            if latents is None:
                latents = x_t
            v = model(
                hidden_states=latents,
                timestep=t_tensor.expand(latents.shape[0]),
                encoder_hidden_states=embedding,
                return_dict=False,
            )
            v = v[0] if isinstance(v, tuple) else v
            return ((v - v_target) ** 2).mean().item()

        # Baseline: content embedding only
        error_baseline = compute_vel_error(e0)

        # Layer 1: + motion text
        error_text = error_baseline
        if motion_text_embedding is not None:
            error_text = compute_vel_error(motion_text_embedding)

        # Layer 2: + Δe
        error_embedding = compute_vel_error(e0 + delta_e)

        # Layer 3: + noise prior (changes initial latent)
        eta_blended = (alpha ** 0.5) * eta_motion + ((1 - alpha) ** 0.5) * eta_random
        x_t_motion = (1 - 0.15) * eta_blended + 0.15 * z0
        error_noise = compute_vel_error(e0, x_t_motion)

        # Combined: all three
        error_combined = compute_vel_error(e0 + delta_e, x_t_motion)

        return {
            "baseline_error": error_baseline,
            "text_only_error": error_text,
            "embedding_only_error": error_embedding,
            "noise_only_error": error_noise,
            "combined_error": error_combined,
            "text_improvement": (error_baseline - error_text) / (error_baseline + 1e-8),
            "embedding_improvement": (error_baseline - error_embedding) / (error_baseline + 1e-8),
            "noise_improvement": (error_baseline - error_noise) / (error_baseline + 1e-8),
            "combined_improvement": (error_baseline - error_combined) / (error_baseline + 1e-8),
            "layer_non_redundancy": {
                "text_residual": (error_embedding - error_combined) / (error_embedding + 1e-8),
                "embedding_residual": (error_noise - error_combined) / (error_noise + 1e-8),
                "noise_residual": (error_text - error_combined) / (error_text + 1e-8),
            },
        }
