"""
SVD-based Spatial and Temporal Filtering for Noise Prior Enhancement.

Two-stage SVD projection (Section 3.3, Eq. 4-7):
    Stage 1 - Spatial Filtering (Eq. 4-5):
        Reshape η_inv to (C*F, H*W), perform SVD
        Find minimal k_s: retained energy after removal ≥ ρ_s
        Subtract top-k_s reconstruction (removes content/appearance)

    Stage 2 - Temporal Retention (Eq. 6):
        Reshape to (C*H*W, F), perform SVD
        Find minimal k_m: top-k_m energy ≥ ρ_m
        Keep only top-k_m reconstruction (preserves motion dynamics)

Paper parameters: ρ_s=0.1, ρ_m=0.9
"""

import torch
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SVDFilter:
    """
    Two-stage SVD filtering for noise prior enhancement.

    Stage 1: Spatial removal — suppress content/appearance leakage
    Stage 2: Temporal retention — preserve motion dynamics
    """

    def __init__(self, rho_s: float = 0.1, rho_m: float = 0.9):
        """
        Args:
            rho_s: Spatial energy threshold (Eq. 4). Default 0.1.
                   Find minimal k_s such that energy AFTER removing top-k_s ≥ ρ_s * total.
            rho_m: Temporal energy threshold (Eq. 6). Default 0.9.
                   Find minimal k_m such that top-k_m energy ≥ ρ_m * total.
        """
        self.rho_s = rho_s
        self.rho_m = rho_m

    def filter(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Apply two-stage SVD filtering.

        Args:
            noise_inv: Inverted noise (B, C, F, H, W) or (C, F, H, W).

        Returns:
            Filtered noise η_temporal with preserved motion dynamics.
        """
        has_batch = noise_inv.dim() == 5
        if has_batch:
            batch_size = noise_inv.shape[0]
            results = []
            for b in range(batch_size):
                result = self._filter_single(noise_inv[b])
                results.append(result)
            return torch.stack(results, dim=0)
        else:
            return self._filter_single(noise_inv)

    def _find_k_spatial(self, S: torch.Tensor) -> int:
        """
        Find minimal k_s via Eq. 4:
            (Σ_{i=k_s+1}^{r_s} σ_i²) / (Σ_{i=1}^{r_s} σ_i²) ≥ ρ_s

        Returns number of leading components to REMOVE.
        """
        energy = S ** 2
        total_energy = energy.sum()

        if total_energy == 0:
            return 1

        cumsum = energy.cumsum(0)
        retained_ratio = (total_energy - cumsum) / total_energy

        mask = retained_ratio >= self.rho_s
        if mask.any():
            k_s = mask.sum().item()
        else:
            k_s = 1

        return max(1, k_s)

    def _find_k_temporal(self, S: torch.Tensor) -> int:
        """
        Find minimal k_m via Eq. 6:
            (Σ_{i=1}^{k_m} σ'_i²) / (Σ_{i=1}^{r_m} σ'_i²) ≥ ρ_m

        Returns number of leading components to KEEP.
        """
        energy = S ** 2
        total_energy = energy.sum()

        if total_energy == 0:
            return 1

        cumsum_ratio = energy.cumsum(0) / total_energy

        mask = cumsum_ratio >= self.rho_m
        if mask.any():
            k_m = mask.to(torch.long).argmax().item() + 1
        else:
            k_m = len(S)

        return max(1, k_m)

    def _filter_single(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Apply two-stage SVD filtering to a single noise tensor.

        Args:
            noise_inv: Noise tensor (C, F, H, W).

        Returns:
            Filtered tensor (C, F, H, W).
        """
        C, F, H, W = noise_inv.shape

        # SVD requires float32 (bfloat16/float16 not supported by CUDA SVD)
        original_dtype = noise_inv.dtype
        if noise_inv.dtype in (torch.bfloat16, torch.float16):
            noise_inv = noise_inv.float()

        # --- Stage 1: Spatial Filtering (Eq. 4-5) ---
        noise_spatial = noise_inv.reshape(C * F, H * W)
        U_s, S_s, Vh_s = torch.linalg.svd(noise_spatial, full_matrices=False)

        k_s = self._find_k_spatial(S_s)
        logger.debug(f"  Spatial SVD: k_s={k_s}/{len(S_s)}, removing top-{k_s} components")

        # Subtract top-k_s (Eq. 5)
        top_k_recon = U_s[:, :k_s] @ torch.diag(S_s[:k_s]) @ Vh_s[:k_s, :]
        noise_spatial_filtered = noise_spatial - top_k_recon
        noise_spatial_filtered = noise_spatial_filtered.reshape(C, F, H, W)

        # --- Stage 2: Temporal Retention (Eq. 6) ---
        noise_temporal = noise_spatial_filtered.reshape(C * H * W, F)
        U_m, S_m, Vh_m = torch.linalg.svd(noise_temporal, full_matrices=False)

        k_m = self._find_k_temporal(S_m)
        k_m = min(k_m, len(S_m))
        logger.debug(f"  Temporal SVD: k_m={k_m}/{len(S_m)}, keeping top-{k_m} components")

        # Keep only top-k_m (Eq. 6)
        noise_temporal_filtered = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
        noise_temporal_filtered = noise_temporal_filtered.reshape(C, F, H, W)

        # Restore original dtype
        if noise_temporal_filtered.dtype != original_dtype:
            noise_temporal_filtered = noise_temporal_filtered.to(original_dtype)

        return noise_temporal_filtered

    def filter_efficient(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient SVD filtering using randomized SVD for large tensors.
        Useful when C*F*H*W is very large.
        """
        has_batch = noise_inv.dim() == 5
        if has_batch:
            results = []
            for b in range(noise_inv.shape[0]):
                results.append(self._filter_single_efficient(noise_inv[b]))
            return torch.stack(results, dim=0)
        else:
            return self._filter_single_efficient(noise_inv)

    def _filter_single_efficient(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """Randomized SVD for spatial stage, full SVD for temporal (F is small)."""
        C, F, H, W = noise_inv.shape

        original_dtype = noise_inv.dtype
        if noise_inv.dtype in (torch.bfloat16, torch.float16):
            noise_inv = noise_inv.float()

        # Stage 1: Spatial with randomized SVD
        noise_spatial = noise_inv.reshape(C * F, H * W)
        max_k_estimate = min(min(C * F, H * W), max(50, int(0.3 * min(C * F, H * W))))
        U_s, S_s, V_s = torch.svd_lowrank(noise_spatial, q=max_k_estimate)

        k_s = self._find_k_spatial(S_s)
        k_s = min(k_s, len(S_s))

        top_k_recon = U_s[:, :k_s] @ torch.diag(S_s[:k_s]) @ V_s[:, :k_s].T
        noise_spatial_filtered = noise_spatial - top_k_recon
        noise_spatial_filtered = noise_spatial_filtered.reshape(C, F, H, W)

        # Stage 2: Temporal with full SVD (F is small, ~21 in latent space)
        noise_temporal = noise_spatial_filtered.reshape(C * H * W, F)
        U_m, S_m, Vh_m = torch.linalg.svd(noise_temporal, full_matrices=False)

        k_m = self._find_k_temporal(S_m)
        k_m = min(k_m, len(S_m))

        noise_temporal_filtered = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
        noise_temporal_filtered = noise_temporal_filtered.reshape(C, F, H, W)

        if noise_temporal_filtered.dtype != original_dtype:
            noise_temporal_filtered = noise_temporal_filtered.to(original_dtype)

        return noise_temporal_filtered


def compute_svd_statistics(noise: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute SVD statistics for analysis/debugging."""
    C, F, H, W = noise.shape

    if noise.dtype in (torch.bfloat16, torch.float16):
        noise = noise.float()

    spatial = noise.reshape(C * F, H * W)
    _, S_spatial, _ = torch.linalg.svd(spatial, full_matrices=False)

    temporal = noise.reshape(C * H * W, F)
    _, S_temporal, _ = torch.linalg.svd(temporal, full_matrices=False)

    spatial_energy = (S_spatial ** 2).cumsum(0) / (S_spatial ** 2).sum()
    temporal_energy = (S_temporal ** 2).cumsum(0) / (S_temporal ** 2).sum()

    return {
        "spatial_singular_values": S_spatial.cpu(),
        "temporal_singular_values": S_temporal.cpu(),
        "spatial_energy_cumulative": spatial_energy.cpu(),
        "temporal_energy_cumulative": temporal_energy.cpu(),
        "spatial_rank": len(S_spatial),
        "temporal_rank": len(S_temporal),
    }
