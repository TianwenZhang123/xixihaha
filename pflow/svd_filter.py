"""
SVD-based Spatial and Temporal Filtering for Noise Prior Enhancement.

This module implements the two-stage SVD projection described in Section 3.3:
1. Spatial filtering: Remove dominant spatial components to reduce content leakage
2. Temporal retention: Preserve temporal dynamics (motion patterns)

Key equations from the paper:
    Spatial (Eq. 4): Find minimal k_s such that retained energy satisfies:
        (Σ_{i=k_s+1}^{r_s} σ_i²) / (Σ_{i=1}^{r_s} σ_i²) ≥ ρ_s
        
    Temporal (Eq. 6): Find minimal k_m such that retained energy satisfies:
        (Σ_{i=1}^{k_m} σ'_i²) / (Σ_{i=1}^{r_m} σ'_i²) ≥ ρ_m

    η_spatial = η_inv - U_s[:,:k_s] @ diag(S_s[:k_s]) @ Vh_s[:k_s,:]  (Eq. 5)
    η_temporal = U_m[:,:k_m] @ diag(S_m[:k_m]) @ Vh_m[:k_m,:]          (Eq. 6 result)

where ρ_s and ρ_m are energy thresholds (not fixed ratios).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class SVDFilter:
    """
    Two-stage SVD filtering for noise prior enhancement.
    
    Stage 1 (Spatial Filtering - Eq. 4-5):
        - Reshape η_inv from (C, F, H, W) to (C*F, H*W)
        - Perform SVD
        - Adaptively find k_s: minimal k such that REMAINING energy ≥ ρ_s of total
        - Subtract top-k_s reconstruction (removes spatial/appearance content)
        
    Stage 2 (Temporal Retention - Eq. 6):
        - Reshape η_spatial from (C, F, H, W) to (C*H*W, F)
        - Perform SVD
        - Adaptively find k_m: minimal k such that top-k energy ≥ ρ_m of total
        - Keep only top-k_m reconstruction (preserves temporal/motion dynamics)
    """
    
    def __init__(self, rho_s: float = 0.1, rho_m: float = 0.9):
        """
        Args:
            rho_s: Spatial energy threshold (Eq. 4).
                   Find minimal k_s such that energy AFTER removing top-k_s ≥ ρ_s * total.
                   Paper default: 0.1 (retain at least 10% energy after spatial suppression).
                   Lower ρ_s → remove more spatial components → stronger filtering.
            rho_m: Temporal energy threshold (Eq. 6).
                   Find minimal k_m such that top-k_m energy ≥ ρ_m * total.
                   Paper default: 0.9 (retain at least 90% temporal energy).
                   Higher ρ_m → keep more temporal components → preserve more motion.
        """
        self.rho_s = rho_s
        self.rho_m = rho_m
        
    def filter(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Apply two-stage SVD filtering to inverted noise.
        
        Args:
            noise_inv: Inverted noise tensor of shape (B, C, F, H, W) or (C, F, H, W).
            
        Returns:
            Filtered noise η_temporal with temporal dynamics preserved
            and spatial content removed.
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
        Find minimal k_s such that retained energy after removing top-k_s satisfies Eq. 4:
            (Σ_{i=k_s+1}^{r_s} σ_i²) / (Σ_{i=1}^{r_s} σ_i²) ≥ ρ_s
            
        This means: find the smallest k_s where removing the top k_s components
        still leaves at least ρ_s fraction of total energy.
        
        Args:
            S: Singular values tensor (sorted descending).
            
        Returns:
            k_s: Number of leading components to remove.
        """
        energy = S ** 2
        total_energy = energy.sum()
        
        if total_energy == 0:
            return 1
        
        # Cumulative energy from top → retained = total - cumsum
        cumsum = energy.cumsum(0)
        # retained_energy[k] = total - sum of top-(k+1) components
        retained_ratio = (total_energy - cumsum) / total_energy
        
        # Find minimal k_s (1-indexed) such that retained_ratio[k_s-1] >= rho_s
        # i.e., after removing top k_s components, remaining energy >= rho_s * total
        mask = retained_ratio >= self.rho_s
        if mask.any():
            # k_s is the last index where condition holds + 1 (to include it)
            k_s = mask.sum().item()
        else:
            k_s = 1
        
        return max(1, k_s)
    
    def _find_k_temporal(self, S: torch.Tensor) -> int:
        """
        Find minimal k_m such that top-k_m energy satisfies Eq. 6:
            (Σ_{i=1}^{k_m} σ'_i²) / (Σ_{i=1}^{r_m} σ'_i²) ≥ ρ_m
            
        This means: find the smallest k_m whose cumulative energy reaches ρ_m of total.
        
        Args:
            S: Singular values tensor (sorted descending).
            
        Returns:
            k_m: Number of leading components to retain.
        """
        energy = S ** 2
        total_energy = energy.sum()
        
        if total_energy == 0:
            return 1
        
        # Cumulative energy ratio from top
        cumsum_ratio = energy.cumsum(0) / total_energy
        
        # Find minimal k_m such that cumsum_ratio[k_m-1] >= rho_m
        mask = cumsum_ratio >= self.rho_m
        if mask.any():
            # First index where ratio crosses threshold → k_m = index + 1
            k_m = mask.to(torch.long).argmax().item() + 1
        else:
            # If never reaches threshold, use all components
            k_m = len(S)
        
        return max(1, k_m)

    def _filter_single(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Apply SVD filtering to a single noise tensor using adaptive energy thresholds.
        
        Args:
            noise_inv: Noise tensor of shape (C, F, H, W).
            
        Returns:
            Filtered noise tensor of shape (C, F, H, W).
        """
        C, F, H, W = noise_inv.shape
        
        # SVD requires float32 (BFloat16 not supported by CUDA SVD kernel)
        original_dtype = noise_inv.dtype
        if noise_inv.dtype == torch.bfloat16 or noise_inv.dtype == torch.float16:
            noise_inv = noise_inv.float()
        
        # Stage 1: Spatial Filtering (Eq. 4-5)
        # Reshape to (C*F, H*W) — N_s in paper notation
        noise_spatial = noise_inv.reshape(C * F, H * W)
        
        # SVD decomposition: N_s = U_s @ diag(S_s) @ Vh_s
        U_s, S_s, Vh_s = torch.linalg.svd(noise_spatial, full_matrices=False)
        
        # Adaptively determine k_s via energy threshold (Eq. 4)
        k_s = self._find_k_spatial(S_s)
        
        # Subtract top-k_s reconstruction (Eq. 5)
        # η_spatial = η_inv - U[:, :k_s] @ diag(S[:k_s]) @ Vh[:k_s, :]
        top_k_reconstruction = U_s[:, :k_s] @ torch.diag(S_s[:k_s]) @ Vh_s[:k_s, :]
        noise_spatial_filtered = noise_spatial - top_k_reconstruction
        
        # Reshape back to (C, F, H, W)
        noise_spatial_filtered = noise_spatial_filtered.reshape(C, F, H, W)
        
        # Stage 2: Temporal Retention (Eq. 6)
        # Reshape to (C*H*W, F) — N_m in paper notation
        noise_temporal = noise_spatial_filtered.reshape(C * H * W, F)
        
        # SVD decomposition: N_m = U_m @ diag(S_m) @ Vh_m
        U_m, S_m, Vh_m = torch.linalg.svd(noise_temporal, full_matrices=False)
        
        # Adaptively determine k_m via energy threshold (Eq. 6)
        k_m = self._find_k_temporal(S_m)
        # k_m should not exceed available singular values
        k_m = min(k_m, len(S_m))
        
        # Reconstruct using only top-k_m components (temporal dynamics)
        # η_temporal = U[:, :k_m] @ diag(S[:k_m]) @ Vh[:k_m, :]
        noise_temporal_filtered = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
        
        # Reshape back to (C, F, H, W)
        noise_temporal_filtered = noise_temporal_filtered.reshape(C, F, H, W)
        
        # Convert back to original dtype
        if noise_temporal_filtered.dtype != original_dtype:
            noise_temporal_filtered = noise_temporal_filtered.to(original_dtype)
        
        return noise_temporal_filtered
    
    def filter_efficient(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient SVD filtering using randomized SVD for large tensors.
        
        For high-resolution videos, full SVD can be very expensive.
        This uses torch's randomized SVD (lowrank) for efficiency.
        
        Args:
            noise_inv: Noise tensor of shape (B, C, F, H, W) or (C, F, H, W).
            
        Returns:
            Filtered noise tensor.
        """
        has_batch = noise_inv.dim() == 5
        if has_batch:
            batch_size = noise_inv.shape[0]
            results = []
            for b in range(batch_size):
                result = self._filter_single_efficient(noise_inv[b])
                results.append(result)
            return torch.stack(results, dim=0)
        else:
            return self._filter_single_efficient(noise_inv)
    
    def _filter_single_efficient(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient SVD filtering using randomized SVD for large tensors.
        
        Uses adaptive energy thresholds same as _filter_single, but with
        randomized SVD (torch.svd_lowrank) for the spatial stage where
        full SVD would be prohibitively expensive.
        
        Note: For the spatial stage, we over-estimate k_s by requesting more
        components than needed, then apply the energy threshold on the returned
        singular values. For temporal stage (F is small), we use full SVD.
        
        Args:
            noise_inv: Noise tensor of shape (C, F, H, W).
            
        Returns:
            Filtered noise tensor of shape (C, F, H, W).
        """
        C, F, H, W = noise_inv.shape
        
        # SVD requires float32 (BFloat16 not supported by CUDA SVD kernel)
        original_dtype = noise_inv.dtype
        if noise_inv.dtype == torch.bfloat16 or noise_inv.dtype == torch.float16:
            noise_inv = noise_inv.float()
        
        # Stage 1: Spatial Filtering with randomized SVD
        noise_spatial = noise_inv.reshape(C * F, H * W)
        
        # Over-estimate: request enough components to find the energy threshold
        # For rho_s=0.1, typically only a small fraction of components are needed
        max_k_estimate = min(min(C * F, H * W), max(50, int(0.3 * min(C * F, H * W))))
        U_s, S_s, V_s = torch.svd_lowrank(noise_spatial, q=max_k_estimate)
        
        # Apply adaptive energy threshold on the returned singular values
        k_s = self._find_k_spatial(S_s)
        k_s = min(k_s, len(S_s))
        
        U_s = U_s[:, :k_s]
        S_s = S_s[:k_s]
        V_s = V_s[:, :k_s]
        
        # Subtract top-k spatial components
        top_k_reconstruction = U_s @ torch.diag(S_s) @ V_s.T
        noise_spatial_filtered = noise_spatial - top_k_reconstruction
        noise_spatial_filtered = noise_spatial_filtered.reshape(C, F, H, W)
        
        # Stage 2: Temporal Retention
        # Since F is usually small (81 frames → latent ~21), use full SVD
        noise_temporal = noise_spatial_filtered.reshape(C * H * W, F)
        U_m, S_m, Vh_m = torch.linalg.svd(noise_temporal, full_matrices=False)
        
        # Apply adaptive energy threshold (Eq. 6)
        k_m = self._find_k_temporal(S_m)
        k_m = min(k_m, len(S_m))
        
        noise_temporal_filtered = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
        noise_temporal_filtered = noise_temporal_filtered.reshape(C, F, H, W)
        
        # Convert back to original dtype
        if noise_temporal_filtered.dtype != original_dtype:
            noise_temporal_filtered = noise_temporal_filtered.to(original_dtype)
        
        return noise_temporal_filtered


def compute_svd_statistics(noise: torch.Tensor) -> dict:
    """
    Compute SVD statistics for analysis/debugging.
    
    Args:
        noise: Noise tensor of shape (C, F, H, W).
        
    Returns:
        Dictionary with singular value statistics.
    """
    C, F, H, W = noise.shape
    
    # Spatial SVD
    spatial = noise.reshape(C * F, H * W)
    _, S_spatial, _ = torch.linalg.svd(spatial, full_matrices=False)
    
    # Temporal SVD  
    temporal = noise.reshape(C * H * W, F)
    _, S_temporal, _ = torch.linalg.svd(temporal, full_matrices=False)
    
    # Compute energy ratios
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
