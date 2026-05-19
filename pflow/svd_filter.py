"""
SVD-based Spatial and Temporal Filtering for Noise Prior Enhancement.

This module implements the two-stage SVD projection described in Section 3.3:
1. Spatial filtering: Remove dominant spatial components to reduce content leakage
2. Temporal retention: Preserve temporal dynamics (motion patterns)

Key equations from the paper:
    Spatial: η_spatial = η_inv - Σ(σ_i * u_i * v_i^T) for i in top k_s  (Eq. 6)
    Temporal: η_temporal = Σ(σ_j * u_j * v_j^T) for j in top k_m        (Eq. 8)
    
where:
    k_s = ⌈ρ_s * min(CF, HW)⌉  (spatial components to remove)
    k_m = ⌈ρ_m * min(CHW, F)⌉  (temporal components to retain)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class SVDFilter:
    """
    Two-stage SVD filtering for noise prior enhancement.
    
    Stage 1 (Spatial Filtering):
        - Reshape η_inv from (C, F, H, W) to (C*F, H*W)
        - Perform SVD, identify top k_s singular values
        - Subtract top-k_s reconstruction (removes spatial content)
        
    Stage 2 (Temporal Retention):
        - Reshape η_spatial from (C, F, H, W) to (C*H*W, F)
        - Perform SVD, identify top k_m singular values
        - Keep only top-k_m reconstruction (preserves temporal motion)
    """
    
    def __init__(self, rho_s: float = 0.1, rho_m: float = 0.9):
        """
        Args:
            rho_s: Spatial retention ratio (fraction of singular values for spatial filtering).
                   Paper uses 0.1, meaning keep 10% → we REMOVE top 10% spatial components.
                   Actually per paper: k_s = ⌈ρ_s * min(CF, HW)⌉ components are REMOVED.
            rho_m: Temporal retention ratio (fraction of singular values to KEEP).
                   Paper uses 0.9, meaning keep 90% temporal dynamics.
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
    
    def _filter_single(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Apply SVD filtering to a single noise tensor.
        
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
        
        # Stage 1: Spatial Filtering
        # Reshape to (C*F, H*W) - each row is a spatial map
        noise_spatial = noise_inv.reshape(C * F, H * W)
        
        # Compute number of components to remove
        k_s = max(1, int(self.rho_s * min(C * F, H * W)))
        
        # SVD decomposition
        U_s, S_s, Vh_s = torch.linalg.svd(noise_spatial, full_matrices=False)
        
        # Reconstruct using only top-k_s components (to be subtracted)
        # η_spatial = η_inv - U[:, :k_s] @ diag(S[:k_s]) @ Vh[:k_s, :]
        top_k_reconstruction = U_s[:, :k_s] @ torch.diag(S_s[:k_s]) @ Vh_s[:k_s, :]
        noise_spatial_filtered = noise_spatial - top_k_reconstruction
        
        # Reshape back to (C, F, H, W)
        noise_spatial_filtered = noise_spatial_filtered.reshape(C, F, H, W)
        
        # Stage 2: Temporal Retention
        # Reshape to (C*H*W, F) - each column is a temporal frame
        noise_temporal = noise_spatial_filtered.reshape(C * H * W, F)
        
        # Compute number of temporal components to keep
        k_m = max(1, int(self.rho_m * min(C * H * W, F)))
        # k_m should not exceed min dimension
        k_m = min(k_m, min(C * H * W, F))
        
        # SVD decomposition
        U_m, S_m, Vh_m = torch.linalg.svd(noise_temporal, full_matrices=False)
        
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
        Memory-efficient SVD filtering for a single tensor using randomized SVD.
        
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
        k_s = max(1, int(self.rho_s * min(C * F, H * W)))
        
        # Use randomized SVD (torch.svd_lowrank) for top-k components
        # This is much faster for large matrices when k << min(m, n)
        U_s, S_s, V_s = torch.svd_lowrank(noise_spatial, q=k_s + 10)
        U_s = U_s[:, :k_s]
        S_s = S_s[:k_s]
        V_s = V_s[:, :k_s]
        
        # Subtract top-k spatial components
        top_k_reconstruction = U_s @ torch.diag(S_s) @ V_s.T
        noise_spatial_filtered = noise_spatial - top_k_reconstruction
        noise_spatial_filtered = noise_spatial_filtered.reshape(C, F, H, W)
        
        # Stage 2: Temporal Retention with randomized SVD
        noise_temporal = noise_spatial_filtered.reshape(C * H * W, F)
        k_m = max(1, int(self.rho_m * min(C * H * W, F)))
        k_m = min(k_m, min(C * H * W, F))
        
        # For temporal, since F is usually small (81 frames → latent ~21),
        # full SVD may actually be faster
        if F <= 128:
            U_m, S_m, Vh_m = torch.linalg.svd(noise_temporal, full_matrices=False)
            noise_temporal_filtered = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
        else:
            U_m, S_m, V_m = torch.svd_lowrank(noise_temporal, q=k_m + 10)
            U_m = U_m[:, :k_m]
            S_m = S_m[:k_m]
            V_m = V_m[:, :k_m]
            noise_temporal_filtered = U_m @ torch.diag(S_m) @ V_m.T
        
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
