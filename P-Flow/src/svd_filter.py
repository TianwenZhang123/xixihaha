"""
SVD-based Motion Prior Extraction (v1 模式).

两阶段流水线:
    Stage 1: Spatial Decontenting (去内容/外观)
    Stage 2: Temporal Retention (保运动动态)

Paper Context:
    原始两阶段 SVD (Section 3.3, Eq. 4-7), v1 模式无额外后处理。
"""

import torch
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class SVDFilterConfig:
    """SVD 滤波器配置 (v1 模式)."""

    # ── Stage 1: 空间去内容 ──
    rho_s: float = 0.1           # 空间能量阈值 (去除 top-k_s 后保留 ≥ ρ_s)

    # ── Stage 2: 时间保运动 ──
    rho_m: float = 0.9           # 时间能量阈值 (保留 top-k_m 使得能量 ≥ ρ_m)

    # ── 效率 ──
    use_fast_svd: bool = True    # 空间维度用 randomized SVD


# ─────────────────────────────────────────────────────────────
# 核心滤波器
# ─────────────────────────────────────────────────────────────

class SVDFilter:
    """
    V2 SVD Motion Prior Extractor.

    三阶段流水线:
        Stage 1: Spatial Decontenting (去内容/外观)
        Stage 2: Temporal Retention (保运动动态)
        Stage 3: Frequency Band Selection + Renormalization (频段选择 + 标准化)
    """

    def __init__(self, config: Optional[SVDFilterConfig] = None, **kwargs):
        """
        支持两种初始化方式:
            1. SVDFilter(config=SVDFilterConfig(...))
            2. SVDFilter(rho_s=0.1, rho_m=0.9)  # 从 kwargs 构建
        """
        if config is not None:
            self.config = config
        else:
            # 向后兼容: 从 kwargs 构建 config
            self.config = SVDFilterConfig(**{
                k: v for k, v in kwargs.items()
                if k in SVDFilterConfig.__dataclass_fields__
            })

    # ── V1 兼容接口 ──
    @property
    def rho_s(self) -> float:
        return self.config.rho_s

    @property
    def rho_m(self) -> float:
        return self.config.rho_m

    def filter(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        主入口: 对反演噪声进行 SVD 滤波.

        Args:
            noise_inv: (B, C, F, H, W) 或 (C, F, H, W)

        Returns:
            滤波后的运动先验噪声, 与输入同 shape.
        """
        has_batch = noise_inv.dim() == 5
        if has_batch:
            results = []
            for b in range(noise_inv.shape[0]):
                results.append(self._process_single(noise_inv[b]))
            return torch.stack(results, dim=0)
        else:
            return self._process_single(noise_inv)

    # 向后兼容
    filter_efficient = filter

    def _process_single(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        单样本完整处理流水线.

        Args:
            noise_inv: (C, F, H, W)

        Returns:
            处理后的噪声 (C, F, H, W)
        """
        C, F, H, W = noise_inv.shape
        original_dtype = noise_inv.dtype
        cfg = self.config

        # SVD 需要 float32
        if noise_inv.dtype in (torch.bfloat16, torch.float16):
            noise_inv = noise_inv.float()

        # ── Stage 1: Spatial Decontenting ──
        noise_after_spatial = self._stage1_spatial(noise_inv)

        # ── Stage 2: Temporal Retention ──
        noise_temporal, S_temporal, k_m = self._stage2_temporal(noise_after_spatial)

        # ── Stage 3: v1 模式 (直接返回 Stage 2 输出) ──
        result = self._stage3_postprocess(noise_temporal)

        # 恢复原始 dtype
        if result.dtype != original_dtype:
            result = result.to(original_dtype)

        return result

    # ─────────────────────────────────────────────────────────────
    # Stage 1: 空间去内容
    # ─────────────────────────────────────────────────────────────

    def _stage1_spatial(self, noise_inv: torch.Tensor) -> torch.Tensor:
        """
        Stage 1 - Spatial Filtering (Eq. 4-5):
            Reshape to (C*F, H*W), SVD, 去除 top-k_s 空间主成分
        """
        C, F, H, W = noise_inv.shape
        noise_2d = noise_inv.reshape(C * F, H * W)

        if self.config.use_fast_svd and (C * F > 100 or H * W > 100):
            # Randomized SVD (更快)
            max_k = min(min(C * F, H * W), max(50, int(0.3 * min(C * F, H * W))))
            U_s, S_s, V_s = torch.svd_lowrank(noise_2d, q=max_k)
            k_s = self._find_k_spatial(S_s)
            k_s = min(k_s, len(S_s))
            top_k_recon = U_s[:, :k_s] @ torch.diag(S_s[:k_s]) @ V_s[:, :k_s].T
        else:
            # Full SVD
            U_s, S_s, Vh_s = torch.linalg.svd(noise_2d, full_matrices=False)
            k_s = self._find_k_spatial(S_s)
            top_k_recon = U_s[:, :k_s] @ torch.diag(S_s[:k_s]) @ Vh_s[:k_s, :]

        noise_filtered = noise_2d - top_k_recon
        logger.debug(f"  [Stage1] Spatial: removed top-{k_s} components")

        return noise_filtered.reshape(C, F, H, W)

    # ─────────────────────────────────────────────────────────────
    # Stage 2: 时间保运动
    # ─────────────────────────────────────────────────────────────

    def _stage2_temporal(
        self, noise_spatial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Stage 2 - Temporal Retention (Eq. 6):
            Reshape to (C*H*W, F), SVD, 保留 top-k_m 时间主成分

        Returns:
            (filtered_noise, singular_values, k_m)
        """
        C, F, H, W = noise_spatial.shape
        noise_2d = noise_spatial.reshape(C * H * W, F)

        # F 通常很小 (≈21 for 81 frames)，用 full SVD
        U_m, S_m, Vh_m = torch.linalg.svd(noise_2d, full_matrices=False)

        k_m = self._find_k_temporal(S_m)
        k_m = min(k_m, len(S_m))

        # 保留 top-k_m
        noise_temporal = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
        noise_temporal = noise_temporal.reshape(C, F, H, W)

        logger.debug(f"  [Stage2] Temporal: kept top-{k_m}/{len(S_m)} components")

        return noise_temporal, S_m, k_m

    # ─────────────────────────────────────────────────────────────
    # Stage 3: 频段选择 + Renormalization
    # ─────────────────────────────────────────────────────────────

    def _stage3_postprocess(self, noise_temporal: torch.Tensor) -> torch.Tensor:
        """Stage 3: v1 模式直接返回 Stage 2 输出."""
        return noise_temporal

    # ─────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────

    def _find_k_spatial(self, S: torch.Tensor) -> int:
        """
        Eq. 4: 找最小 k_s 使得去除 top-k_s 后剩余能量 ≥ ρ_s * total.
        """
        energy = S ** 2
        total = energy.sum()
        if total == 0:
            return 1

        cumsum = energy.cumsum(0)
        retained_ratio = (total - cumsum) / total
        mask = retained_ratio >= self.config.rho_s
        if mask.any():
            k_s = mask.sum().item()
        else:
            k_s = 1
        return max(1, k_s)

    def _find_k_temporal(self, S: torch.Tensor) -> int:
        """
        Eq. 6: 找最小 k_m 使得 top-k_m 能量 ≥ ρ_m * total.
        """
        energy = S ** 2
        total = energy.sum()
        if total == 0:
            return 1

        cumsum_ratio = energy.cumsum(0) / total
        mask = cumsum_ratio >= self.config.rho_m
        if mask.any():
            k_m = mask.to(torch.long).argmax().item() + 1
        else:
            k_m = len(S)
        return max(1, k_m)


# ─────────────────────────────────────────────────────────────
# 诊断/分析工具
# ─────────────────────────────────────────────────────────────

def compute_temporal_energy_ratio(
    noise_inv: torch.Tensor,
    rho_s: float = 0.1,
    rho_m: float = 0.9,
) -> float:
    """
    计算 SVD 滤波后 temporal 成分的能量占比.
    向后兼容 V1 接口.

    Returns:
        ratio: filtered_energy / original_energy (0~1)
    """
    if noise_inv.dim() == 5:
        noise_inv = noise_inv[0]

    if noise_inv.dtype in (torch.bfloat16, torch.float16):
        noise_inv = noise_inv.float()

    original_energy = (noise_inv ** 2).sum().item()
    if original_energy == 0:
        return 0.0

    config = SVDFilterConfig(rho_s=rho_s, rho_m=rho_m)
    svd_filter = SVDFilter(config=config)
    filtered = svd_filter._process_single(noise_inv)
    filtered_energy = (filtered ** 2).sum().item()

    return filtered_energy / original_energy


# ─────────────────────────────────────────────────────────────
# V1 兼容函数 (保留接口)
# ─────────────────────────────────────────────────────────────

def compute_svd_statistics(noise: torch.Tensor) -> Dict[str, torch.Tensor]:
    """V1 兼容: 计算 SVD 统计信息."""
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
