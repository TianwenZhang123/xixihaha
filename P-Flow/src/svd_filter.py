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

    两阶段流水线:
        Stage 1: Spatial Decontenting (去内容/外观)
        Stage 2: Temporal Retention (保运动动态)
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

    def filter(self, noise_inv: torch.Tensor, return_stats: bool = False):
        """
        主入口: 对反演噪声进行 SVD 滤波.

        Args:
            noise_inv: (B, C, F, H, W) 或 (C, F, H, W)
            return_stats: 是否同时返回 SVD 统计信息 (用于 TSR 计算)

        Returns:
            若 return_stats=False: 滤波后的运动先验噪声 (原行为)
            若 return_stats=True: (eta_temporal, stats_dict)
                stats_dict 包含: S_temporal, k_m, k_s
        """
        has_batch = noise_inv.dim() == 5
        if has_batch:
            results = []
            all_stats = []
            for b in range(noise_inv.shape[0]):
                result, stats = self._process_single(noise_inv[b], return_stats=True)
                results.append(result)
                all_stats.append(stats)
            eta_temporal = torch.stack(results, dim=0)
            if return_stats:
                return eta_temporal, all_stats[0]  # 单样本取第一个
            return eta_temporal
        else:
            result, stats = self._process_single(noise_inv, return_stats=True)
            if return_stats:
                return result, stats
            return result


    def _process_single(self, noise_inv: torch.Tensor, return_stats: bool = False):
        """
        单样本完整处理流水线.

        Args:
            noise_inv: (C, F, H, W)
            return_stats: 是否返回 SVD 统计信息

        Returns:
            若 return_stats=False: 处理后的噪声 (C, F, H, W)
            若 return_stats=True: (noise, stats_dict)
        """
        C, F, H, W = noise_inv.shape
        original_dtype = noise_inv.dtype
        cfg = self.config

        # SVD 需要 float32
        if noise_inv.dtype in (torch.bfloat16, torch.float16):
            noise_inv = noise_inv.float()

        # ── Stage 1: Spatial Decontenting ──
        noise_after_spatial, k_s = self._stage1_spatial(noise_inv)

        # ── Stage 2: Temporal Retention ──
        noise_temporal, S_temporal, k_m, Vh_m = self._stage2_temporal(
            noise_after_spatial, return_vh=True
        )

        result = noise_temporal  # v1 模式: 直接返回 Stage 2 输出

        # 恢复原始 dtype
        if result.dtype != original_dtype:
            result = result.to(original_dtype)
        if Vh_m.dtype != original_dtype:
            Vh_m = Vh_m.to(original_dtype)

        if return_stats:
            stats = {
                "S_temporal": S_temporal,
                "k_m": k_m,
                "k_s": k_s,
                "Vh_temporal": Vh_m,  # (k_m, F) 用于运动方向过滤
            }
            return result, stats
        return result

    # ─────────────────────────────────────────────────────────────
    # Stage 1: 空间去内容
    # ─────────────────────────────────────────────────────────────

    def _stage1_spatial(self, noise_inv: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Stage 1 - Spatial Decontenting (Eq. 4-5):
            Reshape to (C*F, H*W), SVD, 去除 top-k_s 空间主成分

        Returns:
            (filtered_noise, k_s):
                filtered_noise: 去外观后的噪声 (C, F, H, W)
                k_s: 移除的成分数
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

        return noise_filtered.reshape(C, F, H, W), k_s

    # ─────────────────────────────────────────────────────────────
    # Stage 2: 时间保运动
    # ─────────────────────────────────────────────────────────────

    def _stage2_temporal(
        self, noise_spatial: torch.Tensor, return_vh: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Stage 2 - Temporal Retention (Eq. 6):
            Reshape to (C*H*W, F), SVD, 保留 top-k_m 时间主成分

        Args:
            return_vh: 若 True, 额外返回 Vh (用于运动方向过滤)

        Returns:
            (filtered_noise, singular_values, k_m, [Vh_m])
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

        if return_vh:
            return noise_temporal, S_m, k_m, Vh_m
        return noise_temporal, S_m, k_m

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
    # Progressive Multi-scale SVD (方向2, --svd_progressive)
    # ─────────────────────────────────────────────────────────────

    def filter_progressive(
        self, noise_inv: torch.Tensor, window_size: int = 8, stride: int = 4,
    ) -> torch.Tensor:
        squeeze_batch = False
        if noise_inv.dim() == 5:
            noise_inv = noise_inv[0]
            squeeze_batch = True
        _, F, _, _ = noise_inv.shape
        windows = []
        for start in range(0, F - window_size + 1, stride):
            end = start + window_size
            win_noise = noise_inv[:, start:end, :, :]
            win_spatial, _ = self._stage1_spatial(win_noise)
            win_temporal, S_m, k_m = self._stage2_temporal(win_spatial)
            weight = k_m / window_size  # 成分越丰富权重越高
            windows.append((start, win_temporal, weight, k_m))

        eta_fused = torch.zeros_like(noise_inv)
        weight_sum = torch.zeros(F, device=noise_inv.device)
        for start, win_eta, w, k_m in windows:
            end = start + window_size
            eta_fused[:, start:end, :, :] += w * win_eta
            weight_sum[start:end] += w
        weight_sum = weight_sum.clamp(min=1e-8)
        eta_fused = eta_fused / weight_sum.view(1, F, 1, 1)

        kms = [km for _, _, _, km in windows]
        logger.info(
            f"  [Progressive SVD] {len(windows)} windows "
            f"(size={window_size}, stride={stride}), k_m={kms}"
        )
        if squeeze_batch:
            eta_fused = eta_fused.unsqueeze(0)
        return eta_fused



