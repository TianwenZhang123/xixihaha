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
        noise_after_spatial, k_s, eta_spatial = self._stage1_spatial(noise_inv)

        # ── Stage 2: Temporal Retention ──
        noise_temporal, S_temporal, k_m = self._stage2_temporal(noise_after_spatial)

        result = noise_temporal  # v1 模式: 直接返回 Stage 2 输出

        # 恢复原始 dtype
        if result.dtype != original_dtype:
            result = result.to(original_dtype)
        if eta_spatial.dtype != original_dtype:
            eta_spatial = eta_spatial.to(original_dtype)

        if return_stats:
            stats = {
                "S_temporal": S_temporal,
                "k_m": k_m,
                "k_s": k_s,
                "eta_spatial": eta_spatial,
            }
            return result, stats
        return result

    # ─────────────────────────────────────────────────────────────
    # Stage 1: 空间去内容
    # ─────────────────────────────────────────────────────────────

    def _stage1_spatial(self, noise_inv: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Stage 1 - Spatial Filtering (Eq. 4-5):
            Reshape to (C*F, H*W), SVD, 去除 top-k_s 空间主成分

        Returns:
            (filtered_noise, k_s, eta_spatial):
                filtered_noise: 去外观后的噪声 (C, F, H, W)
                k_s: 移除的成分数
                eta_spatial: 被移除的外观/内容分量 (C, F, H, W), 可用于 spatial blend
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
        eta_spatial = top_k_recon.reshape(C, F, H, W)
        logger.debug(f"  [Stage1] Spatial: removed top-{k_s} components")

        return noise_filtered.reshape(C, F, H, W), k_s, eta_spatial

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


# ─────────────────────────────────────────────────────────────
# Temporal Signal Reliability (TSR) — 自适应 α 的核心指标
# ─────────────────────────────────────────────────────────────

def compute_temporal_signal_reliability(
    eta_temporal: torch.Tensor,
    S_temporal: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    计算 Temporal Signal Reliability (TSR), 用于自适应 blend 系数 α.

    TSR 由两个分量组成:

        1. Temporal Concentration Ratio (TCR):
           η_temporal 在时间奇异值谱上的能量集中度.
           定义: TCR = S_1^2 / sum(S_i^2)  (第一大时间奇异值的能量占比)
           - 高 TCR → 运动信号集中、结构化 → 可靠
           - 低 TCR → 运动信号分散、近似噪声 → 不可靠

        2. Temporal Autocorrelation (TAC):
           η_temporal 相邻帧间的平均余弦相似度.
           - 高 TAC → 帧间连贯 (真实运动) → 可靠
           - 低 TAC → 帧间独立 (近似噪声) → 不可靠

    TSR = sigmoid_norm(TCR) * TAC

    学术 motivation:
        SVD 两阶段滤波提取 η_temporal 时, 假设其编码了有效的运动先验.
        但对于低运动场景 (如静态室内), SVD 提取的 temporal 分量主要是残差噪声,
        而非结构化运动信号. TSR 量化了这一信号的可靠程度, 使得 blend 系数 α
        能根据信号质量自适应调节: 可靠信号→大α, 不可靠信号→小α甚至0.

    Args:
        eta_temporal: SVD 滤波后的时序噪声, shape (C, F, H, W) 或 (B, C, F, H, W)
        S_temporal: 可选, Stage 2 的时间奇异值 (避免重复 SVD 计算)

    Returns:
        dict with keys:
            tcr: Temporal Concentration Ratio ∈ (0, 1]
            tac: Temporal Autocorrelation ∈ [-1, 1]
            tsr: Temporal Signal Reliability ∈ [0, 1]
    """
    if eta_temporal.dim() == 5:
        eta_temporal = eta_temporal[0]

    original_dtype = eta_temporal.dtype
    if eta_temporal.dtype in (torch.bfloat16, torch.float16):
        eta_temporal = eta_temporal.float()

    C, F, H, W = eta_temporal.shape

    # ── 1. TCR: Temporal Concentration Ratio ──
    if S_temporal is None:
        # 需要重新做 SVD (通常在 filter() 中已计算, 可以通过参数传入)
        noise_2d = eta_temporal.reshape(C * H * W, F)
        _, S_m, _ = torch.linalg.svd(noise_2d, full_matrices=False)
    else:
        S_m = S_temporal

    energy = S_m ** 2
    total_energy = energy.sum()
    if total_energy > 0:
        tcr = (energy[0] / total_energy).item()
    else:
        tcr = 0.0

    # ── 2. TAC: Temporal Autocorrelation ──
    # 计算相邻帧 (dim=1) 之间的平均余弦相似度
    if F < 2:
        tac = 1.0  # 单帧视频, 退化为1
    else:
        # 展平空间维度: (C, F, H*W) → 逐帧计算
        frames = eta_temporal.reshape(C, F, H * W)  # (C, F, H*W)
        # 相邻帧: frame_i 和 frame_{i+1}
        cos_sims = []
        for t in range(F - 1):
            f_curr = frames[:, t, :].flatten()   # (C*H*W,)
            f_next = frames[:, t + 1, :].flatten()  # (C*H*W,)
            norm_curr = f_curr.norm()
            norm_next = f_next.norm()
            if norm_curr > 1e-8 and norm_next > 1e-8:
                cos_sim = torch.nn.functional.cosine_similarity(
                    f_curr.unsqueeze(0), f_next.unsqueeze(0)
                ).item()
                cos_sims.append(cos_sim)
        tac = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

    # ── 3. TSR: 综合可靠性 ──
    # TCR 归一化: 用 sigmoid 将 (0,1] 映射到更平滑的区间
    # 对于 TCR, 典型值范围约 0.05~0.3 (5个消融样本的观测)
    # 我们用 sigmoid 中心设在 0.1 (区分弱/强信号)
    tcr_normalized = torch.sigmoid(torch.tensor(10.0 * (tcr - 0.1))).item()

    # TAC 归一化: 直接用 max(0, TAC), 因为负 TAC 说明帧间反向, 信号不可靠
    tac_normalized = max(0.0, tac)

    tsr = tcr_normalized * tac_normalized

    return {
        "tcr": tcr,
        "tac": tac,
        "tsr": tsr,
    }
