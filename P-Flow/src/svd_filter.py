"""
SVD-based Motion Prior Extraction — V2 (大刀阔斧版).

核心改进 (相比 V1):
    1. Renormalization: SVD 滤波后强制标准化到 N(0,1)，消除方差塌缩问题
       - V1 的 η_temporal std≈0.28-0.41，严重偏离扩散模型的 N(0,1) 假设
       - V2 在滤波后做 (x - mean) / std 重标准化

    2. 多尺度时间分解 (Multi-Scale Temporal Decomposition):
       - 将时间奇异值分为 低频段(top-k_low) 和 高频段(k_low+1 ~ k_m)
       - 低频 = 全局运动方向/趋势 (与 v7e 精确文本描述冲突)
       - 高频 = 局部运动细节/时序纹理 (文本难以描述的微动态)
       - 可选：只注入高频段，避免与文本 motion 描述冲突

    3. 频谱感知 (Spectrum-Aware):
       - 分析奇异值分布的"拐点"(knee point) 自动确定低频/高频分界
       - 提供诊断接口，方便论文分析

    4. 保留 V1 兼容模式:
       - mode="v1" 退化为 V1 行为 (无 renorm, 无频段分离)
       - mode="renorm" 仅加 renorm
       - mode="highfreq" 仅注入高频 + renorm (推荐)

Paper Context:
    原始两阶段 SVD (Section 3.3, Eq. 4-7) 保留不变，
    V2 在 Stage 2 输出之后增加 频段分解 + 标准化 两个后处理步骤。
"""

import torch
import math
from typing import Any, Tuple, Dict, Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class SVDFilterConfig:
    """SVD 滤波器配置 (V2)."""

    # ── Stage 1: 空间去内容 ──
    rho_s: float = 0.1           # 空间能量阈值 (去除 top-k_s 后保留 ≥ ρ_s)

    # ── Stage 2: 时间保运动 ──
    rho_m: float = 0.9           # 时间能量阈值 (保留 top-k_m 使得能量 ≥ ρ_m)

    # ── V2 新增: 滤波模式 ──
    mode: Literal["v1", "renorm", "rescale", "highfreq", "adaptive"] = "adaptive"
    # v1       = 原始行为 (向后兼容)
    # renorm   = V1 + renormalization (强制 N(0,1))
    # rescale  = Direction-Preserving Rescale (等比缩放，保留方向结构)
    # highfreq = 仅高频段 + renormalization
    # adaptive = 自动判断 (推荐): 根据 motion_strength 决定模式

    # ── V2 新增: 频段分离参数 ──
    low_freq_ratio: float = 0.3  # 低频段占 k_m 的比例 (默认前30%为低频)
    knee_auto: bool = True       # 是否自动检测拐点 (优先于 low_freq_ratio)

    # ── V2 新增: Renormalization ──
    renorm_target_std: float = 1.0   # 目标标准差 (扩散模型假设 N(0,1))
    renorm_target_mean: float = 0.0  # 目标均值

    # ── V2 新增: Direction-Preserving Rescale ──
    rescale_target_effective: float = 0.0234  # 目标有效注入量 (v1 中位数 √0.004 × 0.37)
    rescale_only_boost: bool = True           # True: 只放大不缩小 (保护已足够的样本)

    # ── V2 新增: 自适应模式参数 ──
    motion_strength_threshold: float = 0.15  # motion_strength > 此值才注入
    # motion_strength 定义: temporal 能量 / 原始能量

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
            2. SVDFilter(rho_s=0.1, rho_m=0.9, mode="adaptive")  # 向后兼容
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

        # ── Stage 3: Frequency Selection + Renormalization ──
        mode = self._resolve_mode(noise_inv, noise_temporal)
        result = self._stage3_postprocess(
            noise_inv, noise_after_spatial, noise_temporal, S_temporal, k_m, mode
        )

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

    def _stage3_postprocess(
        self,
        noise_original: torch.Tensor,
        noise_after_spatial: torch.Tensor,
        noise_temporal: torch.Tensor,
        S_temporal: torch.Tensor,
        k_m: int,
        mode: str,
    ) -> torch.Tensor:
        """
        Stage 3 - 频段选择 + 标准化.

        根据 mode 执行不同策略:
            - v1: 直接返回 Stage 2 输出 (原始行为)
            - renorm: Stage 2 输出 + renormalization
            - highfreq: 只保留高频时间成分 + renormalization
            - adaptive: 根据 motion_strength 动态选择
        """
        cfg = self.config

        if mode == "v1":
            # 完全兼容 V1，不做任何后处理
            logger.debug("  [Stage3] mode=v1, no postprocessing")
            return noise_temporal

        elif mode == "renorm":
            # 仅做 renormalization
            result = self._renormalize(noise_temporal)
            logger.debug(
                f"  [Stage3] mode=renorm, "
                f"pre_std={noise_temporal.std():.4f} → post_std={result.std():.4f}"
            )
            return result

        elif mode == "rescale":
            # Direction-Preserving Rescale: 等比缩放，保留方向间相对结构
            result = self._rescale(noise_temporal)
            logger.debug(
                f"  [Stage3] mode=rescale, "
                f"pre_std={noise_temporal.std():.4f} → post_std={result.std():.4f}"
            )
            return result

        elif mode == "highfreq":
            # 高频段提取 + renormalization
            C, F, H, W = noise_after_spatial.shape
            noise_2d = noise_after_spatial.reshape(C * H * W, F)
            U_m, S_m, Vh_m = torch.linalg.svd(noise_2d, full_matrices=False)

            # 确定低频/高频分界
            k_low = self._find_knee_point(S_m, k_m)

            # 高频段: 从 k_low 到 k_m
            if k_low < k_m:
                noise_highfreq = (
                    U_m[:, k_low:k_m]
                    @ torch.diag(S_m[k_low:k_m])
                    @ Vh_m[k_low:k_m, :]
                )
                noise_highfreq = noise_highfreq.reshape(C, F, H, W)
            else:
                # 边界情况: 如果 knee 在 k_m 之后，退化为 renorm 模式
                logger.warning(
                    f"  [Stage3] k_low({k_low}) >= k_m({k_m}), "
                    f"fallback to renorm mode"
                )
                noise_highfreq = noise_temporal

            result = self._renormalize(noise_highfreq)
            logger.debug(
                f"  [Stage3] mode=highfreq, "
                f"k_low={k_low}, k_high={k_low}~{k_m}, "
                f"pre_std={noise_highfreq.std():.4f} → post_std={result.std():.4f}"
            )
            return result

        elif mode == "skip":
            # motion 太弱，跳过 SVD，返回 None 信号 (由调用方处理)
            logger.info("  [Stage3] mode=skip (motion too weak)")
            return noise_original  # 返回原始的，让 pipeline 层决定是否跳过

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ─────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────

    def _resolve_mode(
        self, noise_original: torch.Tensor, noise_temporal: torch.Tensor
    ) -> str:
        """
        自适应模式解析.

        如果 config.mode == "adaptive":
            - 计算 motion_strength (temporal 能量 / 原始能量)
            - 强运动 → "highfreq" (分离高频, 避免文本冲突)
            - 弱运动 → "skip" (SVD 提取物无意义)
        """
        cfg = self.config
        if cfg.mode != "adaptive":
            return cfg.mode

        # 计算 motion strength
        original_energy = (noise_original ** 2).sum().item()
        if original_energy == 0:
            return "skip"

        temporal_energy = (noise_temporal ** 2).sum().item()
        motion_strength = temporal_energy / original_energy

        logger.debug(f"  [Adaptive] motion_strength={motion_strength:.4f}")

        if motion_strength < cfg.motion_strength_threshold:
            return "skip"
        else:
            return "highfreq"

    def _find_knee_point(self, S: torch.Tensor, k_m: int) -> int:
        """
        找到奇异值分布的"拐点" (knee point).

        方法: 最大曲率法 (maximum curvature)
            - 对 log(σ) 序列计算二阶差分
            - 最大二阶差分位置 = 拐点

        如果关闭自动检测，使用 low_freq_ratio 的固定比例.

        Returns:
            k_low: 低频段结束索引 (exclusive), 即 [0, k_low) 为低频
        """
        cfg = self.config

        if not cfg.knee_auto or k_m <= 2:
            # 使用固定比例
            k_low = max(1, int(k_m * cfg.low_freq_ratio))
            return k_low

        # 自动拐点检测
        S_valid = S[:k_m]
        if len(S_valid) <= 3:
            return max(1, k_m // 3)

        # 对数域二阶差分 (避免数值问题)
        log_S = torch.log(S_valid.clamp(min=1e-10))
        if len(log_S) < 3:
            return 1

        # 一阶差分 (应全为负，因为奇异值递减)
        diff1 = log_S[1:] - log_S[:-1]
        # 二阶差分 (拐点处为最大值)
        diff2 = diff1[1:] - diff1[:-1]

        if len(diff2) == 0:
            return max(1, k_m // 3)

        # 找最大曲率位置
        knee_idx = diff2.abs().argmax().item() + 1  # +1 因为差分偏移

        # 合理性约束: 拐点不应太靠前也不应太靠后
        knee_idx = max(1, min(knee_idx, k_m - 1))

        return knee_idx

    def _rescale(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Direction-Preserving Rescale (方案 C).

        核心思路:
            不做 (x-mean)/std 的全量归一化，而是做等比缩放:
            - 计算当前 effective injection = sqrt(alpha) * std
            - 如果 current_effective < target_effective，按比例放大
            - 如果 current_effective >= target_effective 且 only_boost=True，不动

        相比 renorm 的优势:
            1. 保留方向信息的相对结构 (各维度比例不变)
            2. 只拉升信号不足的样本，不干扰已经好的样本
            3. 不同样本仍然有不同的注入量 (保持 v1 自适应特性)

        数学等价:
            scale = target_effective / current_effective
            η_out = η_in * scale
            实际 direction_shift = sqrt(alpha) * η_out.std() / η_rand.std()
                                 = sqrt(alpha) * (η_in.std() * scale) / 1.0
                                 = target_effective
        """
        cfg = self.config
        current_std = noise.std()

        if current_std < 1e-8:
            logger.warning("  [Rescale] near-zero std, returning as-is")
            return noise

        # 当前有效注入量 (在 pipeline 中 blend 后的等效影响)
        # pipeline 中: η = √α·η_temporal + √(1-α)·η_random
        # direction_shift ≈ √α · η_temporal.std() (因为 η_random.std() ≈ 1)
        # 注意: 这里的 alpha 是 pipeline 层的，SVD filter 不知道具体值
        # 但我们可以直接用 std 来做等比缩放:
        #   目标 std = target_effective / √α
        # 不过更优雅的做法是: 让 rescale 只关注 std 的绝对值
        # v1 baseline 中位 std ≈ 0.37, 我们目标让所有样本 std >= 0.37
        target_std = cfg.rescale_target_effective / math.sqrt(0.004)  # ≈ 0.37

        if cfg.rescale_only_boost and current_std >= target_std:
            # 当前已经足够强，不缩放
            logger.debug(
                f"  [Rescale] current_std={current_std:.4f} >= "
                f"target_std={target_std:.4f}, no scaling"
            )
            return noise

        # 等比缩放: 保留方向，只调整幅度
        scale = target_std / current_std
        result = noise * scale

        logger.debug(
            f"  [Rescale] scale={scale:.4f}, "
            f"std: {current_std:.4f} → {result.std():.4f}"
        )
        return result

    def _renormalize(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Renormalization: 将噪声标准化到目标分布.

        x_out = (x - mean(x)) / std(x) * target_std + target_mean

        这是 V2 的核心修复:
        - V1 的 η_temporal 方差≈0.1 (std≈0.35)，严重偏离 N(0,1)
        - 扩散模型的 ODE/SDE 求解器假设初始噪声 ~ N(0,1)
        - 不做 renorm → 等效于缩小了 signal strength → 模型"几乎忽略"先验
        """
        cfg = self.config
        std = noise.std()
        mean = noise.mean()

        if std < 1e-8:
            # 退化情况: 几乎全零 → 返回标准高斯
            logger.warning("  [Renorm] near-zero std, returning random noise pattern")
            return torch.randn_like(noise) * cfg.renorm_target_std + cfg.renorm_target_mean

        result = (noise - mean) / std * cfg.renorm_target_std + cfg.renorm_target_mean
        return result

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

def compute_svd_diagnostics(noise_inv: torch.Tensor, config: Optional[SVDFilterConfig] = None) -> Dict[str, Any]:
    """
    综合诊断: 分析 SVD 滤波的各项特征.

    用于论文分析和实验调优. 输出包含:
        - 奇异值分布
        - 各频段能量占比
        - motion_strength
        - 推荐模式
        - renorm 前后的统计量
    """
    if config is None:
        config = SVDFilterConfig()

    if noise_inv.dim() == 5:
        noise_inv = noise_inv[0]

    C, F, H, W = noise_inv.shape
    original_dtype = noise_inv.dtype
    if noise_inv.dtype in (torch.bfloat16, torch.float16):
        noise_inv = noise_inv.float()

    svd_filter = SVDFilter(config=config)

    # Stage 1
    noise_spatial = svd_filter._stage1_spatial(noise_inv)

    # Stage 2 (获取完整 SVD 分解)
    noise_2d = noise_spatial.reshape(C * H * W, F)
    U_m, S_m, Vh_m = torch.linalg.svd(noise_2d, full_matrices=False)
    k_m = svd_filter._find_k_temporal(S_m)
    k_m = min(k_m, len(S_m))

    noise_temporal = U_m[:, :k_m] @ torch.diag(S_m[:k_m]) @ Vh_m[:k_m, :]
    noise_temporal = noise_temporal.reshape(C, F, H, W)

    # 频段分析
    k_low = svd_filter._find_knee_point(S_m, k_m)

    # 各段能量
    energy_total = (S_m ** 2).sum().item()
    energy_low = (S_m[:k_low] ** 2).sum().item() if k_low > 0 else 0
    energy_high = (S_m[k_low:k_m] ** 2).sum().item() if k_low < k_m else 0
    energy_residual = (S_m[k_m:] ** 2).sum().item() if k_m < len(S_m) else 0

    # motion strength
    original_energy = (noise_inv ** 2).sum().item()
    temporal_energy = (noise_temporal ** 2).sum().item()
    motion_strength = temporal_energy / original_energy if original_energy > 0 else 0

    # Renorm 效果
    noise_renormed = svd_filter._renormalize(noise_temporal)

    # 高频段
    if k_low < k_m:
        noise_hf = U_m[:, k_low:k_m] @ torch.diag(S_m[k_low:k_m]) @ Vh_m[k_low:k_m, :]
        noise_hf = noise_hf.reshape(C, F, H, W)
        noise_hf_renormed = svd_filter._renormalize(noise_hf)
    else:
        noise_hf = noise_temporal
        noise_hf_renormed = noise_renormed

    return {
        # 基本信息
        "shape": (C, F, H, W),
        "k_m": k_m,
        "k_low": k_low,
        "total_temporal_components": len(S_m),

        # 奇异值
        "singular_values": S_m.cpu(),
        "singular_values_energy_pct": ((S_m ** 2) / energy_total * 100).cpu(),

        # 频段能量分布
        "energy_low_freq_pct": energy_low / energy_total * 100,
        "energy_high_freq_pct": energy_high / energy_total * 100,
        "energy_residual_pct": energy_residual / energy_total * 100,

        # 运动强度
        "motion_strength": motion_strength,
        "recommended_mode": "highfreq" if motion_strength >= config.motion_strength_threshold else "skip",

        # 标准化前后对比
        "temporal_raw_mean": noise_temporal.mean().item(),
        "temporal_raw_std": noise_temporal.std().item(),
        "temporal_renormed_mean": noise_renormed.mean().item(),
        "temporal_renormed_std": noise_renormed.std().item(),

        # 高频段统计
        "highfreq_raw_std": noise_hf.std().item(),
        "highfreq_renormed_std": noise_hf_renormed.std().item(),
    }


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

    config = SVDFilterConfig(rho_s=rho_s, rho_m=rho_m, mode="v1")
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
