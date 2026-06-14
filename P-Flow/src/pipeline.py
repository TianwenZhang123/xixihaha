"""
P-Flow Unified Pipeline.

一个管线搞定所有配置，通过 flag 开关各改动点：

    Flag              对应改动点                    效果
    ─────────────────────────────────────────────────────────────
    --inversion       Flow Matching Inversion      从参考视频反演噪声
    --svd             SVD Two-stage Filtering      空间去内容 + 时间保运动
    --blend           Noise Prior Blending         混合运动噪声与随机噪声
    --iter N          Iterative VLM Optimization   N轮VLM反馈优化prompt
    --midpoint        Midpoint ODE Solver          二阶中点法(替代Euler)
    --composite       Vertical Composite           三面板拼接送VLM对比


组合示例：
    baseline:     无任何flag → caption + 一次生成
    +noise_prior: --inversion --svd --blend → 噪声先验引导
    +iteration:   --iter 10 → 迭代优化
    full pflow:   --inversion --svd --blend --iter 10 --composite

"""

import os
import json
import time
import shutil
import math
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass

import torch

from .distributed import setup_single_gpu, load_model_single_gpu, cleanup_gpu_memory
from .flow_matching import FlowMatchingInverter, encode_video_to_latents
from .svd_filter import SVDFilter, SVDFilterConfig, compute_temporal_energy_ratio, compute_svd_diagnostics
from .video_utils import (
    load_video, save_video_tensor, normalize_video, denormalize_video,
    create_vertical_composite,
)
from .vlm_client import create_vlm_client

logger = logging.getLogger(__name__)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, work, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, "
    "deformed, blurry, watermark"
)


@dataclass
class PFlowConfig:
    """所有可配置参数，一个 dataclass 搞定。"""

    # ── 模型 ──
    t2v_path: str = "models/Wan2.1-T2V-1.3B-Diffusers"
    dtype: str = "bfloat16"

    # ── 视频生成 ──
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 15
    guidance_scale: float = 5.0
    num_inference_steps: int = 30

    # ── 改动点开关 ──
    use_inversion: bool = False    # Flow Matching Inversion
    use_svd: bool = False          # SVD Filtering
    use_blend: bool = False        # Noise Blending (α mixing)
    use_iter: bool = False         # Iterative VLM Optimization
    use_midpoint: bool = False     # Midpoint ODE Solver
    use_composite: bool = False    # Vertical Composite for VLM
    # ── Noise Prior 参数 ──
    alpha: float = 0.003           # 混合权重 (√α·η_temporal + √(1-α)·η_random)
                                  # P-Flow论文用 0.001, 我们V2 renorm后有效信号增强约3x
                                  # 推荐搜索范围: 0.001 ~ 0.01
    rho_s: float = 0.1            # 空间SVD阈值 (去内容)
    rho_m: float = 0.9            # 时间SVD阈值 (保运动)
    inversion_steps: int = 50     # 反演ODE步数
    use_fast_svd: bool = True     # 使用 randomized SVD 加速滤波 (对大 latent 快 2-3x)
    temporal_energy_threshold: float = 0.0  # 自适应SVD跳过阈值 (0=禁用; >0时低于此值跳过SVD)

    # ── V2 SVD 新增参数 ──
    svd_mode: str = "adaptive"    # SVD 滤波模式: v1 / renorm / rescale / highfreq / adaptive
    svd_low_freq_ratio: float = 0.3  # 低频段占比 (highfreq 模式用)
    svd_knee_auto: bool = True    # 自动拐点检测
    svd_motion_threshold: float = 0.15  # 运动强度阈值 (adaptive 模式)
    svd_diagnostics: bool = True  # 是否保存 SVD 诊断信息 (诊断期间默认开启)

    # ── Quality-Gated Alpha (方案 B) ──
    quality_gated_alpha: bool = False   # 是否启用 per-sample adaptive alpha
    qga_base_alpha: float = 0.004      # 基础 alpha (当 quality_gated_alpha=False 时不影响原有 self.config.alpha)
    qga_low_mult: float = 0.25         # quality=0 时的 alpha 倍率 (base_alpha * 0.25)
    qga_high_mult: float = 2.5         # quality=1 时的 alpha 倍率 (base_alpha * 2.5)

    # ── 方向 C: 频域噪声重塑 (Spectrum-Aligned Noise) ──
    freq_reshape: bool = False          # 是否启用频域重塑 (替代 linear blend)
    freq_reshape_beta: float = 1.0     # 重塑强度: 0=不重塑(纯随机), 1=完全匹配频谱形状
                                       # 推荐搜索范围: 0.3~1.0

    # ── 方向 D: Std-Gated Adaptive Alpha (SGA) ──
    adaptive_alpha: bool = False        # 是否启用 per-sample adaptive alpha
    sga_target_std: float = 0.30       # 目标标准差 (中位数附近, 根据实测样本 std 分布设定)
    sga_alpha_min: float = 0.001       # alpha 下界 (防止完全不注入)
    sga_alpha_max: float = 0.010       # alpha 上界 (防止过度注入)

    # ── 方向 E: Prompt-Orthogonal Decomposition Injection (PODI) ──
    podi: bool = False                  # 是否启用 PODI (只注入与 prompt 对齐的 η_temporal 分量)
    podi_alpha: float = 0.004          # PODI 注入强度 (默认与 baseline alpha 相同, 公平对比)
                                       # 因为注入的是安全分量, 后续可尝试更大值 (0.008~0.02)
    podi_min_alignment: float = 0.01   # 最小对齐度阈值 (alignment < 此值则完全放弃注入)
    podi_proj_mode: str = "mean_pool"  # prompt embedding → latent 投影方式:
                                       #   mean_pool: 对 seq_len 维度均值池化后线性插值到 latent 维度
                                       #   last_token: 取最后一个非 padding token
                                       #   weighted: attention-weighted pooling

    # ── 方向 F: Channel-Energy Gated Injection (CEGI) ──
    cegi: bool = False                  # 是否启用 CEGI (通道能量门控注入)
    cegi_top_k: int = 4                # 注入的 channel 数 (top-k temporal energy channels)
    cegi_alpha: float = 0.02           # 被选中 channel 的注入强度 (集中注入, 比 baseline 大)
    cegi_residual_alpha: float = 0.0   # 未选中 channel 的注入强度 (0=纯随机, >0 保留微弱 prior)

    # ── 方向 G: Multi-Scale Temporal Decomposition Injection (MSTDI) ──
    mstdi: bool = False                 # 是否启用 MSTDI (多尺度时序分解注入)
    mstdi_levels: int = 3              # 金字塔层数 (2=粗/细, 3=粗/中/细, 4=四层)
    mstdi_alpha_base: float = 0.05     # 最粗层的 alpha (强注入控制全局运动)
    mstdi_alpha_decay: float = 0.25    # 每层 alpha 衰减比例 (alpha[i+1] = alpha[i] * decay)
                                       # 默认: L0=0.05, L1=0.0125, L2=0.003125 → 高频几乎不注入

    # ── 方向 H: Temporal Phase Injection (TPI) ──
    tpi: bool = False                   # 是否启用 TPI (时间相位注入)
    tpi_gamma: float = 0.5             # 相位注入强度 (0=纯随机相位, 1=完全用参考相位)
    tpi_freq_min: int = 1              # 最小注入频率 bin (跳过 DC=0)
    tpi_freq_max: int = -1             # 最大注入频率 bin (-1=全部)

    # ── 方向 I: Orthogonal Complement Suppression (OCS) ──
    ocs: bool = False                   # 是否启用 OCS (正交补空间抑制)
    ocs_top_k: int = 3                 # SVD 保留的主成分数
    ocs_suppress_ratio: float = 0.5    # 正交空间的抑制比例 (0=不抑制, 1=完全移除)

    # ── 灰盒: Latent Trajectory Soft Anchor (旧方案，已废弃) ──
    trajectory_anchor: bool = False     # 是否启用轨迹锚定 (旧方案: position lerp，已证明失败)
    anchor_beta_max: float = 0.3       # 最大锚定强度 β_max (推荐搜索: 0.1 ~ 0.5)
    anchor_schedule: str = "cosine_decay"  # β 退火调度: cosine_decay / linear_decay / constant / warmup_decay
    anchor_cache_every_n: int = 1      # inversion 轨迹缓存间隔 (1=全部, 2=隔一步; 用于显存优化)
    anchor_quality_gate: bool = True    # 是否启用轨迹质量门控 (基于 η_temporal 帧间余弦相似度)
    anchor_quality_threshold: float = 0.05  # 帧间 cos 阈值: mean_cos < 此值则跳过 anchor
    anchor_cos_threshold: float = 0.2   # >0 时启用 cos-proportional β 模式 (旧方案2)

    # ── L3 V2: Velocity Direction Anchor (VDA) ──
    # 核心思想: 不做位置 lerp (已证明失败), 改做速度方向微调
    # 数学: 在 Flow ODE 每步去噪后, 用参考轨迹的速度方向信息微调当前 latent
    #   v_ref = (z_ref[t-dt] - z_ref[t]) / dt  (反演速度, 取反=生成方向)
    #   v_gen ≈ (z_gen[t+dt] - z_gen[t]) / dt   (差分估计的生成速度)
    #   v_ref_⊥ = v_ref - (v_ref·v_gen/‖v_gen‖²)·v_gen  (参考速度的正交分量)
    #   Δz = γ · v_ref_⊥ · dt  (方向修正冲量)
    #   z_adjusted = z_current + Δz
    #
    # 与 L2 的关系:
    #   L2 提供起点偏置 (SVD-blended z_T), L3 提供过程方向引导
    #   两者语义正交互补, 不要求起点对齐
    velocity_anchor: bool = False          # 是否启用 VDA (Velocity Direction Anchor)
    vda_gamma: float = 0.03               # VDA 方向引导强度 γ (推荐搜索: 0.01 ~ 0.10)
    # γ 越大, 参考速度方向的影响越强; 太大会偏离 ODE 流形
    vda_schedule: str = "middle_peak"      # γ 调度策略:
    #   constant: γ 恒定
    #   middle_peak: 中间阶段最强, 两端弱 (推荐; 前后期轨迹收敛不需要强引导)
    #   warmup_decay: 先升后降
    #   cosine_decay: 从 γ 递减到 0
    vda_use_perp_only: bool = True         # True: 只注入正交分量 (不改变速度大小); False: 混合投影+正交
    vda_parallel_weight: float = 0.1      # vda_use_perp_only=False 时, 平行分量注入权重 (0.1=微弱)
    vda_quality_gate: bool = True         # 是否启用质量门控 (基于 motion_strength)
    vda_quality_scale: bool = True        # True: 门控值映射为 γ 缩放因子 (不硬跳过); False: 硬跳过
    # 软门控: motion_strength 低 → γ 缩小但不为零 (保留微弱引导)
    vda_norm_clamp: float = 0.0           # >0 时, 每步 Δz 的范数不超过 clamp * ‖z_current‖
    # 防止单步偏移过大导致生成崩溃 (推荐: 0.05~0.10; 0=不限制)
    vda_start_step: int = 1               # VDA 起始步 (step_index >= 此值才启用; 0=第一步就启用)
    # 设为 1 可以跳过第一步 (第一步速度估计不准确)
    vda_end_step: int = -1                # VDA 结束步 (step_index < 此值才启用; -1=到最后一步)
    # 后期步数 cos 自然高, VDA 作用减弱, 可以提前结束节省计算

    # ── 迭代优化参数 ──
    i_max: int = 10               # 迭代轮数

    # ── VLM ──
    vlm_provider: str = "local"
    vlm_model_path: str = "models/Qwen2.5-VL-7B-Instruct"

    # ── 负面 Prompt ──
    negative_prompt: str = ""            # 自定义负面 prompt (空=使用默认 NEGATIVE_PROMPT)
    negative_prompt_file: str = ""       # 按样本加载负面 prompt 的目录 (优先级高于 negative_prompt)

    # ── 其他 ──
    seed: int = 42

    def active_flags(self) -> List[str]:
        """返回当前启用的改动点列表。"""
        flags = []
        if self.use_inversion:
            flags.append("inversion")
        if self.use_svd:
            flags.append("svd")
        if self.use_blend:
            if self.cegi:
                flags.append(f"blend(CEGI: top_k={self.cegi_top_k}, α_inject={self.cegi_alpha}, α_residual={self.cegi_residual_alpha})")
            elif self.mstdi:
                flags.append(f"blend(MSTDI: levels={self.mstdi_levels}, α_base={self.mstdi_alpha_base}, decay={self.mstdi_alpha_decay})")
            elif self.tpi:
                flags.append(f"blend(TPI: γ={self.tpi_gamma}, freq=[{self.tpi_freq_min},{self.tpi_freq_max}])")
            elif self.ocs:
                flags.append(f"blend(OCS: top_k={self.ocs_top_k}, suppress={self.ocs_suppress_ratio})")
            elif self.podi:
                flags.append(f"blend(PODI: α={self.podi_alpha}, min_align={self.podi_min_alignment})")
            elif self.adaptive_alpha:
                flags.append(f"blend(SGA: base={self.alpha}, target_std={self.sga_target_std})")
            elif self.freq_reshape:
                flags.append(f"blend(α={self.alpha})+freq_reshape(β={self.freq_reshape_beta})")
            else:
                flags.append(f"blend(α={self.alpha})")
        if self.trajectory_anchor:
            flags.append(f"trajectory_anchor(β_max={self.anchor_beta_max}, sched={self.anchor_schedule})")
        if self.velocity_anchor:
            flags.append(f"velocity_anchor(γ={self.vda_gamma}, sched={self.vda_schedule}, perp_only={self.vda_use_perp_only})")
        if self.use_iter:
            flags.append(f"iter({self.i_max})")
        if self.use_midpoint:
            flags.append("midpoint")
        if self.use_composite:
            flags.append("composite")
        return flags

    def experiment_name(self) -> str:
        """生成实验名称。"""
        flags = self.active_flags()
        if not flags:
            return "baseline"
        return "pflow_" + "_".join(f.split("(")[0] for f in flags)


class PFlowPipeline:
    """
    统一管线：baseline 和所有改动点共用一个类。

    通过 PFlowConfig 中的 flag 控制行为：
    - 所有 flag 关闭 = baseline (caption → 一次生成)
    - 开启不同 flag = 不同消融配置
    """

    def __init__(self, config: PFlowConfig):
        self.config = config
        self.device = setup_single_gpu()
        self.dtype = getattr(torch, config.dtype)

        self._pipe = None
        self._vlm_client = None
        self._prompt_embeds = None  # 缓存 prompt embedding (供 PODI 使用)

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = load_model_single_gpu(
                model_path=self.config.t2v_path,
                dtype=self.dtype,
                model_type="t2v",
            )
        return self._pipe

    @property
    def vlm_client(self):
        if self._vlm_client is None:
            vlm_cfg = {
                "provider": self.config.vlm_provider,
                "model_path": self.config.vlm_model_path,
                "temperature": 0.7,
                "max_tokens": 2048,
                "max_retries": 3,
                "use_video_mode": True,
                "lazy_load": True,
            }
            self._vlm_client = create_vlm_client(vlm_cfg)
        return self._vlm_client

    # ─────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        video_path: str,
        output_dir: str,
        caption: str = "",
        sample_id: int = 0,
    ) -> Dict[str, Any]:
        """
        运行管线。根据 config 中的 flag 自动决定执行哪些步骤。

        各步骤内部自己管理 no_grad 上下文。

        Args:
            video_path: 参考视频路径
            output_dir: 输出目录
            caption: 初始 caption (为空则用VLM生成)
            sample_id: 样本ID

        Returns:
            实验结果 dict
        """
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        seed = cfg.seed + sample_id
        generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)

        flags = cfg.active_flags()
        logger.info(f"[P-Flow] sample={sample_id}, flags={flags or 'baseline'}")

        # ── Step 1: 加载参考视频 ──
        ref_video = load_video(
            video_path,
            num_frames=cfg.num_frames,
            height=cfg.height,
            width=cfg.width,
            device=self.device,
        )

        # ── Step 2: 生成 caption (如果为空，调 VLM 描述参考视频) ──
        if not caption:
            logger.info("  [Caption] caption 为空，调用 VLM 描述参考视频...")
            caption = self.vlm_client.describe_video(video_path)
            if caption:
                logger.info(f"  [Caption] VLM 生成: {caption[:80]}...")
            else:
                logger.warning("  [Caption] VLM 生成失败，使用默认 caption")
                caption = "a video scene"
            # 保存生成的 caption
            caption_file = out / "vlm_caption.txt"
            caption_file.write_text(caption, encoding="utf-8")

        # ── Step 2.5: 解析负面 prompt ──
        neg_prompt = self._resolve_negative_prompt(sample_id)
        if neg_prompt != NEGATIVE_PROMPT:
            logger.info(f"  [NegPrompt] 使用自定义负面 prompt: {neg_prompt[:60]}...")

        # ── Step 3: 计算噪声先验 (如果启用) ──
        eta_temporal = None
        prompt_embeds_for_diag = None
        ref_latents_enc = None
        if cfg.use_inversion:
            eta_temporal, eta_inv_raw, ref_latents_enc, prompt_embeds_for_diag = \
                self._compute_noise_prior(ref_video, caption)
            # 缓存 prompt embedding 供 PODI 使用
            self._prompt_embeds = prompt_embeds_for_diag

        # ── Step 3.5: Prompt-Noise 方向冲突诊断 ──
        if eta_temporal is not None and prompt_embeds_for_diag is not None:
            self._diagnose_prompt_noise_conflict(
                eta_temporal, prompt_embeds_for_diag, caption, generator
            )

        # ── Step 3.6: 轨迹缓存 (灰盒 Trajectory Anchor / VDA) ──
        ref_trajectory = None
        if cfg.trajectory_anchor or cfg.velocity_anchor:
            if not cfg.use_inversion:
                logger.warning(
                    "  [Trajectory Anchor] trajectory_anchor=True 但 use_inversion=False, "
                    "自动启用 inversion 以获取参考轨迹"
                )
            # 需要参考 latent 和 prompt embedding
            # 如果已经在 Step 3 做过 inversion，复用 ref_latents_enc 和 prompt_embeds
            if ref_latents_enc is not None and prompt_embeds_for_diag is not None:
                ref_lat = ref_latents_enc
                p_emb = prompt_embeds_for_diag
            else:
                # 没做过 inversion，现在做一次 encode + embed
                ref_norm = normalize_video(ref_video).unsqueeze(0)
                ref_lat = encode_video_to_latents(self.pipe, ref_norm, self.device)
                p_emb = self._encode_prompt(caption)

            # 用 invert_with_trajectory 获取带缓存的轨迹
            traj_inverter = FlowMatchingInverter(
                pipe=self.pipe,
                num_inversion_steps=cfg.inversion_steps,
                guidance_scale=1.0,
                device=self.device,
            )
            _, ref_trajectory = traj_inverter.invert_with_trajectory(
                ref_lat, p_emb, p_emb,
                cache_every_n=cfg.anchor_cache_every_n,
            )

            # ── 轨迹质量门控 (V2: η_temporal 帧间余弦相似度) ──
            # 核心思路: 用 η_temporal 的帧间 cos 评估参考视频的运动质量
            #   - mean_cos 高 (>0.1): 参考视频有一致的运动方向 → 适合锚定
            #   - mean_cos 低/负: 参考视频运动混乱/无规律 → 不适合锚定
            # 注: 旧方案用相邻 ODE 轨迹点 cos，因 dt=1/50 太小导致永远≈1.0，已废弃
            #
            # 门控拒绝策略 (V3):
            #   拒绝后完全回退到 baseline（纯随机噪声生成），等价于从未做过 inversion。
            #   关键: 重置 generator seed，使生成阶段的随机状态与 baseline 完全一致。
            #   时间开销: 只多花 inversion 的时间（~1.5min），生成阶段不重复。
            # 注意: VDA 模式下跳过此硬门控, VDA 有自己的软门控 (vda_quality_scale)
            if cfg.anchor_quality_gate and eta_temporal is not None and not cfg.velocity_anchor:
                eta_gate = eta_temporal
                if eta_gate.dim() == 5:
                    if eta_gate.shape[2] > eta_gate.shape[1]:
                        eta_gate = eta_gate.permute(0, 2, 1, 3, 4)
                    num_frames_gate = eta_gate.shape[2]
                elif eta_gate.dim() == 4:
                    num_frames_gate = eta_gate.shape[1]
                    eta_gate = eta_gate.unsqueeze(0)
                else:
                    num_frames_gate = 0

                if num_frames_gate >= 2:
                    frame_cos_sims = []
                    for f in range(num_frames_gate - 1):
                        f1 = eta_gate[0, :, f, :, :].flatten()
                        f2 = eta_gate[0, :, f + 1, :, :].flatten()
                        cos = torch.nn.functional.cosine_similarity(
                            f1.unsqueeze(0), f2.unsqueeze(0)
                        ).item()
                        frame_cos_sims.append(cos)

                    mean_frame_cos = sum(frame_cos_sims) / len(frame_cos_sims)
                    std_frame_cos = (sum((c - mean_frame_cos) ** 2 for c in frame_cos_sims) / len(frame_cos_sims)) ** 0.5
                    logger.info(
                        f"  [Trajectory Anchor] η_temporal 帧间一致性: "
                        f"mean_cos={mean_frame_cos:.4f}, std={std_frame_cos:.4f}, "
                        f"range=[{min(frame_cos_sims):.4f}, {max(frame_cos_sims):.4f}] "
                        f"(阈值={cfg.anchor_quality_threshold})"
                    )
                    if mean_frame_cos >= cfg.anchor_quality_threshold:
                        # ── 门控通过: 保留轨迹锚定 + eta_temporal ──
                        logger.info(
                            f"  [Trajectory Anchor] ✅ 门控通过 "
                            f"(mean_cos={mean_frame_cos:.4f} >= {cfg.anchor_quality_threshold})，"
                            f"启用轨迹锚定 + {'SVD blend' if (cfg.use_blend and cfg.use_svd) else 'standard blend' if cfg.use_blend else 'no blend'}"
                        )
                    else:
                        # ── 门控拒绝: 跳过轨迹锚定 ──
                        ref_trajectory = None

                        if cfg.use_blend and cfg.use_svd:
                            # SVD+Blend 已启用: 保留 eta_temporal 供 SVD blend 使用，
                            # 只禁用轨迹锚定。生成仍用 SVD-blended 噪声（非纯随机）。
                            logger.warning(
                                f"  [Trajectory Anchor] ⚠️ η_temporal 帧间一致性过低 "
                                f"(mean_cos={mean_frame_cos:.4f} < {cfg.anchor_quality_threshold})，"
                                f"跳过轨迹锚定，但保留 SVD blend (α={cfg.alpha})"
                            )
                            # 重置 generator 以确保 SVD blend 中 η_random 的随机状态一致
                            generator = torch.Generator(device=self.device).manual_seed(seed)
                            torch.manual_seed(seed)
                            logger.info(
                                f"  [Trajectory Anchor] Generator 已重置 (seed={seed})，"
                                f"生成将使用 SVD-blended 噪声（保留运动先验，无轨迹约束）"
                            )
                        else:
                            # 无 SVD: 完全回退到纯 baseline（原有逻辑）
                            logger.warning(
                                f"  [Trajectory Anchor] ⚠️ η_temporal 帧间一致性过低 "
                                f"(mean_cos={mean_frame_cos:.4f} < {cfg.anchor_quality_threshold})，"
                                f"参考视频运动混乱，跳过轨迹锚定，完全回退到 baseline"
                            )
                            eta_temporal = None
                            generator = torch.Generator(device=self.device).manual_seed(seed)
                            torch.manual_seed(seed)
                            logger.info(
                                f"  [Trajectory Anchor] Generator 已重置 (seed={seed})，"
                                f"生成将使用与 baseline 完全相同的纯随机噪声"
                            )
                else:
                    logger.warning(
                        f"  [Trajectory Anchor] η_temporal 帧数不足 ({num_frames_gate})，跳过门控检查"
                    )

            if ref_trajectory is not None:
                logger.info(
                    f"  [Trajectory Anchor] Cached {len(ref_trajectory)} trajectory points, "
                    f"t range=[{min(ref_trajectory.keys()):.3f}, {max(ref_trajectory.keys()):.3f}]"
                )

        # ── Step 4: 生成循环 ──
        num_iters = cfg.i_max if cfg.use_iter else 1
        current_prompt = caption
        prev_video = None
        results = []

        # ── 诊断: 噪声决策状态总结 ──
        _diag_anchor = ref_trajectory is not None
        _diag_svd_blend = cfg.use_blend and cfg.use_svd
        _diag_eta_available = eta_temporal is not None
        logger.info(
            f"  [Noise Decision Summary] "
            f"trajectory_anchor={'ACTIVE' if _diag_anchor else 'OFF'}, "
            f"svd_blend={'ENABLED' if _diag_svd_blend else 'DISABLED'}, "
            f"eta_temporal={'AVAILABLE (shape={eta_temporal.shape})' if _diag_eta_available else 'NONE (will use pure random)'}, "
            f"use_blend={cfg.use_blend}"
        )
        if _diag_svd_blend and _diag_eta_available:
            logger.info(
                f"  [Noise Decision] → 将使用 SVD-blended 噪声 (α={cfg.alpha})"
            )
        elif _diag_eta_available and cfg.use_blend:
            logger.info(
                f"  [Noise Decision] → 将使用 standard blend 噪声 (α={cfg.alpha})"
            )
        elif not _diag_eta_available:
            logger.info(
                f"  [Noise Decision] → eta_temporal=None，diffusers 将生成纯随机噪声"
            )

        for i in range(1, num_iters + 1):
            logger.info(f"  iter {i}/{num_iters}: {current_prompt[:60]}...")

            # 获取噪声
            latents = self._get_latents(eta_temporal, generator)

            # 生成视频（VDA / 旧轨迹锚定 / 标准）
            # 两种 L3 方案:
            #   --velocity_anchor  (VDA): 速度方向引导, 不要求起点对齐, 与 L2 正交互补
            #   --trajectory_anchor (旧): position lerp, 纯 L3 有增益, 但与 L2(SVD blend) 组合会崩塌
            #     根因: z_T(SVD-blended) 与 ref_traj 起点几乎正交(cos~0.06),
            #     position lerp 把 z_gen 拉向与 L2 起始偏置冲突的方向
            # 两者互斥: velocity_anchor 优先级高于 trajectory_anchor
            if cfg.velocity_anchor and ref_trajectory is not None:
                gen_video = self._generate_with_vda(
                    current_prompt, latents, generator,
                    ref_trajectory=ref_trajectory,
                    eta_temporal=eta_temporal,
                    negative_prompt=neg_prompt,
                )
            elif cfg.trajectory_anchor and ref_trajectory is not None:
                gen_video = self._generate_with_anchor(
                    current_prompt, latents, generator,
                    ref_trajectory=ref_trajectory,
                    negative_prompt=neg_prompt,
                )
            else:
                gen_video = self._generate(current_prompt, latents, generator,
                                           negative_prompt=neg_prompt)
            video_path_i = str(out / f"iter_{i:02d}.mp4")
            save_video_tensor(gen_video, video_path_i, fps=cfg.fps)

            results.append({
                "iteration": i,
                "prompt": current_prompt,
                "video_path": video_path_i,
            })

            # VLM 迭代优化 (如果启用且不是最后一轮)
            if cfg.use_iter and i < num_iters:
                current_prompt = self._vlm_refine(
                    ref_video, gen_video, prev_video, current_prompt, i
                )

            prev_video = gen_video

        # ── Step 5: 输出最终结果 ──
        final_path = str(out / f"{sample_id}.mp4")
        shutil.copy2(results[-1]["video_path"], final_path)

        elapsed = time.time() - t0
        metadata = {
            "sample_id": sample_id,
            "experiment": cfg.experiment_name(),
            "flags": flags,
            "initial_caption": caption,
            "final_prompt": current_prompt,
            "iterations": num_iters,
            "time_seconds": elapsed,
            "output": final_path,
            "all_iterations": results,
        }
        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"[P-Flow] Done in {elapsed:.1f}s → {final_path}")
        # ── 样本完成总结 (便于后续与指标关联分析) ──
        logger.info(f"  [SAMPLE SUMMARY] sample_id={sample_id}")
        logger.info(f"  [SAMPLE SUMMARY] full_caption={caption}")
        logger.info(f"  [SAMPLE SUMMARY] caption_length={len(caption)} chars, word_count={len(caption.split())}")
        logger.info(f"  [SAMPLE SUMMARY] elapsed={elapsed:.1f}s, output={final_path}")
        return metadata

    # ─────────────────────────────────────────────────────────────
    # 内部方法：各改动点的实现
    # ─────────────────────────────────────────────────────────────

    def _compute_noise_prior(
        self, ref_video: torch.Tensor, prompt: str
    ) -> tuple:
        """
        改动点: Inversion + SVD → η_temporal (V2)

        流程: V_ref → VAE encode → Flow Inversion → SVD V2 → η_temporal

        V2 改进:
            - SVD 滤波后自动 renormalize 到 N(0,1)
            - 支持高频段分离 (避免与 v7e 文本运动描述冲突)
            - 自适应模式根据 motion_strength 动态选择策略

        Returns:
            (eta_temporal, eta_inv_raw, z0, prompt_embeds):
            SVD滤波后的噪声, 原始反演噪声(保留接口), VAE编码latent, prompt embedding
        """
        logger.info("  [Inversion] encoding reference → latent...")
        ref_norm = normalize_video(ref_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_norm, self.device)

        # Flow Matching Inversion
        prompt_embeds = self._encode_prompt(prompt)
        inverter = FlowMatchingInverter(
            pipe=self.pipe,
            num_inversion_steps=self.config.inversion_steps,
            guidance_scale=1.0,
            device=self.device,
        )

        if self.config.use_midpoint:
            logger.info("  [Inversion] midpoint (2nd-order)...")
            eta_inv = inverter.invert_midpoint(
                ref_latents, prompt_embeds, prompt_embeds
            )
        else:
            logger.info("  [Inversion] euler (1st-order)...")
            eta_inv = inverter.invert(
                ref_latents, prompt_embeds, prompt_embeds
            )

        # ── 诊断: Inversion 质量 ──
        eta_rand_ref = torch.randn_like(eta_inv)
        inv_std = eta_inv.std().item()
        inv_mean = eta_inv.mean().item()
        inv_min = eta_inv.min().item()
        inv_max = eta_inv.max().item()
        # 与纯随机噪声的余弦相似度 (应接近 0 如果 inversion 有意义)
        cos_sim_random = torch.nn.functional.cosine_similarity(
            eta_inv.flatten().unsqueeze(0),
            eta_rand_ref.flatten().unsqueeze(0)
        ).item()
        # 与原始 latent 的余弦相似度 (检查 inversion 是否"走远了")
        cos_sim_latent = torch.nn.functional.cosine_similarity(
            eta_inv.flatten().unsqueeze(0),
            ref_latents.flatten().unsqueeze(0)
        ).item()
        logger.info(
            f"  [Inversion Quality] η_inv: std={inv_std:.4f}, mean={inv_mean:.4f}, "
            f"range=[{inv_min:.3f}, {inv_max:.3f}], "
            f"cos_sim(η_inv, random)={cos_sim_random:.4f}, "
            f"cos_sim(η_inv, z0)={cos_sim_latent:.4f}"
        )

        # 保留原始反演噪声
        eta_inv_raw = eta_inv

        # SVD Filtering V2
        if self.config.use_svd:
            svd_config = SVDFilterConfig(
                rho_s=self.config.rho_s,
                rho_m=self.config.rho_m,
                mode=self.config.svd_mode,
                low_freq_ratio=self.config.svd_low_freq_ratio,
                knee_auto=self.config.svd_knee_auto,
                motion_strength_threshold=self.config.svd_motion_threshold,
                use_fast_svd=self.config.use_fast_svd,
            )
            svd_filter = SVDFilter(config=svd_config)

            logger.info(
                f"  [SVD-V2] mode={self.config.svd_mode}, "
                f"ρ_s={self.config.rho_s}, ρ_m={self.config.rho_m}"
            )

            eta_temporal = svd_filter.filter(eta_inv)

            # 可选: 保存诊断信息
            if self.config.svd_diagnostics:
                diag = compute_svd_diagnostics(eta_inv, config=svd_config)
                logger.info(
                    f"  [SVD-V2 Diagnostics] "
                    f"motion_strength={diag['motion_strength']:.4f}, "
                    f"k_m={diag['k_m']}, k_low={diag['k_low']}, "
                    f"raw_std={diag['temporal_raw_std']:.4f}, "
                    f"renormed_std={diag['temporal_renormed_std']:.4f}, "
                    f"energy: low={diag['energy_low_freq_pct']:.1f}% "
                    f"high={diag['energy_high_freq_pct']:.1f}% "
                    f"residual={diag['energy_residual_pct']:.1f}%"
                )
                self._svd_diagnostics = diag
        else:
            eta_temporal = eta_inv

        logger.info(
            f"  η_temporal: mean={eta_temporal.mean():.4f}, std={eta_temporal.std():.4f}"
        )
        return eta_temporal, eta_inv_raw, ref_latents, prompt_embeds

    def _resolve_negative_prompt(self, sample_id: int) -> str:
        """
        解析负面 prompt，优先级：
            1. negative_prompt_file 目录下的 {sample_id}.txt
            2. config.negative_prompt (全局自定义)
            3. 默认 NEGATIVE_PROMPT (硬编码)
        """
        cfg = self.config

        # 优先从按样本的负面 prompt 目录加载
        if cfg.negative_prompt_file:
            neg_file = Path(cfg.negative_prompt_file) / f"{sample_id}.txt"
            if neg_file.exists():
                content = neg_file.read_text(encoding="utf-8").strip()
                if content:
                    return content
                logger.warning(f"  [NegPrompt] 文件为空: {neg_file}, 使用 fallback")

        # 全局自定义负面 prompt
        if cfg.negative_prompt:
            return cfg.negative_prompt

        # 默认
        return NEGATIVE_PROMPT

    def _get_latents(
        self,
        eta_temporal: Optional[torch.Tensor],
        generator: torch.Generator,
    ) -> Optional[torch.Tensor]:
        """
        改动点: Noise Blending / Frequency Reshaping

        原始混合公式: η = √α · η_temporal + √(1-α) · η_random
        频域重塑公式: η = IFFT( |FFT(η_temporal)|^β · phase(FFT(η_random)) ) → renorm to N(0,1)

        方向 C (Spectrum-Aligned Noise Initialization):
            核心洞察: one-shot linear blend 的问题在于直接注入 η_temporal 的"内容",
            对某些样本 η_temporal 是有毒的 (如 sample 7/31 的 XCLIP 崩塌)。

            频域重塑不注入 η_temporal 的具体内容，而只转移其"时间频谱形状":
            - 参考视频快速运动 → 高频能量强 → 重塑后的噪声也高频能量强 → 引导模型生成快运动
            - 参考视频缓慢运动 → 低频能量强 → 重塑后的噪声也低频能量强 → 引导模型生成慢运动
            - 不引入 η_temporal 的空间结构或方向 → 不会产生冲突

            参考论文: FreqPrior (ICLR 2025) — 频域噪声塑形保持 Gaussian 分布

        β (freq_reshape_beta) 控制重塑强度:
            β=0: 不重塑，等价于纯随机噪声
            β=1: 完全匹配 η_temporal 的频谱形状
            推荐: β=0.5~1.0

        为什么比 linear blend 更安全:
            1. 不注入 η_temporal 的具体空间/方向内容 → 无毒性风险
            2. 保持 N(0,1) 分布 → 不违反扩散模型假设
            3. 只传递"运动节奏" → 比传递"运动方向"更通用
        """
        if eta_temporal is None or not self.config.use_blend:
            logger.info(
                f"  [_get_latents] 返回 None → diffusers 纯随机噪声 "
                f"(eta_temporal={'None' if eta_temporal is None else 'EXISTS'}, "
                f"use_blend={self.config.use_blend}, "
                f"velocity_anchor={self.config.velocity_anchor})"
            )
            if self.config.velocity_anchor and eta_temporal is not None:
                logger.info(
                    f"  [_get_latents] VDA模式: 不使用SVD噪声混合, "
                    f"但eta_temporal仍可用于质量门控 "
                    f"(shape={list(eta_temporal.shape)}, "
                    f"std={eta_temporal.std().item():.4f}, "
                    f"mean={eta_temporal.mean().item():.4f})"
                )
            return None  # 让 diffusers 自己采样随机噪声

        eta_random = torch.randn(
            eta_temporal.shape,
            dtype=eta_temporal.dtype,
            device=eta_temporal.device,
            generator=generator,
        )

        # ── 方向 C: 频域噪声重塑作为 η_random 预处理 ──
        # 如果启用 freq_reshape，先对 η_random 做频谱对齐，然后继续走 alpha 混合
        if self.config.freq_reshape:
            eta_random = self._freq_reshape_noise(eta_temporal, eta_random)
            logger.info(
                f"  [FreqReshape+Blend] η_random 已频域重塑 (β={self.config.freq_reshape_beta}), "
                f"继续 alpha={self.config.alpha} 混合"
            )

        # # ══════════════════════════════════════════════════════════════
        # # [DEPRECATED] 方向 C 旧版: 独立 freq_reshape (完全绕开 alpha blend)
        # # 实验结论: 独立模式无法利用 η_temporal 的内容信息，改为预处理叠加模式
        # # ══════════════════════════════════════════════════════════════
        # if self.config.freq_reshape:
        #     eta = self._freq_reshape_noise(eta_temporal, eta_random)
        #     return eta

        # ── 方向 F: Channel-Energy Gated Injection (CEGI) ──
        # 核心思想: 不同 channel 的时序能量不同，只在时序能量高的 channel 集中注入 temporal prior
        # 优势: 信号集中 → 有效 direction_shift 大幅提升; 未选中 channel 保持随机 → 安全缓冲
        if self.config.cegi:
            eta = self._cegi_blend(eta_temporal, eta_random)
            return eta

        # ── 方向 G: Multi-Scale Temporal Decomposition Injection (MSTDI) ──
        # 核心思想: 在空间低频层用大 α 注入 (控制全局运动), 高频层保持随机 (保证视觉质量)
        if self.config.mstdi:
            eta = self._mstdi_blend(eta_temporal, eta_random)
            return eta

        # ── 方向 H: Temporal Phase Injection (TPI) ──
        # 核心思想: 保留 η_random 的幅度谱, 只在时间维相位中注入参考视频的运动信息
        if self.config.tpi:
            eta = self._tpi_blend(eta_temporal, eta_random)
            return eta

        # ── 方向 I: Orthogonal Complement Suppression (OCS) ──
        # 核心思想: 在 η_random 中抑制与 η_temporal 主方向正交的分量, 间接增强 temporal 信号
        if self.config.ocs:
            eta = self._ocs_blend(eta_temporal, eta_random)
            return eta

        # ── 方向 E: Prompt-Orthogonal Decomposition Injection (PODI) ──
        # 核心思想: η_temporal 中包含与 prompt 对齐的分量 (有益) 和正交分量 (可能有害)
        # 只注入对齐分量，过滤掉正交/冲突的成分 → 安全提升 alpha
        # 参考论文: ODC (Orthogonal Drift Correction), Golden Noise (ICCV 2025), InitNO (CVPR 2024)
        if self.config.podi and self._prompt_embeds is not None:
            eta_temporal, alpha = self._podi_decompose(eta_temporal)
            # PODI 已确定 alpha，跳过后续 SGA 等 alpha 选择逻辑
            sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=self.device))
            sqrt_1_minus_alpha = torch.sqrt(torch.tensor(1.0 - alpha, device=self.device))
            eta = sqrt_alpha * eta_temporal + sqrt_1_minus_alpha * eta_random

            # ── 诊断: PODI Blend 效果 ──
            logger.info(
                f"  [PODI Blend] α={alpha:.4f} (√α={sqrt_alpha.item():.4f}), "
                f"η_temporal(aligned) std={eta_temporal.std():.4f}, "
                f"η_mixed std={eta.std():.4f}, mean={eta.mean():.4f}"
            )
            cos_m_t = torch.nn.functional.cosine_similarity(
                eta.flatten().unsqueeze(0),
                eta_temporal.flatten().unsqueeze(0)
            ).item()
            cos_m_r = torch.nn.functional.cosine_similarity(
                eta.flatten().unsqueeze(0),
                eta_random.flatten().unsqueeze(0)
            ).item()
            logger.info(
                f"  [PODI Blend Diag] cos(mixed, temporal_aligned)={cos_m_t:.4f}, "
                f"cos(mixed, random)={cos_m_r:.4f}"
            )
            return eta

        # ── Linear Blend 路径 ──
        alpha = self.config.alpha

        # # ══════════════════════════════════════════════════════════════
        # # [DEPRECATED] Quality-Gated Alpha (方案 B): per-sample adaptive alpha
        # # 实验结论: α=0.01 时 S7 XCLIP -34.5%，QGA 无法拯救有毒 η_temporal
        # # ══════════════════════════════════════════════════════════════
        # if self.config.quality_gated_alpha:
        #     quality = self._compute_direction_quality(eta_temporal)
        #     cfg = self.config
        #     effective_alpha = cfg.qga_base_alpha * (
        #         cfg.qga_low_mult + (cfg.qga_high_mult - cfg.qga_low_mult) * quality
        #     )
        #     logger.info(
        #         f"  [QGA] quality={quality:.4f} → effective_alpha={effective_alpha:.6f} "
        #         f"(base={cfg.qga_base_alpha}, range=[{cfg.qga_base_alpha * cfg.qga_low_mult:.6f}, "
        #         f"{cfg.qga_base_alpha * cfg.qga_high_mult:.6f}])"
        #     )
        #     alpha = effective_alpha

        # ── 方向 D: Std-Gated Adaptive Alpha (SGA) ──
        # 核心思想: η_temporal_std 反映信号"偏离度" (类比 SSNI ICML 2025 的 score norm)
        #   - std 高 → 信号偏离大 / 高频强 → 降低 alpha 保护生成质量
        #   - std 低 → 信号温和 / 低频主导 → 提高 alpha 充分利用时序先验
        # 公式: effective_alpha = base_alpha × (target_std / actual_std)
        #        然后 clamp 到 [alpha_min, alpha_max]
        if self.config.adaptive_alpha:
            actual_std = eta_temporal.std().item()
            cfg = self.config
            raw_alpha = cfg.alpha * (cfg.sga_target_std / max(actual_std, 1e-8))
            effective_alpha = max(cfg.sga_alpha_min, min(cfg.sga_alpha_max, raw_alpha))
            logger.info(
                f"  [SGA] η_temporal_std={actual_std:.4f}, "
                f"raw_alpha={raw_alpha:.6f} → effective_alpha={effective_alpha:.6f} "
                f"(base={cfg.alpha}, target_std={cfg.sga_target_std}, "
                f"range=[{cfg.sga_alpha_min}, {cfg.sga_alpha_max}])"
            )
            alpha = effective_alpha

        sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=self.device))
        sqrt_1_minus_alpha = torch.sqrt(torch.tensor(1.0 - alpha, device=self.device))

        eta = sqrt_alpha * eta_temporal + sqrt_1_minus_alpha * eta_random

        # ── 诊断: Blend 效果 ──
        # 1. 混合后分布
        logger.info(
            f"  [Blend] α={alpha:.4f} (√α={sqrt_alpha.item():.4f}), "
            f"η_temporal std={eta_temporal.std():.4f}, "
            f"η_mixed std={eta.std():.4f}, mean={eta.mean():.4f}"
        )
        # 2. η_temporal 与 η_random 的相关性 (应接近 0)
        cos_t_r = torch.nn.functional.cosine_similarity(
            eta_temporal.flatten().unsqueeze(0),
            eta_random.flatten().unsqueeze(0)
        ).item()
        # 3. η_mixed 与 η_random 的相关性 (α 小时应接近 1.0)
        cos_m_r = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_random.flatten().unsqueeze(0)
        ).item()
        # 4. η_mixed 与 η_temporal 的相关性 (α 小时应接近 √α ≈ 0.055)
        cos_m_t = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_temporal.flatten().unsqueeze(0)
        ).item()
        logger.info(
            f"  [Blend Diag] cos(temporal, random)={cos_t_r:.4f}, "
            f"cos(mixed, random)={cos_m_r:.4f}, "
            f"cos(mixed, temporal)={cos_m_t:.4f}"
        )
        # 5. 有效信号强度: mixed 中来自 temporal 的"方向偏移量"
        direction_shift = (eta - eta_random).norm().item() / eta_random.norm().item()
        logger.info(
            f"  [Blend Diag] direction_shift=‖η-η_rand‖/‖η_rand‖={direction_shift:.6f} "
            f"(越大越说明 temporal 有影响)"
        )

        return eta

    def _freq_reshape_noise(
        self,
        eta_temporal: torch.Tensor,
        eta_random: torch.Tensor,
    ) -> torch.Tensor:
        """
        方向 C 核心实现: Spectrum-Aligned Noise Initialization.

        算法步骤:
            1. 沿时间维度 (frame axis) 对 η_temporal 做 rFFT，取功率谱 |F_t(f)|²
            2. 沿时间维度对 η_random 做 rFFT，得到 F_r(f) = |F_r(f)| · e^{jφ_r(f)}
            3. 计算目标频谱幅度: A_target(f) = |F_t(f)|^β (β=freq_reshape_beta)
            4. 用平滑的频谱形状调制 η_random: F_out(f) = A_target(f) · e^{jφ_r(f)}
               但这会改变总能量，所以需要归一化
            5. IRFFT 回时域
            6. 全局 renormalization 到 N(0,1)

        关键设计:
            - 只沿 frame 维度做 FFT (不动空间维度) → 只传递时间节奏
            - 使用 η_random 的相位 → 保持随机性，不引入 η_temporal 的具体内容
            - β < 1 时做平滑的"部分重塑"，β=0 退化为纯随机
            - 最终 renorm 到 N(0,1) → 满足扩散模型假设

        Inspired by:
            - FreqPrior (ICLR 2025): frequency-domain noise shaping
            - FreeInit (ECCV 2024): low-freq preservation concept
        """
        cfg = self.config
        beta = cfg.freq_reshape_beta

        # eta_temporal shape: (B, C, F, H, W) 或 (1, C, F, H, W)
        # 沿 frame 维度 (dim=2) 做 FFT
        # 先转 float32 保证 FFT 精度
        orig_dtype = eta_random.dtype
        eta_t_f32 = eta_temporal.float()
        eta_r_f32 = eta_random.float()

        # ── Step 1: 计算 η_temporal 的时间频谱幅度 ──
        # rFFT along frame dimension (dim=2 for 5D: B,C,F,H,W)
        frame_dim = 2
        F_temporal = torch.fft.rfft(eta_t_f32, dim=frame_dim)
        # 功率谱: 每个频率 bin 的平均幅度 (在 B,C,H,W 上平均，得到 per-frequency 的形状)
        amp_temporal = F_temporal.abs()  # (B, C, F//2+1, H, W)

        # 对空间维度取均值得到"全局频谱形状" → (1, 1, F//2+1, 1, 1)
        # 这避免了传递空间结构，只保留全局时间节奏
        spectrum_shape = amp_temporal.mean(dim=(0, 1, 3, 4), keepdim=True)  # (1,1,nfreq,1,1)

        # 归一化频谱形状 (使得 mean=1，纯形状信息)
        spectrum_shape = spectrum_shape / (spectrum_shape.mean() + 1e-8)

        # 应用 β 控制重塑强度: shape^β, β=0→全1(不重塑), β=1→完全匹配
        # 使用 log 域插值: exp(β * log(shape)) = shape^β
        reshape_filter = spectrum_shape.pow(beta)  # (1,1,nfreq,1,1)

        # ── Step 2: 对 η_random 做频域调制 ──
        F_random = torch.fft.rfft(eta_r_f32, dim=frame_dim)

        # 保留 η_random 的相位，用 reshape_filter 调制幅度
        F_shaped = F_random * reshape_filter

        # ── Step 3: IRFFT 回时域 ──
        num_frames = eta_r_f32.shape[frame_dim]
        eta_shaped = torch.fft.irfft(F_shaped, n=num_frames, dim=frame_dim)

        # ── Step 4: Renormalization to N(0,1) ──
        eta_mean = eta_shaped.mean()
        eta_std = eta_shaped.std()
        if eta_std < 1e-8:
            logger.warning("  [FreqReshape] near-zero std after IRFFT, falling back to random")
            return eta_random

        eta_out = (eta_shaped - eta_mean) / eta_std

        # 恢复原始 dtype
        eta_out = eta_out.to(orig_dtype)

        # ── 诊断日志 ──
        # 频谱形状分析
        spec_np = spectrum_shape.squeeze().cpu()
        logger.info(
            f"  [FreqReshape] β={beta:.2f}, spectrum_shape: "
            f"DC={spec_np[0]:.3f}, mid={spec_np[len(spec_np)//2]:.3f}, "
            f"high={spec_np[-1]:.3f}, ratio(DC/high)={spec_np[0]/(spec_np[-1]+1e-8):.2f}"
        )
        logger.info(
            f"  [FreqReshape] η_shaped: mean={eta_out.mean():.4f}, std={eta_out.std():.4f}"
        )

        # 与 η_temporal 和 η_random 的相关性
        cos_out_t = torch.nn.functional.cosine_similarity(
            eta_out.flatten().unsqueeze(0).float(),
            eta_temporal.flatten().unsqueeze(0).float()
        ).item()
        cos_out_r = torch.nn.functional.cosine_similarity(
            eta_out.flatten().unsqueeze(0).float(),
            eta_random.flatten().unsqueeze(0).float()
        ).item()
        logger.info(
            f"  [FreqReshape] cos(shaped, temporal)={cos_out_t:.4f}, "
            f"cos(shaped, random)={cos_out_r:.4f}"
        )

        # 验证频谱确实被重塑了: 比较 η_out 和 η_random 的频谱
        F_out = torch.fft.rfft(eta_out.float(), dim=frame_dim)
        amp_out = F_out.abs().mean(dim=(0, 1, 3, 4))
        amp_rand = F_random.abs().mean(dim=(0, 1, 3, 4))
        # 计算频谱相关性: reshaped vs temporal 应 > reshaped vs random
        amp_t_global = amp_temporal.mean(dim=(0, 1, 3, 4))
        corr_out_t = torch.nn.functional.cosine_similarity(
            amp_out.unsqueeze(0), amp_t_global.unsqueeze(0)
        ).item()
        corr_rand_t = torch.nn.functional.cosine_similarity(
            amp_rand.unsqueeze(0), amp_t_global.unsqueeze(0)
        ).item()
        logger.info(
            f"  [FreqReshape] spectrum_corr: shaped_vs_temporal={corr_out_t:.4f}, "
            f"random_vs_temporal={corr_rand_t:.4f} "
            f"(shaped should be higher)"
        )

        return eta_out

    def _cegi_blend(
        self,
        eta_temporal: torch.Tensor,
        eta_random: torch.Tensor,
    ) -> torch.Tensor:
        """
        方向 F 核心实现: Channel-Energy Gated Injection (CEGI).

        核心思想:
            η_temporal 的 16 个 channel 时序能量(temporal variance)差异显著。
            时序能量高的 channel 集中了运动信息，能量低的 channel 主要是空间纹理/噪声。
            CEGI 只在 top-k 高时序能量 channel 集中注入 temporal prior (用更大的 α),
            其余 channel 保持纯随机 (或极小 α)。

        优势:
            - 信号集中: 不在全 16 channel 稀释 α=0.004，而是在 4 channel 上用 α=0.02
              → 有效 direction_shift 提升 ~5x
            - 数据驱动: 不依赖任何外部映射 (vs PODI 的 prompt→channel)
            - 安全缓冲: 12 个未注入 channel 保持随机，给模型留足空间
            - 实现简洁: per-channel 条件 blend

        参数:
            eta_temporal: SVD 滤波后的时序噪声 (B, C, F, H, W)
            eta_random: 标准高斯随机噪声 (B, C, F, H, W)

        Returns:
            eta: 混合后的噪声 (B, C, F, H, W)
        """
        cfg = self.config
        B, C, F, H, W = eta_temporal.shape

        # ── Step 1: 计算每个 channel 的时序能量 ──
        # temporal_var[c] = Var(η_temporal[:, c, :, :, :], dim=frame).mean()
        # 即: 在 frame 维度上的方差，再空间平均
        eta_f32 = eta_temporal.float()

        # (B, C, F, H, W) → 在 F 维度上算方差 → (B, C, H, W) → 空间平均 → (B, C)
        temporal_var = eta_f32.var(dim=2).mean(dim=(0, 2, 3))  # (C,)

        # ── Step 2: 排序选择 top-k channels ──
        top_k = min(cfg.cegi_top_k, C)
        sorted_indices = torch.argsort(temporal_var, descending=True)
        selected = sorted_indices[:top_k]
        not_selected = sorted_indices[top_k:]

        # 诊断日志
        logger.info(
            f"  [CEGI] Channel temporal variance (sorted desc): "
            f"{temporal_var[sorted_indices].tolist()[:8]}"
        )
        logger.info(
            f"  [CEGI] Selected top-{top_k} channels: {selected.tolist()}, "
            f"var range: [{temporal_var[selected[-1]]:.4f}, {temporal_var[selected[0]]:.4f}]"
        )
        logger.info(
            f"  [CEGI] Not selected channels var range: "
            f"[{temporal_var[not_selected[-1]]:.4f}, {temporal_var[not_selected[0]]:.4f}]"
            if len(not_selected) > 0 else "  [CEGI] All channels selected"
        )

        # ── Step 3: 选择性注入 ──
        eta = eta_random.clone()

        alpha_inject = cfg.cegi_alpha
        alpha_residual = cfg.cegi_residual_alpha

        sqrt_a_inject = (alpha_inject ** 0.5)
        sqrt_1ma_inject = ((1.0 - alpha_inject) ** 0.5)
        sqrt_a_residual = (alpha_residual ** 0.5)
        sqrt_1ma_residual = ((1.0 - alpha_residual) ** 0.5)

        # 注入选中的 channel
        for c in selected:
            c_idx = c.item()
            eta[:, c_idx, :, :, :] = (
                sqrt_a_inject * eta_temporal[:, c_idx, :, :, :]
                + sqrt_1ma_inject * eta_random[:, c_idx, :, :, :]
            )

        # 未选中的 channel: 保持随机或微弱注入
        if alpha_residual > 0:
            for c in not_selected:
                c_idx = c.item()
                eta[:, c_idx, :, :, :] = (
                    sqrt_a_residual * eta_temporal[:, c_idx, :, :, :]
                    + sqrt_1ma_residual * eta_random[:, c_idx, :, :, :]
                )
        # else: 未选中 channel 已经是 eta_random (clone 时保留)

        # ── Step 4: Per-channel renorm (保证每个 channel 是 N(0,1)) ──
        for c_idx in range(C):
            ch = eta[:, c_idx, :, :, :]
            ch_mean = ch.mean()
            ch_std = ch.std()
            if ch_std > 1e-8:
                eta[:, c_idx, :, :, :] = (ch - ch_mean) / ch_std

        # ── 诊断: 整体统计 ──
        direction_shift = (eta - eta_random).norm().item() / eta_random.norm().item()
        cos_m_r = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_random.flatten().unsqueeze(0)
        ).item()
        cos_m_t = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_temporal.flatten().unsqueeze(0)
        ).item()

        logger.info(
            f"  [CEGI Blend] α_inject={alpha_inject:.4f} (top-{top_k}), "
            f"α_residual={alpha_residual:.4f} (rest-{C - top_k})"
        )
        logger.info(
            f"  [CEGI Blend Diag] direction_shift={direction_shift:.6f}, "
            f"cos(mixed, random)={cos_m_r:.4f}, "
            f"cos(mixed, temporal)={cos_m_t:.4f}"
        )
        logger.info(
            f"  [CEGI Blend] η_out: mean={eta.mean():.4f}, std={eta.std():.4f}"
        )

        return eta

    def _mstdi_blend(
        self,
        eta_temporal: torch.Tensor,
        eta_random: torch.Tensor,
    ) -> torch.Tensor:
        """
        方向 G 核心实现: Multi-Scale Temporal Decomposition Injection (MSTDI).

        核心思想:
            将噪声在空间维度做多尺度分解 (Gaussian Pyramid)，
            在粗尺度 (低频) 用大 α 注入 temporal prior → 控制全局运动方向,
            在细尺度 (高频) 用小/零 α → 保持随机性保证视觉质量。

        理论支撑:
            - FreeInit (ECCV 2024): 低频分量决定全局运动
            - Video-MSG (2025): 多尺度引导策略
            - 扩散模型 coarse-to-fine 特性: 低频结构有"杠杆效应"

        算法:
            1. 对 η_temporal 和 η_random 分别做 spatial avg pooling 到多个尺度
            2. 在每个尺度上用递减的 α 做 linear blend
            3. 通过 Laplacian 差分重建全分辨率噪声
            4. 最终 renorm 到 N(0,1)
        """
        cfg = self.config
        B, C, F, H, W = eta_temporal.shape
        num_levels = cfg.mstdi_levels

        # 计算每层的 alpha (指数衰减)
        alphas = []
        a = cfg.mstdi_alpha_base
        for i in range(num_levels):
            alphas.append(a)
            a *= cfg.mstdi_alpha_decay

        logger.info(
            f"  [MSTDI] levels={num_levels}, alpha_schedule={[f'{a:.5f}' for a in alphas]}"
        )

        # ── Step 1: 构建多尺度 blended tensors ──
        # 使用 avg_pool3d 对 spatial 维度下采样 (保持 frame 维度不变)
        import torch.nn.functional as F_nn

        eta_t_f32 = eta_temporal.float()
        eta_r_f32 = eta_random.float()

        # 每层的 blend 结果 (从粗到细)
        blended_levels = []
        for level_idx in range(num_levels):
            scale_factor = 2 ** (num_levels - 1 - level_idx)  # L0=最粗(4x缩), L2=原始(1x)

            if scale_factor > 1:
                # 下采样 spatial 维度: (B, C, F, H, W) → (B, C, F, H/s, W/s)
                # 先 reshape 为 (B*C, F, H, W) 再 pool，再 reshape 回来
                eta_t_down = F_nn.avg_pool3d(
                    eta_t_f32.reshape(B * C, F, H, W).unsqueeze(1),
                    kernel_size=(1, scale_factor, scale_factor),
                    stride=(1, scale_factor, scale_factor),
                ).squeeze(1).reshape(B, C, F, H // scale_factor, W // scale_factor)

                eta_r_down = F_nn.avg_pool3d(
                    eta_r_f32.reshape(B * C, F, H, W).unsqueeze(1),
                    kernel_size=(1, scale_factor, scale_factor),
                    stride=(1, scale_factor, scale_factor),
                ).squeeze(1).reshape(B, C, F, H // scale_factor, W // scale_factor)
            else:
                eta_t_down = eta_t_f32
                eta_r_down = eta_r_f32

            # Blend at this level
            alpha_level = alphas[level_idx]
            sqrt_a = alpha_level ** 0.5
            sqrt_1ma = (1.0 - alpha_level) ** 0.5
            blended = sqrt_a * eta_t_down + sqrt_1ma * eta_r_down

            blended_levels.append((blended, scale_factor, alpha_level))

            logger.info(
                f"  [MSTDI] Level {level_idx}: scale=1/{scale_factor}, "
                f"shape={list(blended.shape)}, α={alpha_level:.5f}"
            )

        # ── Step 2: 从粗到细重建 (Laplacian-style) ──
        # 策略: 从最粗层开始，逐层上采样并加上细层的高频残差
        # 最粗层直接用 blended，中间层取 (blended - upsample(coarser)) 作为细节
        # 最终 = coarsest_upsampled + detail_1 + detail_2 + ...

        # 从最粗层开始
        result = blended_levels[0][0]  # 最粗层的 blend

        for level_idx in range(1, num_levels):
            blended_curr, scale_curr, _ = blended_levels[level_idx]
            _, scale_prev, _ = blended_levels[level_idx - 1]

            # 上采样前一层结果到当前层分辨率
            target_h = H // scale_curr
            target_w = W // scale_curr

            # result 当前分辨率是上一层的, 需要上采样
            result_up = F_nn.interpolate(
                result.reshape(B * C, F, result.shape[3], result.shape[4]).unsqueeze(1),
                size=(F, target_h, target_w),
                mode='trilinear',
                align_corners=False,
            ).squeeze(1).reshape(B, C, F, target_h, target_w)

            # 当前层的高频细节 = blended_curr - upsample(上一层 blended)
            prev_blended = blended_levels[level_idx - 1][0]
            prev_up = F_nn.interpolate(
                prev_blended.reshape(B * C, F, prev_blended.shape[3], prev_blended.shape[4]).unsqueeze(1),
                size=(F, target_h, target_w),
                mode='trilinear',
                align_corners=False,
            ).squeeze(1).reshape(B, C, F, target_h, target_w)

            high_freq_detail = blended_curr - prev_up

            # 累积: 粗层上采样 + 当前层高频细节
            result = result_up + high_freq_detail

        # ── Step 3: 全局 renorm 到 N(0,1) ──
        eta = result
        eta = (eta - eta.mean()) / (eta.std() + 1e-8)
        eta = eta.to(eta_temporal.dtype)

        # ── 诊断 ──
        direction_shift = (eta - eta_random).norm().item() / eta_random.norm().item()
        cos_m_r = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_random.flatten().unsqueeze(0).float()
        ).item()
        cos_m_t = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_temporal.flatten().unsqueeze(0).float()
        ).item()

        logger.info(
            f"  [MSTDI Blend Diag] direction_shift={direction_shift:.6f}, "
            f"cos(mixed, random)={cos_m_r:.4f}, cos(mixed, temporal)={cos_m_t:.4f}"
        )
        logger.info(f"  [MSTDI Blend] η_out: mean={eta.mean():.4f}, std={eta.std():.4f}")

        return eta

    def _tpi_blend(
        self,
        eta_temporal: torch.Tensor,
        eta_random: torch.Tensor,
    ) -> torch.Tensor:
        """
        方向 H 核心实现: Temporal Phase Injection (TPI).

        核心思想:
            在 3D 频域中保留 η_random 的幅度谱 (保证功率谱不变 → Gaussian property),
            只在时间维度的相位中注入 η_temporal 的运动结构。

        理论支撑:
            - 信号处理经典实验: 相位决定结构 (交换两张图的幅度和相位, 视觉内容跟随相位)
            - 方向 C 频域重塑只改了幅度 → 几乎无效
            - 相位携带的信息量远大于幅度

        算法:
            1. 对 η_temporal 和 η_random 做时间维度 rFFT
            2. 提取 η_temporal 的相位, η_random 的幅度和相位
            3. 对时间频率 > 0 的 bin 做相位插值: φ_out = (1-γ)φ_rand + γ·φ_ref
            4. 用 η_random 的幅度 + 混合相位重建
            5. IRFFT 回时域, renorm 到 N(0,1)
        """
        cfg = self.config
        gamma = cfg.tpi_gamma
        B, C, F, H, W = eta_temporal.shape

        eta_t_f32 = eta_temporal.float()
        eta_r_f32 = eta_random.float()

        # ── Step 1: 时间维度 FFT (dim=2, frame 维度) ──
        F_ref = torch.fft.rfft(eta_t_f32, dim=2)   # (B, C, F//2+1, H, W) complex
        F_rand = torch.fft.rfft(eta_r_f32, dim=2)  # (B, C, F//2+1, H, W) complex

        # ── Step 2: 分离幅度和相位 ──
        amp_rand = F_rand.abs()        # η_random 的幅度 (保留)
        phase_rand = F_rand.angle()    # η_random 的相位
        phase_ref = F_ref.angle()      # η_temporal 的相位 (参考)

        # ── Step 3: 选择性相位注入 (跳过 DC) ──
        freq_bins = F_ref.shape[2]  # F//2 + 1
        freq_min = cfg.tpi_freq_min
        freq_max = freq_bins if cfg.tpi_freq_max < 0 else min(cfg.tpi_freq_max, freq_bins)

        # 构造 phase_out: 默认等于 phase_rand, 在指定频率范围内做插值
        phase_out = phase_rand.clone()

        if freq_min < freq_max:
            # 相位插值: 使用圆周插值避免 wrapping 问题
            # 方法: 将相位差包裹到 [-π, π], 再做线性插值
            phase_diff = phase_ref[:, :, freq_min:freq_max, :, :] - phase_rand[:, :, freq_min:freq_max, :, :]
            # Wrap to [-π, π]
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            # Interpolate
            phase_out[:, :, freq_min:freq_max, :, :] = phase_rand[:, :, freq_min:freq_max, :, :] + gamma * phase_diff

        # ── Step 4: 重建频域信号 (保留 random 幅度 + 混合相位) ──
        F_out = amp_rand * torch.exp(1j * phase_out)

        # ── Step 5: IRFFT 回时域 ──
        eta_f32 = torch.fft.irfft(F_out, n=F, dim=2)  # (B, C, F, H, W)

        # ── Step 6: 全局 renorm 到 N(0,1) ──
        eta = (eta_f32 - eta_f32.mean()) / (eta_f32.std() + 1e-8)
        eta = eta.to(eta_temporal.dtype)

        # ── 诊断 ──
        direction_shift = (eta - eta_random).norm().item() / eta_random.norm().item()
        cos_m_r = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_random.flatten().unsqueeze(0).float()
        ).item()
        cos_m_t = torch.nn.functional.cosine_similarity(
            eta.flatten().unsqueeze(0),
            eta_temporal.flatten().unsqueeze(0).float()
        ).item()

        logger.info(
            f"  [TPI] γ={gamma:.2f}, freq_range=[{freq_min}, {freq_max}), "
            f"total_freq_bins={freq_bins}"
        )
        logger.info(
            f"  [TPI Blend Diag] direction_shift={direction_shift:.6f}, "
            f"cos(mixed, random)={cos_m_r:.4f}, cos(mixed, temporal)={cos_m_t:.4f}"
        )
        logger.info(f"  [TPI Blend] η_out: mean={eta.mean():.4f}, std={eta.std():.4f}")

        return eta

    def _ocs_blend(
        self,
        eta_temporal: torch.Tensor,
        eta_random: torch.Tensor,
    ) -> torch.Tensor:
        """
        方向 I 核心实现: Orthogonal Complement Suppression (OCS).

        核心思想:
            不注入 η_temporal 的内容 (避免毒性),
            而是对 η_random 做处理: 抑制与 η_temporal 主方向正交的分量。
            效果: η_random "偏向" temporal 主方向，间接增强 temporal 信号的影响力。

        与 PODI 的区别:
            - PODI: 从 η_temporal 中提取信号 → 信号太弱 (7%~11%)
            - OCS: 从 η_random 中抑制噪声 → 相对增强 temporal 方向
            - OCS 不直接注入 temporal content → 不引入"毒性"

        算法:
            1. 对 η_temporal (flatten per-frame) 做 SVD，取 top-k 主成分 V_k
            2. 将 η_random (per-frame) 分解: proj (在 V_k 上) + orth (正交补)
            3. 抑制 orth: η_out = proj + (1 - suppress_ratio) * orth
            4. Renorm 到 N(0,1)
        """
        cfg = self.config
        B, C, F, H, W = eta_temporal.shape
        top_k = cfg.ocs_top_k
        suppress = cfg.ocs_suppress_ratio

        eta_t_f32 = eta_temporal.float()
        eta_r_f32 = eta_random.float()

        # ── Step 1: 对 η_temporal 提取主成分方向 ──
        # Reshape: (B, C, F, H, W) → (F, C*H*W) — 每帧一个高维向量
        spatial_dim = C * H * W
        eta_t_flat = eta_t_f32.reshape(B, C, F, H * W).permute(0, 2, 1, 3).reshape(B * F, spatial_dim)
        # 取第一个 batch (B=1 通常)
        eta_t_mat = eta_t_flat[:F, :]  # (F, spatial_dim)

        # SVD: 只需要 top-k 右奇异向量
        # 由于 spatial_dim >> F, 用 eta_t_mat^T @ eta_t_mat 的特征向量更高效
        # 但直接用 torch.linalg.svd 的 partial 版本
        try:
            # 使用 randomized SVD 加速 (F 通常只有 21)
            U, S, Vh = torch.linalg.svd(eta_t_mat, full_matrices=False)  # Vh: (F, spatial_dim)
            V_k = Vh[:top_k, :]  # (k, spatial_dim) — top-k 主方向
        except Exception as e:
            logger.warning(f"  [OCS] SVD failed: {e}, falling back to normal blend")
            alpha = cfg.alpha
            sqrt_a = alpha ** 0.5
            sqrt_1ma = (1.0 - alpha) ** 0.5
            return (sqrt_a * eta_temporal + sqrt_1ma * eta_random)

        logger.info(
            f"  [OCS] SVD singular values (top-{top_k}): {S[:top_k].tolist()}, "
            f"energy ratio: {(S[:top_k]**2).sum() / (S**2).sum():.4f}"
        )

        # ── Step 2: 对 η_random 做投影分解 ──
        eta_r_flat = eta_r_f32.reshape(B, C, F, H * W).permute(0, 2, 1, 3).reshape(B * F, spatial_dim)
        eta_r_mat = eta_r_flat[:F, :]  # (F, spatial_dim)

        # 投影到 V_k 子空间: proj = (η_r @ V_k^T) @ V_k
        proj_coeffs = eta_r_mat @ V_k.T  # (F, k)
        proj = proj_coeffs @ V_k          # (F, spatial_dim)
        orth = eta_r_mat - proj           # (F, spatial_dim) — 正交补

        # ── Step 3: 抑制正交补 ──
        eta_out_flat = proj + (1.0 - suppress) * orth  # (F, spatial_dim)

        # ── Step 4: Reshape 回原始形状 + renorm ──
        # (F, C*H*W) → (1, C, F, H, W)
        eta_out = eta_out_flat.reshape(F, C, H, W).unsqueeze(0).permute(0, 1, 2, 3, 4)
        # 实际上需要: (F, C*H*W) → (F, C, H, W) → permute 为 (1, C, F, H, W)
        eta_out = eta_out_flat.reshape(F, C, H, W).permute(1, 0, 2, 3).unsqueeze(0)
        # 现在是 (1, C, F, H, W) ✓

        # Renorm
        eta_out = (eta_out - eta_out.mean()) / (eta_out.std() + 1e-8)
        eta_out = eta_out.to(eta_temporal.dtype)

        # ── 诊断 ──
        direction_shift = (eta_out - eta_random).norm().item() / eta_random.norm().item()
        cos_m_r = torch.nn.functional.cosine_similarity(
            eta_out.flatten().unsqueeze(0),
            eta_random.flatten().unsqueeze(0).float()
        ).item()
        cos_m_t = torch.nn.functional.cosine_similarity(
            eta_out.flatten().unsqueeze(0),
            eta_temporal.flatten().unsqueeze(0).float()
        ).item()

        logger.info(
            f"  [OCS] top_k={top_k}, suppress_ratio={suppress:.2f}"
        )
        logger.info(
            f"  [OCS Blend Diag] direction_shift={direction_shift:.6f}, "
            f"cos(mixed, random)={cos_m_r:.4f}, cos(mixed, temporal)={cos_m_t:.4f}"
        )
        logger.info(f"  [OCS Blend] η_out: mean={eta_out.mean():.4f}, std={eta_out.std():.4f}")

        return eta_out

    def _podi_decompose(
        self,
        eta_temporal: torch.Tensor,
    ) -> tuple:
        """
        方向 E 核心实现: Prompt-Orthogonal Decomposition Injection (PODI).

        核心思想:
            η_temporal 包含与 prompt 语义方向对齐的分量 (有益运动信息)
            和正交/冲突的分量 (可能导致 XCLIP 崩塌)。
            PODI 通过将 prompt embedding 投影到 latent 空间，
            分解 η_temporal 为 parallel + orthogonal，只注入 parallel 部分。

        算法步骤:
            1. 从缓存的 prompt_embeds 提取方向向量 d_text (降维到 latent dim)
            2. 将 η_temporal (flatten) 投影到 d_text 方向:
               η_parallel = (η_temporal · d_text / ||d_text||²) × d_text
            3. 计算 alignment score = ||η_parallel|| / ||η_temporal||
               - alignment 高 → η_temporal 大部分与 prompt 对齐 → 安全，用大 alpha
               - alignment 低 → η_temporal 与 prompt 正交/冲突 → 只注入 parallel 部分
            4. 对 η_parallel 做 renorm 到 N(0,1)
            5. 确定 effective_alpha (基于 alignment score 动态缩放)

        理论基础:
            - ODC (OpenReview 2024): text embedding 在去噪中沿正交方向漂移导致语义不对齐
            - Golden Noise (ICCV 2025): 噪声的"好坏"取决于其与 text 的语义耦合度
            - InitNO (CVPR 2024): cross-attention 对齐度可度量噪声质量

        与 SGA 的区别:
            - SGA 只看 std (标量)，不知道 η_temporal 的"方向"对不对
            - PODI 看向量对齐度，直接解决 S7/S31 的方向冲突问题
            - 两者可正交叠加: PODI 过滤方向 + SGA 调节强度

        Returns:
            (eta_aligned, alpha): 对齐后的 η_temporal 分量, 确定的 alpha 值
        """
        cfg = self.config
        prompt_embeds = self._prompt_embeds  # shape: (1, seq_len, hidden_dim)

        # ── Step 1: 从 prompt embedding 提取方向向量 ──
        # prompt_embeds shape: (batch=1, seq_len, hidden_dim)
        # 需要将其降维为一个与 eta_temporal flatten 后同维度的方向向量
        # 策略: 先池化到 (1, hidden_dim)，再线性插值/repeat 到 latent 维度

        if cfg.podi_proj_mode == "mean_pool":
            # 对 seq_len 维度均值池化 → (1, hidden_dim)
            d_text_raw = prompt_embeds.mean(dim=1).squeeze(0)  # (hidden_dim,)
        elif cfg.podi_proj_mode == "last_token":
            # 取最后一个 token → (hidden_dim,)
            d_text_raw = prompt_embeds[0, -1, :]  # (hidden_dim,)
        elif cfg.podi_proj_mode == "weighted":
            # attention-weighted: 用 norm 作为 weight (高 norm 的 token 更重要)
            norms = prompt_embeds[0].norm(dim=-1, keepdim=True)  # (seq_len, 1)
            weights = norms / (norms.sum() + 1e-8)
            d_text_raw = (prompt_embeds[0] * weights).sum(dim=0)  # (hidden_dim,)
        else:
            d_text_raw = prompt_embeds.mean(dim=1).squeeze(0)

        # ── Step 2: 将 d_text 投影到 η_temporal 的 latent 空间 ──
        # η_temporal shape: (B, C, F, H, W) — 通常 (1, 16, 21, 60, 104)
        # d_text_raw shape: (hidden_dim,) — 通常 T5 是 4096 维
        # 策略: 将 d_text 重复/投影到 latent 的 channel 维度方向
        #
        # 关键洞察: 我们不需要逐元素对齐，而是利用 d_text 作为"语义方向"
        # 在 channel 维度上做投影 — channel 是 latent 的特征通道,
        # 每个 channel 编码了不同的语义信息

        eta_shape = eta_temporal.shape  # (B, C, F, H, W)
        B, C, F, H, W = eta_shape

        # 将 d_text 投影到 C 维空间 (通道方向)
        # 方法: 对 d_text 做分段平均池化到 C 维
        hidden_dim = d_text_raw.shape[0]
        d_text_f32 = d_text_raw.float()

        # 分段平均池化: hidden_dim → C
        # 将 hidden_dim 分成 C 段，每段取平均
        chunk_size = hidden_dim // C
        if chunk_size > 0:
            d_channel = d_text_f32[:chunk_size * C].reshape(C, chunk_size).mean(dim=1)  # (C,)
        else:
            # 如果 hidden_dim < C (极少见)，直接插值
            d_channel = torch.nn.functional.interpolate(
                d_text_f32.unsqueeze(0).unsqueeze(0),
                size=C, mode='linear', align_corners=False
            ).squeeze()  # (C,)

        # 归一化方向向量
        d_norm = d_channel.norm()
        if d_norm < 1e-8:
            logger.warning("  [PODI] d_text norm ≈ 0, falling back to normal blend")
            return eta_temporal, cfg.alpha

        d_unit = d_channel / d_norm  # (C,) 单位方向向量

        # ── Step 3: 在 channel 维度上做正交分解 ──
        # η_temporal: (B, C, F, H, W) — 将 channel 视为"特征方向"
        # 对每个 (B, F, H, W) 位置，投影其 C 维向量到 d_unit 方向

        eta_f32 = eta_temporal.float()

        # d_unit 扩展为 (1, C, 1, 1, 1) 方便广播
        d_expanded = d_unit.reshape(1, C, 1, 1, 1)

        # 投影: proj_scalar = sum(eta * d_unit, dim=C) → (B, 1, F, H, W)
        proj_scalar = (eta_f32 * d_expanded).sum(dim=1, keepdim=True)  # (B, 1, F, H, W)

        # η_parallel = proj_scalar * d_unit → (B, C, F, H, W)
        eta_parallel = proj_scalar * d_expanded  # (B, C, F, H, W)

        # η_orthogonal = η_temporal - η_parallel
        eta_orthogonal = eta_f32 - eta_parallel

        # ── Step 4: 计算 alignment score ──
        # alignment = ||η_parallel||₂ / ||η_temporal||₂
        norm_parallel = eta_parallel.norm().item()
        norm_total = eta_f32.norm().item()
        alignment = norm_parallel / (norm_total + 1e-8)

        # 各分量的能量占比
        norm_orthogonal = eta_orthogonal.norm().item()

        logger.info(
            f"  [PODI] Decomposition: "
            f"||η_parallel||={norm_parallel:.4f}, "
            f"||η_orthogonal||={norm_orthogonal:.4f}, "
            f"||η_total||={norm_total:.4f}, "
            f"alignment={alignment:.4f}"
        )

        # ── Step 5: 根据 alignment 决定注入策略 ──
        if alignment < cfg.podi_min_alignment:
            # alignment 极低: η_temporal 几乎完全与 prompt 正交 → 完全放弃注入
            logger.info(
                f"  [PODI] ⚠️ alignment={alignment:.4f} < threshold={cfg.podi_min_alignment} "
                f"→ η_temporal 与 prompt 严重不对齐，放弃注入 (fallback to random)"
            )
            # 返回 zero eta_temporal (等效于只用 eta_random)
            return torch.zeros_like(eta_temporal), 0.0

        # alignment 合格: 只注入 parallel 分量
        # Renormalize η_parallel 到 N(0,1) (保持扩散模型假设)
        parallel_mean = eta_parallel.mean()
        parallel_std = eta_parallel.std()

        if parallel_std < 1e-8:
            logger.warning("  [PODI] η_parallel std ≈ 0, falling back to normal blend")
            return eta_temporal.to(eta_temporal.dtype), cfg.alpha

        eta_aligned = (eta_parallel - parallel_mean) / parallel_std
        eta_aligned = eta_aligned.to(eta_temporal.dtype)

        # Effective alpha: 直接使用 podi_alpha (不做 alignment 缩放)
        # 设计理由: PODI 的价值在于"注入更安全的内容" (parallel 分量)
        # 而不是"改变注入量"。alignment 只做门控 (低于 min_alignment 时拒绝注入),
        # 注入量由 podi_alpha 直接控制，方便与 baseline alpha 做公平对比。
        # 如果想要更激进：因为注入的是安全分量，podi_alpha 可以设得比 baseline 的 alpha 大。
        effective_alpha = cfg.podi_alpha

        logger.info(
            f"  [PODI] alignment={alignment:.4f} → "
            f"effective_alpha={effective_alpha:.6f} "
            f"(podi_alpha={cfg.podi_alpha}, alignment used for gate only)"
        )
        logger.info(
            f"  [PODI] η_aligned: mean={eta_aligned.mean():.4f}, std={eta_aligned.std():.4f}"
        )

        # 诊断: channel 方向的分布
        channel_proj_mean = proj_scalar.mean().item()
        channel_proj_std = proj_scalar.std().item()
        logger.info(
            f"  [PODI] Channel projection: mean={channel_proj_mean:.4f}, "
            f"std={channel_proj_std:.4f}, d_unit top3={d_unit[:3].tolist()}"
        )

        return eta_aligned, effective_alpha

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """调用 Wan 2.1-1.3B 生成视频。"""
        cfg = self.config
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or NEGATIVE_PROMPT,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "guidance_scale": cfg.guidance_scale,
            "num_inference_steps": cfg.num_inference_steps,
            "generator": generator,
            "output_type": "pt",
        }
        if latents is not None:
            kwargs["latents"] = latents

        output = self.pipe(**kwargs)

        # 处理输出格式
        if hasattr(output, "frames"):
            video = output.frames
            if isinstance(video, list):
                import torchvision.transforms as T
                frames = [T.ToTensor()(f) for f in video[0]]
                video = torch.stack(frames, dim=1)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:
                    video = video[0]
                    if video.shape[0] == cfg.num_frames:
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]

        if video.min() < 0:
            video = denormalize_video(video)
        return video.clamp(0, 1)

    @torch.no_grad()
    def _generate_with_anchor(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        ref_trajectory: Dict[float, torch.Tensor],
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """
        灰盒生成：Latent Trajectory Soft Anchor。

        在标准 diffusers 生成流程中，通过 callback_on_step_end 在每步去噪后
        将当前 latent 向参考轨迹做 lerp 拉回，实现轨迹级运动引导。

        数学等价于在 Flow ODE 上加弹性恢复力:
            dz/dt = v_θ(z_t, t, c) + β_t * (z_ref_t - z_t)

        Args:
            prompt: 生成 prompt
            latents: 初始噪声 (可由 _get_latents 提供, 含 SVD prior)
            generator: 随机数生成器
            ref_trajectory: 参考轨迹 {t_value: z_ref_tensor(cpu)}
            negative_prompt: 负面 prompt
        """
        cfg = self.config
        num_steps = cfg.num_inference_steps

        # 预计算 β 退火调度
        beta_values = self._compute_beta_schedule(
            num_steps, cfg.anchor_beta_max, cfg.anchor_schedule
        )

        # 将参考轨迹的 t 值排序，用于最近邻查找
        traj_keys = sorted(ref_trajectory.keys())

        # 日志
        logger.info(
            f"  [Trajectory Anchor] β_max={cfg.anchor_beta_max}, "
            f"schedule={cfg.anchor_schedule}, "
            f"trajectory points={len(traj_keys)}, "
            f"gen steps={num_steps}"
        )
        logger.info(
            f"  [Trajectory Anchor] β schedule (first 5): "
            f"{[f'{b:.4f}' for b in beta_values[:5]]}, "
            f"(last 5): {[f'{b:.4f}' for b in beta_values[-5:]]}"
        )

        # ── 方案 2: cos-proportional β (L2+L3 融合) ──
        # 当 latents 非 None（经过 SVD blend），启用 cos-proportional 模式：
        #   effective_β = β_t × max(0, cos(gen, ref))
        # 力度正比于对齐度，不存在阈值跳变，不需要 cos_threshold
        # 纯 L3 (latents=None) 不启用此模式，保持原有行为
        use_cos_proportional = (latents is not None) and cfg.anchor_cos_threshold > 0
        if use_cos_proportional:
            logger.info(
                f"  [Trajectory Anchor] cos-proportional β 模式已启用 "
                f"(检测到 SVD-blended latents, effective_β = β_t × cos(gen,ref))"
            )
        else:
            logger.info(
                f"  [Trajectory Anchor] 标准模式 "
                f"(latents={'None→diffusers随机' if latents is None else 'provided'}, "
                f"cos_proportional={'OFF' if not use_cos_proportional else 'ON'})"
            )

        # ── 诊断: z_T 与 ref_trajectory 起点的对齐度 ──
        # ref_trajectory 来自 inversion(ref_latent → eta_inv)
        # L2 模式下 z_T = SVD-blended，和 eta_inv 不同
        # 这个 cos 值预示了前几步是否需要跳过
        t_max = max(traj_keys)
        z_ref_start = ref_trajectory[t_max]
        if latents is not None:
            cos_start = torch.nn.functional.cosine_similarity(
                latents.flatten().unsqueeze(0),
                z_ref_start.flatten().to(latents.device).unsqueeze(0),
            ).item()
            logger.info(
                f"  [Trajectory Anchor] 起点对齐诊断: "
                f"cos(z_T_actual, ref_traj[t={t_max:.3f}])={cos_start:.4f} "
                f"({'⚠️ 严重偏移' if cos_start < 0.1 else '⚡ 轻度偏移' if cos_start < 0.5 else '✅ 对齐良好'})"
            )
            logger.info(
                f"  [Trajectory Anchor] 源头差异说明: "
                f"z_T = SVD-blended (α={cfg.alpha}), "
                f"ref_traj[t=1] = raw eta_inv (无SVD无blend), "
                f"差异来自 SVD 滤波 + α-blend 稀释"
            )
        else:
            logger.info(
                f"  [Trajectory Anchor] 起点对齐诊断: "
                f"latents=None (diffusers 将随机采样), 预期 cos≈0"
            )

        # 锚定统计（在 callback 中累积）
        anchor_stats = {"steps_applied": 0, "skipped": 0, "total_shift": 0.0, "per_step": [], "first_anchor_step": None}

        def trajectory_anchor_callback(pipe, step_index, timestep, callback_kwargs):
            """每步去噪后，将 latent 向参考轨迹做 lerp。"""
            latents_current = callback_kwargs["latents"]

            # WanPipeline 的去噪从 t=1(noise) → t=0(data)
            # step_index: 0, 1, ..., num_steps-1
            # 去噪进度: 第 step_index 步完成后，对应 t = 1 - (step_index+1)/num_steps
            # 即 step 0 完成后 t≈0.97, 最后一步完成后 t≈0.0
            t_progress = 1.0 - (step_index + 1) / num_steps

            # 在参考轨迹中找最近的 t 点
            t_ref = min(traj_keys, key=lambda t: abs(t - t_progress))

            # 获取当前步的 β
            beta_t = beta_values[step_index]

            if beta_t > 1e-6:
                z_ref = ref_trajectory[t_ref].to(
                    device=latents_current.device,
                    dtype=latents_current.dtype,
                )

                # 计算余弦相似度
                cos_gen_ref = torch.nn.functional.cosine_similarity(
                    latents_current.flatten().unsqueeze(0),
                    z_ref.flatten().unsqueeze(0),
                ).item()

                # ── 方案 2: cos-proportional β ──
                # effective_β = β_t × max(0, cos(gen, ref))
                # 力度正比于对齐度: cos 低→几乎不拉, cos 高→正常拉
                # 纯 L3 模式不启用此逻辑，保持原始 β_t
                if use_cos_proportional:
                    effective_beta = beta_t * max(0.0, cos_gen_ref)
                else:
                    effective_beta = beta_t

                # effective_β 过小时跳过（节省计算）
                if effective_beta < 1e-5:
                    anchor_stats["skipped"] += 1
                    logger.info(
                        f"    [Anchor step {step_index:2d}/{num_steps}] "
                        f"t={t_progress:.3f}→ref_t={t_ref:.3f}, "
                        f"cos={cos_gen_ref:.4f}, β_sched={beta_t:.4f}, "
                        f"effective_β={effective_beta:.6f} → SKIP (力度不足)"
                    )
                    return callback_kwargs

                # 首次启用记录
                if anchor_stats["first_anchor_step"] is None:
                    anchor_stats["first_anchor_step"] = step_index
                    logger.info(
                        f"    [Anchor step {step_index:2d}/{num_steps}] "
                        f"🟢 首次有效 anchor "
                        f"(cos={cos_gen_ref:.4f}, effective_β={effective_beta:.4f})"
                    )

                # Lerp 锚定: z_anchored = (1-effective_β)*z_gen + effective_β*z_ref
                latents_anchored = (1 - effective_beta) * latents_current + effective_beta * z_ref

                # 计算偏移量
                shift = (latents_anchored - latents_current).norm().item()

                anchor_stats["steps_applied"] += 1
                anchor_stats["total_shift"] += shift
                anchor_stats["per_step"].append({
                    "step": step_index, "t": t_progress, "t_ref": t_ref,
                    "beta": beta_t, "effective_beta": effective_beta,
                    "shift": shift, "cos_gen_ref": cos_gen_ref,
                })

                # 每步详细日志
                logger.info(
                    f"    [Anchor step {step_index:2d}/{num_steps}] "
                    f"t={t_progress:.3f}→ref_t={t_ref:.3f}, "
                    f"β_sched={beta_t:.4f}, effective_β={effective_beta:.4f}, "
                    f"shift={shift:.2f}, cos={cos_gen_ref:.4f}, "
                    f"latent_std={latents_current.std().item():.4f}"
                )

                callback_kwargs["latents"] = latents_anchored
            else:
                logger.debug(
                    f"    [Anchor step {step_index:2d}/{num_steps}] "
                    f"t={t_progress:.3f}, β={beta_t:.6f} < 1e-6, skipped"
                )

            return callback_kwargs

        # 构建生成参数
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or NEGATIVE_PROMPT,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "guidance_scale": cfg.guidance_scale,
            "num_inference_steps": num_steps,
            "generator": generator,
            "output_type": "pt",
            "callback_on_step_end": trajectory_anchor_callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        if latents is not None:
            kwargs["latents"] = latents

        output = self.pipe(**kwargs)

        # 日志：锚定效果总结
        n_applied = anchor_stats["steps_applied"]
        n_skipped = anchor_stats["skipped"]
        first_step = anchor_stats["first_anchor_step"]
        total_shift = anchor_stats["total_shift"]
        avg_shift = total_shift / max(n_applied, 1)
        logger.info("  ═══════════════════════════════════════════════")
        logger.info(f"  [Trajectory Anchor] 生成完成总结:")
        logger.info(
            f"    applied={n_applied}, skipped={n_skipped} "
            f"({'cos-proportional' if use_cos_proportional else 'standard'}), "
            f"total={n_applied+n_skipped}/{num_steps}"
        )
        logger.info(
            f"    first_anchor_step={first_step if first_step is not None else 'NEVER'}, "
            f"total_shift={total_shift:.4f}, avg_shift={avg_shift:.4f}"
        )
        if anchor_stats["per_step"]:
            # 前期 vs 后期对比
            mid = len(anchor_stats["per_step"]) // 2
            early_steps = anchor_stats["per_step"][:mid]
            late_steps = anchor_stats["per_step"][mid:]
            early_avg_cos = sum(s["cos_gen_ref"] for s in early_steps) / max(len(early_steps), 1)
            late_avg_cos = sum(s["cos_gen_ref"] for s in late_steps) / max(len(late_steps), 1)
            early_avg_shift = sum(s["shift"] for s in early_steps) / max(len(early_steps), 1)
            late_avg_shift = sum(s["shift"] for s in late_steps) / max(len(late_steps), 1)
            logger.info(
                f"    前半段 (step 0~{mid-1}): avg_cos(gen,ref)={early_avg_cos:.4f}, "
                f"avg_shift={early_avg_shift:.2f}"
            )
            logger.info(
                f"    后半段 (step {mid}~{n_applied-1}): avg_cos(gen,ref)={late_avg_cos:.4f}, "
                f"avg_shift={late_avg_shift:.2f}"
            )
            logger.info(
                f"    趋势: cos 从 {early_avg_cos:.4f}→{late_avg_cos:.4f} "
                f"({'收敛' if late_avg_cos > early_avg_cos else '发散'}), "
                f"shift 从 {early_avg_shift:.2f}→{late_avg_shift:.2f} "
                f"({'衰减 ✓' if late_avg_shift < early_avg_shift else '异常 ⚠️'})"
            )
        logger.info("  ═══════════════════════════════════════════════")

        # 处理输出格式（同 _generate）
        if hasattr(output, "frames"):
            video = output.frames
            if isinstance(video, list):
                import torchvision.transforms as T
                frames = [T.ToTensor()(f) for f in video[0]]
                video = torch.stack(frames, dim=1)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:
                    video = video[0]
                    if video.shape[0] == cfg.num_frames:
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]

        if video.min() < 0:
            video = denormalize_video(video)
        return video.clamp(0, 1)

    def _compute_beta_schedule(
        self,
        num_steps: int,
        beta_max: float,
        schedule_type: str,
    ) -> List[float]:
        """
        计算每步的轨迹锚定强度 β_t。

        设计原则:
        - 前期 (接近噪声，step_index 小): 强锚定，确保运动方向正确
        - 后期 (接近数据，step_index 大): 弱/无锚定，让模型自由生成细节

        Args:
            num_steps: 总去噪步数
            beta_max: 最大 β 值
            schedule_type: 调度类型

        Returns:
            List[float]: 长度为 num_steps 的 β 值列表
        """
        import math

        if schedule_type == "cosine_decay":
            # β_t = β_max * cos(π/2 * i/(N-1))
            # step 0 → β_max, step N-1 → 0
            beta = [
                beta_max * math.cos(math.pi / 2 * i / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
        elif schedule_type == "linear_decay":
            # β_t = β_max * (1 - i/(N-1))
            beta = [
                beta_max * (1.0 - i / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
        elif schedule_type == "constant":
            beta = [beta_max] * num_steps
        elif schedule_type == "warmup_decay":
            # 前 20% warmup 到 β_max，后 80% cosine decay 到 0
            warmup_steps = max(int(0.2 * num_steps), 1)
            beta = []
            for i in range(warmup_steps):
                beta.append(beta_max * (0.5 + 0.5 * i / max(warmup_steps - 1, 1)))
            remaining = num_steps - warmup_steps
            for i in range(remaining):
                beta.append(
                    beta_max * math.cos(math.pi / 2 * i / max(remaining - 1, 1))
                )
        else:
            logger.warning(
                f"  [Trajectory Anchor] Unknown schedule '{schedule_type}', "
                f"falling back to cosine_decay"
            )
            return self._compute_beta_schedule(num_steps, beta_max, "cosine_decay")

        return beta

    def _compute_gamma_schedule(
        self,
        num_steps: int,
        gamma_max: float,
        schedule_type: str,
    ) -> List[float]:
        """
        计算每步的 VDA 方向引导强度 γ_t。

        设计原则 (与 position lerp 的 β 调度不同):
        - 前期 (接近噪声, step_index 小): 弱引导 (轨迹还在收敛, 方向不确定)
        - 中期 (中间阶段): 强引导 (这是最需要方向修正的时候, 位置 lerp 在此失败)
        - 后期 (接近数据, step_index 大): 弱引导 (轨迹自然收敛, ODE 流形约束强)

        Args:
            num_steps: 总去噪步数
            gamma_max: 最大 γ 值
            schedule_type: 调度类型

        Returns:
            List[float]: 长度为 num_steps 的 γ 值列表
        """

        if schedule_type == "constant":
            gamma = [gamma_max] * num_steps
        elif schedule_type == "middle_peak":
            # 中间峰: γ_t = γ_max * sin(π * i / (N-1))
            # step 0 → 0, step N/2 → γ_max, step N-1 → 0
            gamma = [
                gamma_max * math.sin(math.pi * i / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
        elif schedule_type == "warmup_decay":
            # 前 20% warmup, 后 80% cosine decay
            warmup_steps = max(int(0.2 * num_steps), 1)
            gamma = []
            for i in range(warmup_steps):
                gamma.append(gamma_max * (0.5 + 0.5 * i / max(warmup_steps - 1, 1)))
            remaining = num_steps - warmup_steps
            for i in range(remaining):
                gamma.append(
                    gamma_max * math.cos(math.pi / 2 * i / max(remaining - 1, 1))
                )
        elif schedule_type == "cosine_decay":
            # 从 γ_max 递减到 0 (与 position lerp 类似)
            gamma = [
                gamma_max * math.cos(math.pi / 2 * i / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
        else:
            logger.warning(
                f"  [VDA Schedule] Unknown schedule '{schedule_type}', "
                f"falling back to middle_peak"
            )
            return self._compute_gamma_schedule(num_steps, gamma_max, "middle_peak")

        # ── 调度日志 ──
        logger.info(
            f"  [VDA Schedule] type={schedule_type}, gamma_max={gamma_max}, "
            f"num_steps={num_steps}"
        )
        # 显示关键节点的 gamma 值
        if num_steps <= 10:
            logger.info(
                f"  [VDA Schedule] all values: {[f'{g:.4f}' for g in gamma]}"
            )
        else:
            peak_idx = max(range(num_steps), key=lambda i: gamma[i])
            logger.info(
                f"  [VDA Schedule] peak at step {peak_idx} (γ={gamma[peak_idx]:.4f}), "
                f"first 5: {[f'{g:.4f}' for g in gamma[:5]]}, "
                f"last 5: {[f'{g:.4f}' for g in gamma[-5:]]}"
            )
            # 显示中间5步
            mid = num_steps // 2
            logger.info(
                f"  [VDA Schedule] middle 5: {[f'{g:.4f}' for g in gamma[mid-2:mid+3]]}"
            )
        logger.info(
            f"  [VDA Schedule] sum(γ)={sum(gamma):.4f}, "
            f"mean(γ)={sum(gamma)/len(gamma):.4f}"
        )

        return gamma

    def _compute_vda_quality_scale(
        self,
        eta_temporal: Optional[torch.Tensor],
    ) -> float:
        """
        计算VDA质量缩放因子 (替代旧方案的质量门控)。

        核心思路: 基于 η_temporal 的帧间余弦相似度 (motion coherence)
        和 motion_strength (时序能量占比) 计算一个 0~1 的缩放因子。

        软门控策略 (vda_quality_scale=True):
            - motion_coherence 高 → scale → 1.0 (完整引导)
            - motion_coherence 低 → scale → 0.1~0.3 (保留微弱引导, 不硬跳过)
            - 这样即使样本运动混乱, 也能获得"微弱的方向暗示"

        硬门控策略 (vda_quality_scale=False):
            - motion_coherence < threshold → scale = 0 (完全跳过, 退回旧方案)

        Args:
            eta_temporal: SVD 滤波后的噪声

        Returns:
            scale: 0~1 的缩放因子, 用于 effective_γ = γ_t * scale
        """
        cfg = self.config

        if eta_temporal is None:
            logger.info("  [VDA Quality] eta_temporal=None → scale=1.0 (无门控)")
            return 1.0

        logger.info(
            f"  [VDA Quality] 输入 eta_temporal: shape={list(eta_temporal.shape)}, "
            f"dtype={eta_temporal.dtype}, device={eta_temporal.device}"
        )

        # ── 计算 motion coherence (帧间余弦相似度) ──
        eta_gate = eta_temporal
        if eta_gate.dim() == 5:
            if eta_gate.shape[2] > eta_gate.shape[1]:
                eta_gate = eta_gate.permute(0, 2, 1, 3, 4)
                logger.info(
                    f"  [VDA Quality] permute: [{eta_temporal.shape}] → [{list(eta_gate.shape)}] "
                    f"(frame>channel, swapped)"
                )
            num_frames_gate = eta_gate.shape[2]
        elif eta_gate.dim() == 4:
            num_frames_gate = eta_gate.shape[1]
            eta_gate = eta_gate.unsqueeze(0)
            logger.info(
                f"  [VDA Quality] unsqueeze: [{list(eta_temporal.shape)}] → [{list(eta_gate.shape)}]"
            )
        else:
            logger.warning(
                f"  [VDA Quality] 不支持的维度: dim={eta_gate.dim()}, shape={list(eta_gate.shape)} → scale=1.0"
            )
            return 1.0

        if num_frames_gate < 2:
            logger.info(
                f"  [VDA Quality] 帧数={num_frames_gate} < 2, 无法计算帧间cos → scale=1.0"
            )
            return 1.0

        logger.info(
            f"  [VDA Quality] 解析后: eta_gate shape={list(eta_gate.shape)}, "
            f"num_frames={num_frames_gate}"
        )

        frame_cos_sims = []
        for f in range(num_frames_gate - 1):
            f1 = eta_gate[0, :, f, :, :].flatten()
            f2 = eta_gate[0, :, f + 1, :, :].flatten()
            cos = torch.nn.functional.cosine_similarity(
                f1.unsqueeze(0), f2.unsqueeze(0)
            ).item()
            frame_cos_sims.append(cos)

        mean_cos = sum(frame_cos_sims) / len(frame_cos_sims)
        min_cos = min(frame_cos_sims)
        max_cos = max(frame_cos_sims)

        logger.info(
            f"  [VDA Quality] 逐帧cos (共{len(frame_cos_sims)}对): "
            f"mean={mean_cos:.4f}, min={min_cos:.4f}, max={max_cos:.4f}"
        )
        # 显示前5帧和后5帧的cos值
        if len(frame_cos_sims) > 10:
            logger.info(
                f"    前5帧: {[f'{c:.3f}' for c in frame_cos_sims[:5]]}, "
                f"后5帧: {[f'{c:.3f}' for c in frame_cos_sims[-5:]]}"
            )
        else:
            logger.info(
                f"    所有帧: {[f'{c:.3f}' for c in frame_cos_sims]}"
            )

        # ── 计算 motion strength (从 SVD diagnostics 中获取) ──
        motion_strength = 0.0
        if hasattr(self, '_svd_diagnostics') and self._svd_diagnostics is not None:
            motion_strength = self._svd_diagnostics.get('motion_strength', 0.0)
            logger.info(
                f"  [VDA Quality] SVD diagnostics: motion_strength={motion_strength:.4f}"
            )
        else:
            logger.info(
                f"  [VDA Quality] 无 SVD diagnostics (未使用 SVD), motion_strength=0.0"
            )

        logger.info(
            f"  [VDA Quality] motion_coherence={mean_cos:.4f}, "
            f"motion_strength={motion_strength:.4f}"
        )

        if not cfg.vda_quality_scale:
            # 硬门控: 低于阈值 → 0, 高于 → 1
            if mean_cos < cfg.anchor_quality_threshold:
                logger.info(
                    f"  [VDA Quality] ❌ 硬门控: mean_cos={mean_cos:.4f} < {cfg.anchor_quality_threshold}, "
                    f"scale=0 (跳过 VDA)"
                )
                return 0.0
            else:
                logger.info(
                    f"  [VDA Quality] ✅ 硬门控: mean_cos={mean_cos:.4f} >= {cfg.anchor_quality_threshold}, "
                    f"scale=1.0"
                )
                return 1.0

        # ── 软门控: sigmoid 映射 ──
        # mean_cos 范围大约 [-0.2, 0.3]
        # 映射到 [min_scale, 1.0]
        min_scale = 0.1  # 即使运动混乱, 也保留 10% 引导
        threshold = cfg.anchor_quality_threshold  # 默认 0.05

        # 使用 sigmoid 平滑映射
        # 当 mean_cos = threshold 时, scale ≈ 0.5
        # 当 mean_cos >> threshold 时, scale → 1.0
        # 当 mean_cos << threshold 时, scale → min_scale
        k = 20.0  # sigmoid 陡度
        exp_arg = -k * (mean_cos - threshold)
        logger.info(
            f"  [VDA Quality] sigmoid计算: mean_cos={mean_cos:.6f}, threshold={threshold}, "
            f"k={k}, exp_arg={exp_arg:.4f}"
        )
        # 数值保护: 防止 exp overflow
        exp_arg = max(min(exp_arg, 500.0), -500.0)
        sigmoid_val = 1.0 / (1.0 + math.exp(exp_arg))
        scale = min_scale + (1.0 - min_scale) * sigmoid_val

        logger.info(
            f"  [VDA Quality] 软门控: mean_cos={mean_cos:.4f}, "
            f"threshold={threshold}, sigmoid_val={sigmoid_val:.4f}, scale={scale:.4f} "
            f"({'✅ 强引导' if scale > 0.7 else '⚡ 弱引导' if scale > 0.3 else '⚠️ 微弱引导'})"
        )

        return scale

    @torch.no_grad()
    def _generate_with_vda(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        ref_trajectory: Dict[float, torch.Tensor],
        eta_temporal: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """
        L3 V2: Velocity Direction Anchor (VDA) 生成。

        与旧方案 (position lerp) 的核心区别:
            - 旧方案: z_anchored = (1-β)*z_gen + β*z_ref (直接在位置空间拉)
              问题: z_gen 和 z_ref 从几乎正交的起点出发, 位置 lerp 无物理意义
            - 新方案: z_adjusted = z_current + γ * v_ref_⊥ * dt (在速度方向空间微调)
              优势: 不要求起点对齐, 只关心"下一步往哪走"

        数学:
            1. 从 ref_trajectory 中取相邻两点, 计算参考速度方向:
               v_ref = (z_ref[t_prev] - z_ref[t_curr]) / dt_ref
               (反演过程从 t=1→0, 所以取反后得到生成方向的速度)
            2. 在当前 latent 位置, 计算参考速度的正交分量:
               v_ref_⊥ = v_ref - (v_ref · v_gen / ‖v_gen‖²) * v_gen
            3. 将正交分量作为方向修正冲量加到当前 latent 上:
               Δz = γ * v_ref_⊥ * dt
               z_adjusted = z_current + Δz

        Args:
            prompt: 生成 prompt
            latents: 初始噪声 (可由 _get_latents 提供, 含 SVD prior)
            generator: 随机数生成器
            ref_trajectory: 参考轨迹 {t_value: z_ref_tensor(cpu)}
            eta_temporal: SVD 滤波后的噪声 (用于质量门控)
            negative_prompt: 负面 prompt
        """
        cfg = self.config
        num_steps = cfg.num_inference_steps

        logger.info("  ═══════════════════════════════════════════════")
        logger.info("  [VDA] 开始 Velocity Direction Anchor 生成")
        logger.info("  ═══════════════════════════════════════════════")

        # ── 预计算 γ 调度 ──
        gamma_values = self._compute_gamma_schedule(
            num_steps, cfg.vda_gamma, cfg.vda_schedule
        )

        # ── 质量门控缩放 ──
        quality_scale = 1.0
        if cfg.vda_quality_gate:
            quality_scale = self._compute_vda_quality_scale(eta_temporal)
            if quality_scale < 1e-6:
                # 完全跳过, 走标准生成
                logger.info(
                    f"  [VDA] 质量门控 scale≈0, 跳过 VDA, 走标准生成"
                )
                return self._generate(prompt, latents, generator, negative_prompt)

        # ── 参考轨迹的 t 值排序 ──
        traj_keys = sorted(ref_trajectory.keys())
        dt_ref = 1.0 / max(len(traj_keys) - 1, 1)  # ref 轨迹的时间步长
        dt_gen = 1.0 / num_steps  # 生成轨迹的时间步长

        # ── 诊断: 起点对齐度 ──
        t_max = max(traj_keys)
        z_ref_start = ref_trajectory[t_max]
        if latents is not None:
            cos_start = torch.nn.functional.cosine_similarity(
                latents.flatten().unsqueeze(0),
                z_ref_start.flatten().to(latents.device).unsqueeze(0),
            ).item()
            logger.info(
                f"  [VDA] 起点对齐诊断: "
                f"cos(z_T, ref_traj[t={t_max:.3f}])={cos_start:.4f} "
                f"({'⚠️ 严重偏移' if cos_start < 0.1 else '⚡ 轻度偏移' if cos_start < 0.5 else '✅ 对齐良好'})"
            )
            logger.info(
                f"  [VDA] 注意: VDA 不要求起点对齐, 只使用参考轨迹的速度方向"
            )

        # ── 日志: 配置总结 ──
        logger.info(
            f"  [VDA] γ_max={cfg.vda_gamma}, schedule={cfg.vda_schedule}, "
            f"quality_scale={quality_scale:.4f}, "
            f"perp_only={cfg.vda_use_perp_only}, "
            f"norm_clamp={cfg.vda_norm_clamp}, "
            f"steps=[{cfg.vda_start_step}, {cfg.vda_end_step if cfg.vda_end_step >= 0 else num_steps}]"
        )
        logger.info(
            f"  [VDA] γ schedule (first 5): "
            f"{[f'{g:.4f}' for g in gamma_values[:5]]}, "
            f"(last 5): {[f'{g:.4f}' for g in gamma_values[-5:]]}"
        )

        # ── VDA 统计 ──
        vda_stats = {
            "steps_applied": 0,
            "steps_skipped": 0,
            "total_shift": 0.0,
            "per_step": [],
            "first_vda_step": None,
            "avg_ref_speed_norm": 0.0,
            "avg_perp_ratio": 0.0,  # 正交分量占比
        }

        # 保存前一步的 latent (用于差分估计生成速度)
        prev_latent_holder = [None]

        def vda_callback(pipe, step_index, timestep, callback_kwargs):
            """VDA callback: 速度方向引导 (带 try-except 错误保护)。"""
            try:
                latents_current = callback_kwargs["latents"]

                # ── 步数范围检查 ──
                end_step = cfg.vda_end_step if cfg.vda_end_step >= 0 else num_steps
                if step_index < cfg.vda_start_step or step_index >= end_step:
                    prev_latent_holder[0] = latents_current.clone()
                    vda_stats["steps_skipped"] += 1
                    if step_index == cfg.vda_start_step - 1:
                        logger.info(
                            f"    [VDA] step {step_index}: 范围外, 下一步开始 VDA "
                            f"(start_step={cfg.vda_start_step})"
                        )
                    return callback_kwargs

                # ── 当前 t 进度 ──
                # WanPipeline: step_index 完成后, t 从 ~1.0 降到 1-(step_index+1)/N
                t_progress = 1.0 - (step_index + 1) / num_steps

                # ── Step 1: 获取参考速度方向 ──
                # 在 ref_trajectory 中找当前 t 和前一步 t 的最近点
                t_curr_ref = min(traj_keys, key=lambda t: abs(t - t_progress))
                t_prev_ref = min(traj_keys, key=lambda t: abs(t - (t_progress + dt_gen)))

                if t_curr_ref == t_prev_ref:
                    # 找不到两个不同的参考点 (边界情况), 跳过
                    prev_latent_holder[0] = latents_current.clone()
                    vda_stats["steps_skipped"] += 1
                    if step_index % 10 == 0:
                        logger.debug(
                            f"    [VDA step {step_index:2d}] 跳过: "
                            f"t_curr_ref==t_prev_ref={t_curr_ref:.4f}"
                        )
                    return callback_kwargs

                z_ref_curr = ref_trajectory[t_curr_ref].to(
                    device=latents_current.device, dtype=latents_current.dtype
                )
                z_ref_prev = ref_trajectory[t_prev_ref].to(
                    device=latents_current.device, dtype=latents_current.dtype
                )

                # 参考速度: 反演方向 (t 大 → t 小), 取差后得到反演方向的速度
                dt_actual = t_prev_ref - t_curr_ref
                if abs(dt_actual) < 1e-10:
                    prev_latent_holder[0] = latents_current.clone()
                    vda_stats["steps_skipped"] += 1
                    return callback_kwargs

                v_ref = (z_ref_prev - z_ref_curr) / abs(dt_actual)

                # ── Step 2: 估计生成速度 (差分) ──
                if prev_latent_holder[0] is None:
                    # 第一步没有速度信息, 跳过
                    prev_latent_holder[0] = latents_current.clone()
                    vda_stats["steps_skipped"] += 1
                    logger.info(
                        f"    [VDA step {step_index:2d}] 跳过: 首步无速度信息"
                    )
                    return callback_kwargs

                # 检查 prev_latent_holder 的 device 和 dtype
                if prev_latent_holder[0].device != latents_current.device or \
                   prev_latent_holder[0].dtype != latents_current.dtype:
                    logger.warning(
                        f"    [VDA step {step_index:2d}] device/dtype 不匹配: "
                        f"prev=({prev_latent_holder[0].device}, {prev_latent_holder[0].dtype}) "
                        f"vs curr=({latents_current.device}, {latents_current.dtype}), 自动修正"
                    )
                    prev_latent_holder[0] = prev_latent_holder[0].to(
                        device=latents_current.device, dtype=latents_current.dtype
                    )

                v_gen = (latents_current - prev_latent_holder[0]) / dt_gen

                # ── Step 3: 速度方向分解 ──
                v_ref_flat = v_ref.flatten()
                v_gen_flat = v_gen.flatten()

                v_gen_norm_sq = v_gen_flat.dot(v_gen_flat)
                if v_gen_norm_sq < 1e-12:
                    # 生成速度几乎为零, 无法做投影分解
                    prev_latent_holder[0] = latents_current.clone()
                    vda_stats["steps_skipped"] += 1
                    logger.debug(
                        f"    [VDA step {step_index:2d}] 跳过: v_gen_norm_sq={v_gen_norm_sq:.2e} ≈ 0"
                    )
                    return callback_kwargs

                # v_ref 在 v_gen 方向的投影 (平行分量)
                proj_scalar = v_ref_flat.dot(v_gen_flat) / v_gen_norm_sq
                v_ref_parallel = proj_scalar * v_gen_flat
                v_ref_perp = v_ref_flat - v_ref_parallel

                # 正交分量占比
                v_ref_norm = v_ref_flat.norm().item()
                v_gen_norm = math.sqrt(v_gen_norm_sq.item())
                v_ref_perp_norm = v_ref_perp.norm().item()
                perp_ratio = v_ref_perp_norm / max(v_ref_norm, 1e-8)

                # v_ref 与 v_gen 的夹角 (关键诊断指标)
                cos_v_ref_v_gen = (v_ref_flat.dot(v_gen_flat) / (v_ref_norm * v_gen_norm)).item()
                cos_v_ref_v_gen = max(min(cos_v_ref_v_gen, 1.0), -1.0)  # clamp to [-1, 1]
                angle_deg = math.degrees(math.acos(cos_v_ref_v_gen))

                # ── Step 4: 计算方向修正冲量 ──
                gamma_t = gamma_values[step_index] * quality_scale

                if cfg.vda_use_perp_only:
                    # 只注入正交分量 (不改变速度大小, 最安全)
                    delta_z = gamma_t * v_ref_perp * dt_gen
                else:
                    # 混合注入: 正交分量 + 微弱平行分量
                    delta_z = gamma_t * (
                        v_ref_perp + cfg.vda_parallel_weight * v_ref_parallel
                    ) * dt_gen

                # 恢复形状
                delta_z = delta_z.reshape(latents_current.shape)

                # ── Step 5: 范数钳制 (安全阀) ──
                latent_norm = latents_current.norm().item()
                delta_norm_before_clamp = delta_z.norm().item()
                clamped = False
                if cfg.vda_norm_clamp > 0:
                    max_delta = cfg.vda_norm_clamp * latent_norm
                    if delta_norm_before_clamp > max_delta:
                        delta_z = delta_z * (max_delta / delta_norm_before_clamp)
                        clamped = True
                        logger.info(
                            f"    [VDA step {step_index:2d}] ⚠️ Δz clamped: "
                            f"{delta_norm_before_clamp:.4f} → {max_delta:.4f} "
                            f"(ratio was {delta_norm_before_clamp/latent_norm:.4f}, "
                            f"limit={cfg.vda_norm_clamp})"
                        )

                # ── Step 6: 应用修正 ──
                latents_adjusted = latents_current + delta_z

                # 计算偏移量 (用于统计)
                shift = delta_z.norm().item()
                shift_ratio = shift / max(latent_norm, 1e-8)  # Δz / ‖z‖

                vda_stats["steps_applied"] += 1
                vda_stats["total_shift"] += shift
                vda_stats["avg_ref_speed_norm"] += v_ref_norm
                vda_stats["avg_perp_ratio"] += perp_ratio
                vda_stats["per_step"].append({
                    "step": step_index,
                    "t": t_progress,
                    "t_curr_ref": t_curr_ref,
                    "t_prev_ref": t_prev_ref,
                    "gamma_t": gamma_t,
                    "shift": shift,
                    "shift_ratio": shift_ratio,
                    "v_ref_norm": v_ref_norm,
                    "v_gen_norm": v_gen_norm,
                    "perp_ratio": perp_ratio,
                    "proj_scalar": proj_scalar.item(),
                    "angle_deg": angle_deg,
                    "cos_v_ref_v_gen": cos_v_ref_v_gen,
                    "clamped": clamped,
                })

                if vda_stats["first_vda_step"] is None:
                    vda_stats["first_vda_step"] = step_index

                # 每 5 步详细日志 + 最后一步
                if step_index % 5 == 0 or step_index == num_steps - 1 or step_index == cfg.vda_start_step:
                    logger.info(
                        f"    [VDA step {step_index:2d}/{num_steps}] "
                        f"t={t_progress:.3f}, γ_eff={gamma_t:.4f}, "
                        f"shift={shift:.4f} ({shift_ratio:.4f}·‖z‖), "
                        f"angle(v_ref,v_gen)={angle_deg:.1f}°, "
                        f"v_ref={v_ref_norm:.2f}, v_gen={v_gen_norm:.2f}, "
                        f"perp_ratio={perp_ratio:.4f}, proj={proj_scalar.item():.4f}"
                    )

                callback_kwargs["latents"] = latents_adjusted
                prev_latent_holder[0] = latents_adjusted.clone()

                return callback_kwargs

            except Exception as e:
                # 错误保护: 任何异常都不应中断生成流程
                logger.error(
                    f"    [VDA step {step_index:2d}] ❌ 异常: {type(e).__name__}: {e}"
                )
                import traceback
                logger.error(traceback.format_exc())
                logger.warning(
                    f"    [VDA step {step_index:2d}] 跳过本步 VDA, 继续标准生成"
                )
                # 不修改 latents, 返回原始 callback_kwargs
                if "latents" in callback_kwargs:
                    prev_latent_holder[0] = callback_kwargs["latents"].clone()
                return callback_kwargs

        # ── 构建生成参数 ──
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or NEGATIVE_PROMPT,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "guidance_scale": cfg.guidance_scale,
            "num_inference_steps": num_steps,
            "generator": generator,
            "output_type": "pt",
            "callback_on_step_end": vda_callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        if latents is not None:
            kwargs["latents"] = latents

        output = self.pipe(**kwargs)

        # ── VDA 统计总结 ──
        n_applied = vda_stats["steps_applied"]
        n_skipped = vda_stats["steps_skipped"]
        total_shift = vda_stats["total_shift"]
        avg_shift = total_shift / max(n_applied, 1)
        avg_ref_speed = vda_stats["avg_ref_speed_norm"] / max(n_applied, 1)
        avg_perp_ratio = vda_stats["avg_perp_ratio"] / max(n_applied, 1)

        logger.info("  ═══════════════════════════════════════════════")
        logger.info(f"  [VDA] 生成完成总结:")
        logger.info(
            f"    applied={n_applied}, skipped={n_skipped}, "
            f"total={n_applied+n_skipped}/{num_steps}"
        )
        logger.info(
            f"    first_vda_step={vda_stats['first_vda_step'] if vda_stats['first_vda_step'] is not None else 'NEVER'}, "
            f"total_shift={total_shift:.4f}, avg_shift={avg_shift:.4f}"
        )
        logger.info(
            f"    avg_ref_speed_norm={avg_ref_speed:.2f}, "
            f"avg_perp_ratio={avg_perp_ratio:.4f}"
        )
        if vda_stats["per_step"]:
            # ── 角度统计 ──
            angles = [s["angle_deg"] for s in vda_stats["per_step"]]
            avg_angle = sum(angles) / len(angles)
            min_angle = min(angles)
            max_angle = max(angles)
            logger.info(
                f"    angle(v_ref,v_gen): avg={avg_angle:.1f}°, "
                f"min={min_angle:.1f}°, max={max_angle:.1f}°"
            )

            # ── shift_ratio 统计 ──
            shift_ratios = [s["shift_ratio"] for s in vda_stats["per_step"]]
            avg_shift_ratio = sum(shift_ratios) / len(shift_ratios)
            max_shift_ratio = max(shift_ratios)
            logger.info(
                f"    shift_ratio (Δz/‖z‖): avg={avg_shift_ratio:.6f}, "
                f"max={max_shift_ratio:.6f}"
            )

            # ── clamp 统计 ──
            n_clamped = sum(1 for s in vda_stats["per_step"] if s.get("clamped", False))
            if n_clamped > 0:
                logger.info(
                    f"    ⚠️ {n_clamped}/{n_applied} 步被 clamp "
                    f"(norm_clamp={cfg.vda_norm_clamp})"
                )
            else:
                logger.info(
                    f"    ✅ 无 clamp 发生 (norm_clamp={cfg.vda_norm_clamp})"
                )

            # ── v_gen 统计 ──
            v_gen_norms = [s.get("v_gen_norm", 0) for s in vda_stats["per_step"]]
            avg_v_gen = sum(v_gen_norms) / len(v_gen_norms)
            logger.info(
                f"    v_gen_norm: avg={avg_v_gen:.2f}, "
                f"v_ref_norm: avg={avg_ref_speed:.2f}, "
                f"ratio(v_ref/v_gen)={avg_ref_speed/max(avg_v_gen, 1e-8):.4f}"
            )

            # ── 前后半段对比 ──
            mid = len(vda_stats["per_step"]) // 2
            early = vda_stats["per_step"][:mid]
            late = vda_stats["per_step"][mid:]
            early_avg_perp = sum(s["perp_ratio"] for s in early) / max(len(early), 1)
            late_avg_perp = sum(s["perp_ratio"] for s in late) / max(len(late), 1)
            early_avg_shift = sum(s["shift"] for s in early) / max(len(early), 1)
            late_avg_shift = sum(s["shift"] for s in late) / max(len(late), 1)
            early_avg_angle = sum(s.get("angle_deg", 0) for s in early) / max(len(early), 1)
            late_avg_angle = sum(s.get("angle_deg", 0) for s in late) / max(len(late), 1)
            logger.info(
                f"    前半段: avg_perp_ratio={early_avg_perp:.4f}, "
                f"avg_shift={early_avg_shift:.4f}, avg_angle={early_avg_angle:.1f}°"
            )
            logger.info(
                f"    后半段: avg_perp_ratio={late_avg_perp:.4f}, "
                f"avg_shift={late_avg_shift:.4f}, avg_angle={late_avg_angle:.1f}°"
            )
            logger.info(
                f"    趋势: perp_ratio {early_avg_perp:.4f}→{late_avg_perp:.4f} "
                f"({'正交性减弱' if late_avg_perp < early_avg_perp else '正交性增强'}), "
                f"shift {early_avg_shift:.4f}→{late_avg_shift:.4f} "
                f"({'衰减 ✓' if late_avg_shift < early_avg_shift else '增强 ⚠️'}), "
                f"angle {early_avg_angle:.1f}°→{late_avg_angle:.1f}° "
                f"({'收敛 ✓' if late_avg_angle < early_avg_angle else '发散 ⚠️'})"
            )

            # ── 逐步 shift 曲线 (紧凑格式, 每10步一组) ──
            if n_applied > 10:
                shift_curve = [f"{s['shift']:.3f}" for s in vda_stats["per_step"]]
                logger.info(
                    f"    shift curve (所有步): [{', '.join(shift_curve)}]"
                )
                ratio_curve = [f"{s['shift_ratio']:.5f}" for s in vda_stats["per_step"]]
                logger.info(
                    f"    shift_ratio curve: [{', '.join(ratio_curve)}]"
                )

        logger.info("  ═══════════════════════════════════════════════")

        # 处理输出格式（同 _generate）
        if hasattr(output, "frames"):
            video = output.frames
            if isinstance(video, list):
                import torchvision.transforms as T
                frames = [T.ToTensor()(f) for f in video[0]]
                video = torch.stack(frames, dim=1)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:
                    video = video[0]
                    if video.shape[0] == cfg.num_frames:
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]

        if video.min() < 0:
            video = denormalize_video(video)
        return video.clamp(0, 1)

    def _vlm_refine(
        self,
        ref_video: torch.Tensor,
        gen_video: torch.Tensor,
        prev_video: Optional[torch.Tensor],
        current_prompt: str,
        iteration: int,
    ) -> str:
        """
        改动点: Iterative VLM Optimization (+ Composite)

        创建对比视频 → VLM分析 → 返回优化后的prompt
        """
        # 创建VLM输入
        composite_path = f"/tmp/pflow_composite_iter{iteration}.mp4"

        if self.config.use_composite:
            # 三面板垂直拼接
            videos = [ref_video, gen_video] if prev_video is None else [ref_video, prev_video, gen_video]
            composite = create_vertical_composite(videos)
            save_video_tensor(composite, composite_path, fps=self.config.fps)
        else:
            # 仅发送生成视频
            save_video_tensor(gen_video, composite_path, fps=self.config.fps)

        # 调用VLM
        try:
            result = self.vlm_client.analyze_and_refine(
                composite_video_path=composite_path,
                current_prompt=current_prompt,
                iteration=iteration,
                i_max=self.config.i_max,
            )
            refined = result.get("refined_prompt", "")
            if refined and refined.strip():
                return refined
        except Exception as e:
            logger.warning(f"  VLM failed at iter {iteration}: {e}")

        return current_prompt

    def _compute_direction_quality(self, eta_temporal: torch.Tensor) -> float:
        """
        计算 SVD 方向质量分数 (Quality-Gated Alpha 方案 B)。

        综合三个子指标:
            1. temporal_coherence: 相邻帧余弦相似度均值 (方向一致性)
            2. spatial_anisotropy: 空间能量分布的各向异性 (非均匀=有方向)
            3. first_last_consistency: 首末帧余弦相似度 (长程一致性)

        Returns:
            quality ∈ [0, 1], 越高表示 SVD 方向越可靠
        """
        # 解析 eta_temporal 的帧维度
        eta = eta_temporal
        if eta.dim() == 5:
            # [B, C, F, H, W]
            if eta.shape[2] > eta.shape[1]:
                eta = eta.permute(0, 2, 1, 3, 4)  # → [B, C, F, H, W]
            num_frames = eta.shape[2]
            # reshape to (F, -1)
            frames_flat = eta[0].permute(1, 0, 2, 3).reshape(num_frames, -1)  # [F, C*H*W]
        elif eta.dim() == 4:
            # [C, F, H, W]
            num_frames = eta.shape[1]
            frames_flat = eta.permute(1, 0, 2, 3).reshape(num_frames, -1)  # [F, C*H*W]
        else:
            logger.warning(f"  [QGA] eta_temporal 维度异常: {eta.shape}, 返回 quality=0.5")
            return 0.5

        if num_frames < 2:
            logger.warning("  [QGA] 帧数<2, 返回 quality=0.5")
            return 0.5

        # ── 子指标 1: temporal_coherence ──
        # 相邻帧 cosine similarity 的均值
        cos_sims = []
        for f in range(num_frames - 1):
            cos = torch.nn.functional.cosine_similarity(
                frames_flat[f].unsqueeze(0), frames_flat[f + 1].unsqueeze(0)
            ).item()
            cos_sims.append(cos)
        temporal_coherence = max(0.0, min(1.0, sum(cos_sims) / len(cos_sims)))

        # ── 子指标 2: spatial_anisotropy ──
        # 按空间维度分 4 个象限，计算各象限能量占比
        if eta.dim() == 5:
            spatial = eta[0]  # [C, F, H, W]
        else:
            spatial = eta  # [C, F, H, W]
        # 总能量按空间分布: sum over C and F → [H, W]
        energy_map = spatial.pow(2).sum(dim=0).sum(dim=0)  # [H, W]
        h, w = energy_map.shape
        q_tl = energy_map[:h // 2, :w // 2].sum().item()
        q_tr = energy_map[:h // 2, w // 2:].sum().item()
        q_bl = energy_map[h // 2:, :w // 2].sum().item()
        q_br = energy_map[h // 2:, w // 2:].sum().item()
        total_energy = q_tl + q_tr + q_bl + q_br + 1e-8
        quadrant_ratios = [q_tl / total_energy, q_tr / total_energy,
                           q_bl / total_energy, q_br / total_energy]
        max_ratio = max(quadrant_ratios)
        # 归一化: 0.25 是均匀时的值, 0.50 时为满分
        spatial_anisotropy = max(0.0, min(1.0, (max_ratio - 0.25) / 0.25))

        # ── 子指标 3: first_last_consistency ──
        # 首帧和末帧的 cosine similarity
        first_last_cos = torch.nn.functional.cosine_similarity(
            frames_flat[0].unsqueeze(0), frames_flat[-1].unsqueeze(0)
        ).item()
        first_last_consistency = max(0.0, min(1.0, first_last_cos))

        # ── 综合 ──
        quality = (
            0.5 * temporal_coherence
            + 0.3 * spatial_anisotropy
            + 0.2 * first_last_consistency
        )
        quality = max(0.0, min(1.0, quality))

        logger.info(
            f"  [QGA Quality] temporal_coherence={temporal_coherence:.4f}, "
            f"spatial_anisotropy={spatial_anisotropy:.4f} (max_quad_ratio={max_ratio:.4f}), "
            f"first_last_consistency={first_last_consistency:.4f} → quality={quality:.4f}"
        )
        return quality

    def _encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        """
        编码文本到 embedding。

        Args:
            prompt: 文本 caption
            max_sequence_length: T5 最大序列长度。必须与生成阶段一致（WanPipeline.__call__ 默认 512）。
        """
        import inspect

        if hasattr(self.pipe, "encode_prompt"):
            sig = inspect.signature(self.pipe.encode_prompt)
            params = sig.parameters
            kwargs = {"prompt": prompt}
            if "device" in params:
                kwargs["device"] = self.device
            if "num_videos_per_prompt" in params:
                kwargs["num_videos_per_prompt"] = 1
            if "do_classifier_free_guidance" in params:
                kwargs["do_classifier_free_guidance"] = False
            if "max_sequence_length" in params:
                kwargs["max_sequence_length"] = max_sequence_length
            result = self.pipe.encode_prompt(**kwargs)
            return result[0] if isinstance(result, tuple) else result
        else:
            inputs = self.pipe.tokenizer(
                prompt, padding="max_length",
                max_length=max_sequence_length,
                truncation=True, return_tensors="pt",
            )
            return self.pipe.text_encoder(inputs.input_ids.to(self.device))[0]

    def _diagnose_prompt_noise_conflict(
        self,
        eta_temporal: torch.Tensor,
        prompt_embeds: torch.Tensor,
        caption: str,
        generator: torch.Generator,
    ) -> None:
        """
        诊断 Prompt-SVD 运动方向冲突。

        核心问题假设:
            v7e 的精炼 prompt 已经包含精确的运动描述 (如 "camera slowly pans left")，
            而 SVD 从参考视频中提取的运动方向可能与 prompt 描述的方向不一致。
            当两者冲突时，SVD noise prior 反而会干扰生成质量。

        诊断策略:
            1. 分析 η_temporal 的时间方向性 (帧间差分的方向一致性)
            2. 检查 prompt 中是否包含运动关键词
            3. 分析 η_temporal 各帧的方向变化模式
            4. 计算 "temporal coherence" — 相邻帧噪声的余弦相似度
        """
        logger.info("  ═══════════════════════════════════════════════")
        logger.info("  [CONFLICT DIAG] Prompt-SVD 方向冲突诊断")
        logger.info("  ═══════════════════════════════════════════════")

        # ── 1. Caption 运动关键词分析 ──
        motion_keywords = {
            "direction": ["left", "right", "up", "down", "forward", "backward",
                         "pan", "tilt", "zoom", "rotate", "orbit", "track"],
            "speed": ["slow", "fast", "rapid", "gentle", "steady", "quick",
                     "gradually", "smoothly", "abruptly"],
            "camera": ["camera", "lens", "angle", "perspective", "view",
                      "close-up", "wide shot", "dolly"],
            "motion": ["move", "sway", "drift", "flow", "glide", "shift",
                      "wave", "flutter", "shake", "tremble", "bounce"],
        }

        caption_lower = caption.lower()
        found_keywords = {}
        for category, keywords in motion_keywords.items():
            matches = [kw for kw in keywords if kw in caption_lower]
            if matches:
                found_keywords[category] = matches

        logger.info(f"  [CONFLICT DIAG] Caption 运动关键词:")
        if found_keywords:
            for cat, kws in found_keywords.items():
                logger.info(f"    {cat}: {kws}")
            logger.info(
                f"  [CONFLICT DIAG] ⚠️ Caption 包含 {sum(len(v) for v in found_keywords.values())} 个运动关键词 "
                f"→ prompt 已有明确运动指令，SVD 方向若不一致则产生冲突"
            )
        else:
            logger.info(
                f"  [CONFLICT DIAG] ✓ Caption 不含显式运动关键词 → SVD 冲突风险低"
            )

        # ── 2. η_temporal 时间方向性分析 ──
        # η_temporal shape: [B, C, F, H, W] 或 [B, F, C, H, W]
        # 通过帧间差分分析运动方向的一致性
        eta = eta_temporal
        if eta.dim() == 5:
            # 假设 [B, C, F, H, W]
            if eta.shape[2] > eta.shape[1]:
                # 可能是 [B, F, C, H, W]，转成 [B, C, F, H, W]
                eta = eta.permute(0, 2, 1, 3, 4)
            num_frames = eta.shape[2]
        elif eta.dim() == 4:
            # [C, F, H, W]
            num_frames = eta.shape[1]
            eta = eta.unsqueeze(0)  # → [1, C, F, H, W]
        else:
            logger.warning(f"  [CONFLICT DIAG] η_temporal 维度异常: {eta.shape}, 跳过帧分析")
            return

        logger.info(f"  [CONFLICT DIAG] η_temporal shape={list(eta_temporal.shape)}, parsed frames={num_frames}")

        if num_frames < 2:
            logger.warning("  [CONFLICT DIAG] 帧数<2, 无法分析时间方向性")
            return

        # ── 3. 帧间余弦相似度 (Temporal Coherence) ──
        # 如果 SVD 提取了有意义的运动，相邻帧应该有方向性 (cos > 0)
        frame_cos_sims = []
        for f in range(num_frames - 1):
            f1 = eta[0, :, f, :, :].flatten()
            f2 = eta[0, :, f + 1, :, :].flatten()
            cos = torch.nn.functional.cosine_similarity(
                f1.unsqueeze(0), f2.unsqueeze(0)
            ).item()
            frame_cos_sims.append(cos)

        mean_cos = sum(frame_cos_sims) / len(frame_cos_sims)
        std_cos = (sum((c - mean_cos) ** 2 for c in frame_cos_sims) / len(frame_cos_sims)) ** 0.5
        max_cos = max(frame_cos_sims)
        min_cos = min(frame_cos_sims)

        logger.info(
            f"  [CONFLICT DIAG] 帧间余弦相似度: mean={mean_cos:.4f}, std={std_cos:.4f}, "
            f"range=[{min_cos:.4f}, {max_cos:.4f}]"
        )
        if mean_cos > 0.3:
            logger.info(
                f"  [CONFLICT DIAG] ⚠️ 高帧间相关性 (mean_cos={mean_cos:.4f} > 0.3) "
                f"→ SVD 提取了强方向性运动，如与 prompt 方向不一致则冲突严重"
            )
        elif mean_cos > 0.1:
            logger.info(
                f"  [CONFLICT DIAG] ℹ️ 中等帧间相关性 (mean_cos={mean_cos:.4f}) "
                f"→ SVD 有一定方向偏好"
            )
        else:
            logger.info(
                f"  [CONFLICT DIAG] ✓ 低帧间相关性 (mean_cos={mean_cos:.4f} ≤ 0.1) "
                f"→ SVD 噪声接近随机，冲突风险低"
            )

        # ── 4. 帧间差分方向一致性 (Motion Direction Consistency) ──
        # Δf = frame[f+1] - frame[f], 检查所有 Δf 是否指向同一方向
        if num_frames >= 3:
            deltas = []
            for f in range(num_frames - 1):
                delta = eta[0, :, f + 1, :, :] - eta[0, :, f, :, :]
                deltas.append(delta.flatten())

            # 计算相邻 delta 之间的余弦相似度
            delta_cos_sims = []
            for i in range(len(deltas) - 1):
                cos = torch.nn.functional.cosine_similarity(
                    deltas[i].unsqueeze(0), deltas[i + 1].unsqueeze(0)
                ).item()
                delta_cos_sims.append(cos)

            mean_delta_cos = sum(delta_cos_sims) / len(delta_cos_sims)
            logger.info(
                f"  [CONFLICT DIAG] 帧间差分方向一致性: mean_Δcos={mean_delta_cos:.4f} "
                f"(>0.5=强方向运动, <0.1=无规律)"
            )

            # 第一个 delta 与最后一个 delta 的方向
            if len(deltas) >= 2:
                first_last_cos = torch.nn.functional.cosine_similarity(
                    deltas[0].unsqueeze(0), deltas[-1].unsqueeze(0)
                ).item()
                logger.info(
                    f"  [CONFLICT DIAG] 首末帧差分方向相关: cos(Δ_first, Δ_last)={first_last_cos:.4f} "
                    f"(高值=持续单向运动)"
                )

        # ── 5. SVD 运动能量的空间分布 ──
        # 哪些空间区域 SVD 有最大影响？
        spatial_energy = eta[0].pow(2).sum(dim=0).sum(dim=0)  # [H, W]
        h, w = spatial_energy.shape
        # 四象限能量分布
        top_left = spatial_energy[:h // 2, :w // 2].sum().item()
        top_right = spatial_energy[:h // 2, w // 2:].sum().item()
        bot_left = spatial_energy[h // 2:, :w // 2].sum().item()
        bot_right = spatial_energy[h // 2:, w // 2:].sum().item()
        total = top_left + top_right + bot_left + bot_right + 1e-8

        logger.info(
            f"  [CONFLICT DIAG] SVD 空间能量分布 (%):"
            f" TL={top_left / total * 100:.1f}%"
            f" TR={top_right / total * 100:.1f}%"
            f" BL={bot_left / total * 100:.1f}%"
            f" BR={bot_right / total * 100:.1f}%"
        )

        # 偏移方向判断
        left_energy = (top_left + bot_left) / total
        right_energy = (top_right + bot_right) / total
        top_energy = (top_left + top_right) / total
        bot_energy = (bot_left + bot_right) / total

        svd_direction_hints = []
        if left_energy > 0.55:
            svd_direction_hints.append("SVD 能量偏左")
        elif right_energy > 0.55:
            svd_direction_hints.append("SVD 能量偏右")
        if top_energy > 0.55:
            svd_direction_hints.append("SVD 能量偏上")
        elif bot_energy > 0.55:
            svd_direction_hints.append("SVD 能量偏下")

        if svd_direction_hints:
            logger.info(
                f"  [CONFLICT DIAG] SVD 方向偏好: {', '.join(svd_direction_hints)}"
            )
            # 检查是否与 prompt 中的方向关键词冲突
            if found_keywords.get("direction"):
                prompt_dirs = found_keywords["direction"]
                logger.info(
                    f"  [CONFLICT DIAG] ⚠️ Prompt 方向: {prompt_dirs} vs SVD: {svd_direction_hints}"
                    f" → 请人工判断是否冲突"
                )
        else:
            logger.info(
                f"  [CONFLICT DIAG] SVD 能量空间均匀分布，无明显方向偏好"
            )

        # ── 6. 关键结论 ──
        conflict_risk = "LOW"
        reasons = []

        if found_keywords and mean_cos > 0.1:
            conflict_risk = "MEDIUM"
            reasons.append("prompt 有运动词 + SVD 有方向性")
        if found_keywords and mean_cos > 0.3:
            conflict_risk = "HIGH"
            reasons.append("prompt 有运动词 + SVD 方向性强")
        if found_keywords.get("direction") and svd_direction_hints:
            conflict_risk = "HIGH"
            reasons.append("prompt 和 SVD 都有方向偏好")

        logger.info(
            f"  [CONFLICT DIAG] ★ 冲突风险评估: {conflict_risk}"
        )
        if reasons:
            logger.info(f"  [CONFLICT DIAG]   原因: {'; '.join(reasons)}")
        logger.info(
            f"  [CONFLICT DIAG] 建议: 若 CLIP 下降且风险为 HIGH，"
            f"考虑: (1) 降低 α 至 0.0005; (2) 对该样本禁用 SVD; "
            f"(3) 使用简化 prompt 无运动描述"
        )
        logger.info("  ═══════════════════════════════════════════════")
