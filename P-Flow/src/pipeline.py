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
            if self.freq_reshape:
                flags.append(f"blend(α={self.alpha})+freq_reshape(β={self.freq_reshape_beta})")
            else:
                flags.append(f"blend(α={self.alpha})")
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
        if cfg.use_inversion:
            eta_temporal, eta_inv_raw, ref_latents_enc, prompt_embeds_for_diag = \
                self._compute_noise_prior(ref_video, caption)

        # ── Step 3.5: Prompt-Noise 方向冲突诊断 ──
        if eta_temporal is not None and prompt_embeds_for_diag is not None:
            self._diagnose_prompt_noise_conflict(
                eta_temporal, prompt_embeds_for_diag, caption, generator
            )

        # ── Step 4: 生成循环 ──
        num_iters = cfg.i_max if cfg.use_iter else 1
        current_prompt = caption
        prev_video = None
        results = []

        for i in range(1, num_iters + 1):
            logger.info(f"  iter {i}/{num_iters}: {current_prompt[:60]}...")

            # 获取噪声
            latents = self._get_latents(eta_temporal, generator)

            # 生成视频
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
