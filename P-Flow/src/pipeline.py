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
from .svd_filter import SVDFilter
from .velocity_matching import VelocityMatcher
from .attn_inject import AttnInjector, AttnInjectConfig, AttentionKVCache
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
    use_velocity: bool = False     # Velocity Field Matching (Layer 2, Δe embedding)
    use_iter: bool = False         # Iterative VLM Optimization
    use_midpoint: bool = False     # Midpoint ODE Solver
    use_composite: bool = False    # Vertical Composite for VLM
    # ── Noise Prior 参数 ──
    alpha: float = 0.001           # 混合权重 (√α·η_temporal + √(1-α)·η_random)
    rho_s: float = 0.1            # 空间SVD阈值 (去内容)
    rho_m: float = 0.9            # 时间SVD阈值 (保运动)
    inversion_steps: int = 50     # 反演ODE步数
    use_fast_svd: bool = True     # 使用 randomized SVD 加速滤波 (对大 latent 快 2-3x)

    # ── Velocity Matching 参数 ──
    velocity_steps: int = 30      # Δe 优化步数 (轻量版, VMAD用100)
    velocity_lr: float = 1e-3     # Δe 优化学习率
    velocity_T_m: float = 1.0     # 时间步范围 (1.0=复现, 0.3=运动迁移)
    velocity_K: int = 4           # 每步采样的时间步数 (stratified, 降低梯度方差)
    velocity_motion_weight: float = 1.0  # 运动区域加权强度 (0=关闭, 1=全开)
    embed_strength: float = 0.005 # Δe 注入强度 (验证最优: 0.005)

    # ── Attention Injection 参数 (Layer 4) ──
    use_attn_inject: bool = False          # 启用 Self-Attention K/V 注入
    attn_inject_gamma: float = 0.3         # 注入强度 γ (0=不注入, 1=完全替换)
    attn_inject_blocks: str = "all"        # 注入的 block 范围: "all", "first_half", "last_half", 或 "0,5,10,15,20,25"
    attn_inject_block_schedule: str = "uniform"  # block维度γ调度: uniform/front_heavy/back_heavy
    attn_inject_timestep_schedule: str = "linear_decay"  # 时间维度γ调度: constant/linear_decay/cosine_decay

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
            flags.append("blend")
        if self.use_velocity:
            flags.append("velocity")
        if self.use_attn_inject:
            flags.append(f"attn_inject(γ={self.attn_inject_gamma})")
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
        self._attn_injector: Optional['AttnInjector'] = None

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

        注意: 不再使用 @torch.no_grad() 装饰器，因为 velocity matching 需要梯度。
        各不需要梯度的步骤（inversion, generation）内部自己管理 no_grad 上下文。

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
        eta_inv_raw = None  # 未经 SVD 的原始反演噪声，供 velocity matching 使用
        z0_cached = None    # 缓存 VAE 编码结果，避免重复编码
        prompt_embeds_cached = None  # 缓存 prompt embedding
        if cfg.use_inversion:
            eta_temporal, eta_inv_raw, z0_cached, prompt_embeds_cached = self._compute_noise_prior(ref_video, caption)

        # ── Step 3.2: Attention KV Caching (Layer 4, 如果启用) ──
        if cfg.use_attn_inject and cfg.use_inversion:
            self._setup_attn_injection(ref_video, caption, prompt_embeds_cached)

        # ── Step 3.5: Velocity Field Matching — 计算 Δe (如果启用) ──
        delta_e = None
        if cfg.use_velocity and cfg.use_inversion and eta_inv_raw is not None:
            delta_e = self._compute_delta_e(
                ref_video, caption, eta_inv_raw,
                z0=z0_cached, e0=prompt_embeds_cached,
            )

        # ── Step 4: 生成循环 ──
        # Note: if attn_inject is active, _generate_with_attn_inject is used instead of _generate
        num_iters = cfg.i_max if cfg.use_iter else 1
        current_prompt = caption
        prev_video = None
        results = []

        for i in range(1, num_iters + 1):
            logger.info(f"  iter {i}/{num_iters}: {current_prompt[:60]}...")

            # 获取噪声
            latents = self._get_latents(eta_temporal, generator)

            # 生成视频（根据启用的改动点选择生成方式）
            if cfg.use_attn_inject and self._attn_injector is not None:
                gen_video = self._generate_with_attn_inject(
                    current_prompt, latents, generator, delta_e, cfg.embed_strength,
                    negative_prompt=neg_prompt,
                )
            elif delta_e is not None:
                gen_video = self._generate_with_embedding_hook(
                    current_prompt, latents, generator, delta_e, cfg.embed_strength,
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

        # ── Step 4.5: 清理 Attention Injector ──
        if self._attn_injector is not None:
            mem_mb = self._attn_injector.cache.memory_usage_mb()
            logger.info(f"  [AttnInject] Clearing cache ({mem_mb:.1f} MB)")
            self._attn_injector.clear()
            self._attn_injector = None

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
        return metadata

    # ─────────────────────────────────────────────────────────────
    # 内部方法：各改动点的实现
    # ─────────────────────────────────────────────────────────────

    def _compute_noise_prior(
        self, ref_video: torch.Tensor, prompt: str
    ) -> tuple:
        """
        改动点: Inversion + SVD → η_temporal

        流程: V_ref → VAE encode → Flow Inversion → (SVD filter) → η_temporal

        Returns:
            (eta_temporal, eta_inv_raw, z0, prompt_embeds):
            SVD滤波后的噪声, 原始反演噪声, VAE编码latent(缓存), prompt embedding(缓存)
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

        # 保留原始反演噪声 (velocity matching 需要)
        eta_inv_raw = eta_inv

        # SVD Filtering (如果启用)
        if self.config.use_svd:
            logger.info(f"  [SVD] ρ_s={self.config.rho_s}, ρ_m={self.config.rho_m}, fast={self.config.use_fast_svd}")
            svd_filter = SVDFilter(
                rho_s=self.config.rho_s, rho_m=self.config.rho_m
            )
            if self.config.use_fast_svd:
                eta_temporal = svd_filter.filter_efficient(eta_inv)
            else:
                eta_temporal = svd_filter.filter(eta_inv)
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
        改动点: Noise Blending

        η = √α · η_temporal + √(1-α) · η_random
        """
        if eta_temporal is None or not self.config.use_blend:
            return None  # 让 diffusers 自己采样随机噪声

        eta_random = torch.randn(
            eta_temporal.shape,
            dtype=eta_temporal.dtype,
            device=eta_temporal.device,
            generator=generator,
        )

        alpha = self.config.alpha
        eta = (
            torch.sqrt(torch.tensor(alpha, device=self.device)) * eta_temporal
            + torch.sqrt(torch.tensor(1.0 - alpha, device=self.device)) * eta_random
        )
        return eta

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

    def _compute_delta_e(
        self, ref_video: torch.Tensor, caption: str, eta_inv: torch.Tensor,
        z0: torch.Tensor = None, e0: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        改动点: Velocity Field Matching → Δe (v2)

        使用增强版 velocity matching 计算 embedding 残差：
        - 分层多时间步采样 (K=4) 降低梯度方差
        - Padding mask 集中优化有语义的 token 位置
        - 运动区域加权 loss (LTD-inspired)

        Args:
            ref_video: 参考视频张量
            caption: 当前 caption
            eta_inv: 原始反演噪声 (未经SVD滤波)
            z0: 缓存的 VAE 编码 latent (避免重复编码)
            e0: 缓存的 prompt embedding (避免重复编码)

        Returns:
            delta_e: embedding 残差 (B, L, D)
        """
        logger.info("  [Velocity] Computing Δe via lightweight velocity matching...")
        cfg = self.config

        # Reuse cached z0 or encode (避免重复 VAE 编码)
        if z0 is None:
            ref_norm = normalize_video(ref_video).unsqueeze(0)
            z0 = encode_video_to_latents(self.pipe, ref_norm, self.device)
        else:
            logger.info("    [Velocity] Reusing cached z0 (skip VAE encode)")

        # Reuse cached e0 or encode (避免重复 T5 forward)
        if e0 is None:
            e0 = self._encode_prompt(caption)
        else:
            logger.info("    [Velocity] Reusing cached prompt_embeds (skip T5 encode)")

        # Get actual token length (excluding padding) for gradient masking
        token_length = self._get_token_length(caption)

        # Run velocity matching optimization (v2)
        matcher = VelocityMatcher(
            pipe=self.pipe,
            T_m=cfg.velocity_T_m,
            num_opt_steps=cfg.velocity_steps,
            lr=cfg.velocity_lr,
            num_timesteps_per_step=cfg.velocity_K,
            motion_weight_strength=cfg.velocity_motion_weight,
            device=self.device,
        )

        result = matcher.optimize(z0=z0, e0=e0, eta_inv=eta_inv, token_length=token_length)
        delta_e = result["delta_e"]

        logger.info(
            f"  [Velocity] Done: ||Δe||={delta_e.norm().item():.4f}, "
            f"final_loss={result['final_loss']:.6f}"
        )
        return delta_e

    def _get_token_length(self, caption: str, max_sequence_length: int = 512) -> int:
        """
        获取 caption 经 tokenizer 编码后的有效 token 长度 (不含 padding)。

        用于 velocity matching 的 padding mask，确保 Δe 只在有语义的位置优化。
        """
        tokenizer = self.pipe.tokenizer
        inputs = tokenizer(
            caption, padding="max_length",
            max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        # attention_mask: 1 for real tokens, 0 for padding
        token_length = inputs.attention_mask.sum().item()
        return int(token_length)

    def _generate_with_embedding_hook(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        delta_e: torch.Tensor,
        strength: float,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """
        改动点: 通过 text encoder hook 注入 Δe 生成视频。

        保留 pipeline 的正常 prompt 处理路径 (CFG, negative prompt, attention masks)，
        仅在 text encoder 输出上叠加一个微小的 Δe 扰动。

        公式: e_final = e_original + strength * delta_e
        strength ≈ 0.005 时 ||injection|| ≈ 0.18 vs ||e0|| ≈ 1448 (约0.01%扰动)
        """
        # Install hook on text encoder
        text_encoder = self.pipe.text_encoder
        hook_handle = None
        hook_applied = [False]

        def text_encoder_hook(module, input, output):
            """Add delta_e to text encoder output (positive prompt only)."""
            if hook_applied[0]:
                return output  # Only apply once (positive prompt, not negative)

            if isinstance(output, tuple):
                hidden_states = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output

            # Align delta_e shape with hidden_states
            de = delta_e.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if de.shape[1] != hidden_states.shape[1]:
                min_len = min(de.shape[1], hidden_states.shape[1])
                de_aligned = torch.zeros_like(hidden_states)
                de_aligned[:, :min_len, :] = de[:, :min_len, :]
                de = de_aligned

            # Inject: hidden_states += strength * delta_e
            hidden_states = hidden_states + strength * de
            hook_applied[0] = True

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            elif hasattr(output, "last_hidden_state"):
                output.last_hidden_state = hidden_states
                return output
            else:
                return hidden_states

        # Register hook
        hook_handle = text_encoder.register_forward_hook(text_encoder_hook)

        try:
            video = self._generate(prompt, latents, generator,
                                   negative_prompt=negative_prompt)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        logger.info(
            f"    [Velocity] Hook applied: {hook_applied[0]}, "
            f"injection strength={strength:.6f}"
        )
        return video

    def _setup_attn_injection(
        self, ref_video: torch.Tensor, caption: str,
        prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """
        改动点: Layer 4 — Self-Attention K/V Injection (Setup Phase).

        在 inversion 路径上进行一次额外的 forward pass，缓存每个 step 每个 block
        的 attn1 输入（用于后续注入时重算参考 K/V）。

        流程:
            1. 解析 block 配置 → 确定哪些 block 需要注入
            2. 创建 AttnInjector 并安装 caching hooks
            3. 沿 inversion timestep 重新跑一次 transformer forward（不需要完整 ODE，
               只在 generation 会用到的那些 timestep 上跑单步 forward 来缓存 attn 输入）

        注意: 这个方法在 _compute_noise_prior 之后调用，此时 inversion trajectory 已知。
        """
        cfg = self.config
        logger.info(
            f"  [AttnInject] Setting up: γ={cfg.attn_inject_gamma}, "
            f"blocks={cfg.attn_inject_blocks}, "
            f"schedule={cfg.attn_inject_block_schedule}/{cfg.attn_inject_timestep_schedule}"
        )

        # Parse block selection
        inject_blocks = self._parse_inject_blocks(cfg.attn_inject_blocks)

        # Create config
        attn_cfg = AttnInjectConfig(
            gamma=cfg.attn_inject_gamma,
            block_schedule=cfg.attn_inject_block_schedule,
            timestep_schedule=cfg.attn_inject_timestep_schedule,
            inject_blocks=inject_blocks,
            inject_v=True,
        )

        # Create injector
        transformer = self.pipe.transformer
        self._attn_injector = AttnInjector(transformer, attn_cfg)

        # Encode reference video latents
        ref_norm = normalize_video(ref_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_norm, self.device)

        # Get prompt embeddings
        if prompt_embeds is None:
            prompt_embeds = self._encode_prompt(caption)

        # Run caching: we simulate the generation timestep schedule and at each step,
        # compute the corresponding x_t on the inversion trajectory, then forward through
        # the transformer with caching hooks active.
        num_steps = cfg.num_inference_steps
        # Generation goes from t=1 (noise) to t=0 (data)
        # For flow matching: x_t = (1-t)*noise + t*data
        # At generation step i, t = 1 - i/N (decreasing from 1 to 0)

        # We use the same inversion result: given ref_latents (z₁=data) and eta_inv (z₀=noise)
        # x_t = (1-t)*eta_inv + t*ref_latents  →  this is the reference trajectory point at time t

        # Get inverted noise (re-use from prior step via a lightweight re-inversion or from cache)
        # Since _compute_noise_prior was already called, the eta_inv is the noise we got.
        # We need to reconstruct it. The simplest approach: use the same inversion logic
        # but only cache the intermediate trajectory. For efficiency, we'll just recompute x_t
        # analytically (flow matching defines x_t = (1-t)ε + t*z₁)

        logger.info(f"  [AttnInject] Caching attn1 inputs for {num_steps} generation steps...")

        # Re-run inversion to get eta_inv (or reuse if available)
        # Since _compute_noise_prior already ran, we have the prompt_embeds.
        # We perform a lightweight inversion just to get eta_inv (without SVD filtering)
        inverter = FlowMatchingInverter(
            pipe=self.pipe,
            num_inversion_steps=cfg.inversion_steps,
            guidance_scale=1.0,
            device=self.device,
        )
        with torch.no_grad():
            if cfg.use_midpoint:
                eta_inv = inverter.invert_midpoint(ref_latents, prompt_embeds, prompt_embeds)
            else:
                eta_inv = inverter.invert(ref_latents, prompt_embeds, prompt_embeds)

        # Now cache: for each generation timestep, construct x_t and forward through transformer
        with torch.no_grad():
            for step_idx in range(num_steps):
                # t goes from 1 to 0 during generation (same as diffusers scheduler)
                t = 1.0 - step_idx / num_steps
                t_tensor = torch.full(
                    (ref_latents.shape[0],), t * 1000.0,
                    device=self.device, dtype=ref_latents.dtype
                )

                # Reference trajectory: x_t = (1-t)*η_inv + t*z₁
                x_t_ref = (1.0 - t) * eta_inv + t * ref_latents

                # Install caching hooks
                self._attn_injector.start_caching(step_idx=step_idx)

                # Forward pass (just to trigger hooks, output discarded)
                _ = self.pipe.transformer(
                    hidden_states=x_t_ref,
                    timestep=t_tensor,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )

                self._attn_injector.stop_caching()

        mem_mb = self._attn_injector.cache.memory_usage_mb()
        logger.info(
            f"  [AttnInject] Caching complete: {num_steps} steps × "
            f"{len(inject_blocks) if inject_blocks else 'all'} blocks, "
            f"memory={mem_mb:.1f} MB"
        )

    def _parse_inject_blocks(self, blocks_str: str) -> Optional[List[int]]:
        """Parse block selection string into list of indices."""
        num_blocks = 30  # Wan2.1-1.3B has 30 blocks

        if blocks_str == "all":
            return None  # None means all blocks
        elif blocks_str == "first_half":
            return list(range(num_blocks // 2))
        elif blocks_str == "last_half":
            return list(range(num_blocks // 2, num_blocks))
        else:
            # Parse comma-separated indices: "0,5,10,15,20,25"
            try:
                indices = [int(x.strip()) for x in blocks_str.split(",")]
                return [i for i in indices if 0 <= i < num_blocks]
            except ValueError:
                logger.warning(f"  [AttnInject] Invalid block spec '{blocks_str}', using all")
                return None

    @torch.no_grad()
    def _generate_with_attn_inject(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        delta_e: Optional[torch.Tensor] = None,
        embed_strength: float = 0.005,
        negative_prompt: str = "",
    ) -> torch.Tensor:
        """
        改动点: Layer 4 — 带 Self-Attention K/V Injection 的生成。

        与标准 _generate 相同，但在每个 denoising step 中通过 hook 注入参考 K/V。
        可以与 Layer 3 (Δe embedding hook) 同时使用。

        Args:
            prompt: Generation prompt.
            latents: Initial noise (may include noise prior from L2).
            generator: RNG.
            delta_e: Optional Δe from velocity matching (Layer 3).
            embed_strength: Δe injection strength (only used if delta_e is not None).
        """
        cfg = self.config
        injector = self._attn_injector
        num_steps = cfg.num_inference_steps

        # If we also have delta_e, install the text encoder hook
        text_encoder_hook_handle = None
        if delta_e is not None:
            text_encoder_hook_handle = self._install_embedding_hook(delta_e, embed_strength)

        # We need to manually run the denoising loop to insert injection hooks at each step.
        # Use the pipe's scheduler and transformer directly.

        # --- Encode prompt ---
        prompt_embeds = self._encode_prompt(prompt)
        neg_text = negative_prompt or NEGATIVE_PROMPT
        negative_prompt_embeds = self._encode_prompt(neg_text)

        # --- Prepare latents ---
        if latents is None:
            # Generate random latents (same shape as what the pipe would generate)
            # Shape: (1, C, F_latent, H_latent, W_latent)
            # Wan2.1: C=16, F_latent=(num_frames-1)/4+1=21, H_latent=H/8=60, W_latent=W/8=104
            latent_channels = self.pipe.transformer.config.in_channels
            f_latent = (cfg.num_frames - 1) // 4 + 1
            h_latent = cfg.height // 8
            w_latent = cfg.width // 8
            latents = torch.randn(
                (1, latent_channels, f_latent, h_latent, w_latent),
                device=self.device, dtype=self.dtype, generator=generator,
            )
        else:
            latents = latents.to(device=self.device, dtype=self.dtype)

        # --- Prepare scheduler ---
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps  # e.g., [999, 966, 933, ..., 0]

        # Scale latents by scheduler's init sigma if required
        latents = latents * scheduler.init_noise_sigma

        # --- Denoising loop with injection ---
        for step_idx, t in enumerate(timesteps):
            # Compute timestep ratio for schedule (1.0 at start, 0.0 at end)
            timestep_ratio = 1.0 - step_idx / max(num_steps - 1, 1)

            # Start injecting for this step
            injector.start_injecting(step_idx=step_idx, timestep_ratio=timestep_ratio)

            # Expand latents for CFG
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Expand timestep
            t_expand = t.expand(latent_model_input.shape[0])

            # Combine embeddings for CFG
            encoder_hidden_states = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )

            # Model forward (hooks will inject KV)
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t_expand,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            # Stop injecting
            injector.stop_injecting()

            # CFG
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + cfg.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # --- Remove embedding hook if installed ---
        if text_encoder_hook_handle is not None:
            text_encoder_hook_handle.remove()

        # --- Decode latents ---
        from .flow_matching import decode_latents_to_video
        video = decode_latents_to_video(self.pipe, latents.unsqueeze(0) if latents.dim() == 4 else latents, self.device)

        # Format output
        if video.dim() == 5:
            video = video[0]  # Remove batch dim: (C, F, H, W)
        if video.min() < 0:
            video = denormalize_video(video)
        return video.clamp(0, 1)

    def _install_embedding_hook(
        self, delta_e: torch.Tensor, strength: float
    ):
        """Install text encoder hook for Δe injection (reusable helper)."""
        text_encoder = self.pipe.text_encoder
        hook_applied = [False]

        def text_encoder_hook(module, input, output):
            if hook_applied[0]:
                return output
            if isinstance(output, tuple):
                hidden_states = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output

            de = delta_e.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if de.shape[1] != hidden_states.shape[1]:
                min_len = min(de.shape[1], hidden_states.shape[1])
                de_aligned = torch.zeros_like(hidden_states)
                de_aligned[:, :min_len, :] = de[:, :min_len, :]
                de = de_aligned

            hidden_states = hidden_states + strength * de
            hook_applied[0] = True

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            elif hasattr(output, "last_hidden_state"):
                output.last_hidden_state = hidden_states
                return output
            else:
                return hidden_states

        handle = text_encoder.register_forward_hook(text_encoder_hook)
        return handle

    def _encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        """
        编码文本到 embedding。

        Args:
            prompt: 文本 caption
            max_sequence_length: T5 最大序列长度。必须与生成阶段一致（WanPipeline.__call__
                默认 512），否则优化时的 Δe 和注入时的 embedding 空间不匹配。
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
