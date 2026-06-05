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

    # ── 迭代优化参数 ──
    i_max: int = 10               # 迭代轮数

    # ── VLM ──
    vlm_provider: str = "local"
    vlm_model_path: str = "models/Qwen2.5-VL-7B-Instruct"

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

        # ── Step 3: 计算噪声先验 (如果启用) ──
        eta_temporal = None
        eta_inv_raw = None  # 未经 SVD 的原始反演噪声，供 velocity matching 使用
        z0_cached = None    # 缓存 VAE 编码结果，避免重复编码
        prompt_embeds_cached = None  # 缓存 prompt embedding
        if cfg.use_inversion:
            eta_temporal, eta_inv_raw, z0_cached, prompt_embeds_cached = self._compute_noise_prior(ref_video, caption)

        # ── Step 3.5: Velocity Field Matching — 计算 Δe (如果启用) ──
        delta_e = None
        if cfg.use_velocity and cfg.use_inversion and eta_inv_raw is not None:
            delta_e = self._compute_delta_e(
                ref_video, caption, eta_inv_raw,
                z0=z0_cached, e0=prompt_embeds_cached,
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

            # 生成视频（如果有 Δe，通过 embedding hook 注入）
            if delta_e is not None:
                gen_video = self._generate_with_embedding_hook(
                    current_prompt, latents, generator, delta_e, cfg.embed_strength
                )
            else:
                gen_video = self._generate(current_prompt, latents, generator)
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
    ) -> torch.Tensor:
        """调用 Wan 2.1-1.3B 生成视频。"""
        cfg = self.config
        kwargs = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
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
            video = self._generate(prompt, latents, generator)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        logger.info(
            f"    [Velocity] Hook applied: {hook_applied[0]}, "
            f"injection strength={strength:.6f}"
        )
        return video

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
