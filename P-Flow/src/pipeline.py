"""
P-Flow Unified Pipeline.

3层架构：
    L1: Prompt Rewrite (VLM)
    L2: SVD Noise Prior (Inversion + SVD Filtering + Blend)
    L3: Feature Injection (DiT特征空间信息注入)

开关：
    Flag              层级   效果
    ─────────────────────────────────────────────
    --inversion       L2    从参考视频反演噪声
    --svd             L2    SVD空间去内容+时间保运动
    --blend           L2    混合运动噪声与随机噪声
    --feature_inject  L3    DiT特征空间注入
    --iter N          L1    N轮VLM反馈优化prompt
    --midpoint        -     二阶中点法(替代Euler)
    --composite       L1    三面板拼接送VLM对比

组合示例：
    baseline:       无flag → caption + 一次生成
    +noise_prior:   --inversion --svd --blend → L2噪声先验
    +FI:            --inversion --svd --blend --feature_inject → L2+L3
    full pflow:     --inversion --svd --blend --feature_inject --iter 10 --composite
"""

import os
import csv
import json
import time
import shutil
import math
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field

import torch

from .distributed import setup_single_gpu, load_model_single_gpu, cleanup_gpu_memory
from .flow_matching import FlowMatchingInverter, encode_video_to_latents
from .svd_filter import SVDFilter, SVDFilterConfig, compute_temporal_signal_reliability
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
    alpha: float = 0.003           # 混合权重 (√α·η_temporal + √(1-α-β)·η_random)
                                  # P-Flow论文用 0.001, 推荐搜索范围: 0.001 ~ 0.01
    adaptive_alpha: bool = False   # 是否启用 TSR-guided 自适应 α
    alpha_max: float = 0.006       # 自适应 α 的上限 (TSR=1 且 M_d=1 时 α_fusion 可达此值)
    alpha_min: float = 0.0         # 自适应 α 的下限 (TSR=0 时取此值, 0=完全关闭SVD blend)
    tsr_tcr_center: float = 0.1    # TCR sigmoid 中心 (TCR < center → 低可靠性)
    tsr_tcr_slope: float = 10.0    # TCR sigmoid 斜率 (越大越陡峭, 区分越锐利)
    beta: float = 0.0              # 外观分量混合权重 (√β·η_spatial), 0=不使用外观分量
                                  # 推荐范围: 0.0~0.005, 需 α+β < 1.0
                                  # η_spatial 是 SVD Stage 1 去掉的外观/内容分量
                                  # 对"完全复原"场景有用: 低运动视频的运动信号弱时, 外观信号可补充
    rho_s: float = 0.1            # 空间SVD阈值 (去内容)
    rho_m: float = 0.9            # 时间SVD阈值 (保运动)
    inversion_steps: int = 50     # 反演ODE步数
    use_fast_svd: bool = True     # 使用 randomized SVD 加速滤波 (对大 latent 快 2-3x)

    # ── L3: Feature Injection (FI) ──
    # 核心思想: 不做 latent 空间的方向修正 (VDA), 改做 DiT 特征空间的信息注入
    # 反演过程中缓存 DiT 每步的 cross-attention 输出, 生成时以残差方式注入
    # 优势:
    #   1. 特征空间语义对齐比 latent 空间方向对齐更鲁棒
    #   2. 不修改 ODE 积分路径, 只修改 DiT 中间表示
    #   3. 类似 ControlNet 的零训练注入, 但不需要训练
    feature_inject: bool = False          # 是否启用 Feature Injection
    fi_layers: str = "all"                # 注入哪些层: "all" / "mid" / "last" / 逗号分隔的层号
    #   all: 所有 transformer 块 (30 层全注入)
    #   mid: 中间 1/3 层 (layer 10~19, 高语义层)
    #   last: 最后 1/3 层 (layer 20~29, 细节层)
    #   "5,10,15,20": 指定层号
    fi_lambda: float = 0.1               # FI 注入强度 λ (推荐 0.01~0.3)
    # h_injected = h_current + λ * (h_ref - h_current) = (1-λ)*h_current + λ*h_ref
    # λ=0: 无注入, λ=1: 完全替换为参考特征
    fi_schedule: str = "middle_peak"      # λ 调度策略 (同 VDA: middle_peak / warmup_decay / cosine_decay / constant)
    fi_quality_gate: bool = True          # 是否启用质量门控 (基于 mean_cos)
    fi_adaptive_gate: bool = True         # 是否启用特征对齐自适应门控
    fi_adaptive_temp: float = 5.0          # 自适应门控温度 (越大越敏感, 推荐 3~10)
    fi_quality_threshold: float = 0.05    # 质量门控阈值 (mean_cos < threshold → 弱引导)
    fi_md_gate: bool = False              # 是否启用 M_d 对 FI QS 的修正 (消融用, 默认关闭)
    fi_cache_mode: str = "attention"      # 缓存什么特征:
    #   attention: cross-attention 输出 (语义对齐, 推荐)
    #   hidden: 完整 hidden_states (信息丰富但维度大)
    #   mlp: MLP 输出 (更高级语义)

    # ── 迭代优化参数 ──
    i_max: int = 10               # 迭代轮数

    # ── VLM ──
    vlm_provider: str = "local"
    vlm_model_path: str = "models/Qwen2.5-VL-7B-Instruct"

    # ── M_d (Motion Definiteness) 融合门控 ──
    md_file: str = ""                 # M_d 查表 CSV 路径 (由 scripts/compute_md.py 生成)
    alpha_floor: float = 0.004        # M_d 确认物体运动时的保底 α (=旧版固定值)
                                     # α_eff = max(α_floor * max(M_d, alpha_md_floor), α_min + f(M_d,TSR) * (α_max - α_min))
    alpha_md_floor: float = 0.3       # α 保底中 M_d 的下限，防止 M_d=0 时 α 完全归零
                                     # md_for_alpha = max(M_d, alpha_md_floor)
    fi_qs_md_floor: float = 0.5       # Quality Scale × M_d 修正的保底下限
                                     # QS_eff = QS * max(M_d, fi_qs_md_floor)
                                     # 防止 M_d=0.0 时 FI 被完全关闭
    fi_alpha_coupling: bool = True    # FI λ 与 L2 α 协同缩放
                                     # α_eff/α_ref 比例缩放 FI λ_max
                                     # α_ref=0.004 (SVD+FI 基线固定值)
                                     # 当 α_eff 低时, FI 注入同步降低
    fi_alpha_ref: float = 0.004       # α 参考值 (SVD+FI 基线固定 α)

    # ── PNA (Prompt-Noise Alignment) 在线门控 ──
    pna_probe: bool = False            # 启用 PNA 探测 (在线测量 η_temporal 方向是否有利)
                                     # 用模型一步前向比较 mixed vs random 噪声
                                     # 替代 LLM M_d 离线判断，更准确
    pna_probe_step: float = 0.95       # 探测的 t 值 (接近1.0=纯噪声, 影响最大)
    pna_alpha_max: float = 0.006        # PNA 门控的 α 上限
    pna_alpha_min: float = 0.0005       # PNA 门控的 α 下限 (不允许完全为0)

    # ── 其他 ──
    seed: int = 42

    # ── 运行时缓存 (非 CLI 参数) ──
    _md_lookup: Dict[int, float] = field(default_factory=dict, repr=False)

    def load_md_scores(self):
        """从 CSV 加载 M_d 查表到 _md_lookup。"""
        if self._md_lookup or not self.md_file:
            return
        path = Path(self.md_file)
        if not path.exists():
            logger.warning(f"  [M_d] 查表文件不存在: {self.md_file}, M_d 不生效")
            return
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._md_lookup[int(row["sample_id"])] = float(row["md_score"])
        logger.info(f"  [M_d] 已加载 {len(self._md_lookup)} 条评分 from {self.md_file}")

    def get_md(self, sample_id: int) -> float:
        """获取指定样本的 M_d 值，未找到则返回 1.0 (默认信任 TSR)。"""
        if self._md_lookup:
            return self._md_lookup.get(sample_id, 1.0)
        return 1.0  # 无查表时默认 M_d=1.0，退化为纯 TSR 门控

    def active_flags(self) -> List[str]:
        """返回当前启用的改动点列表。"""
        flags = []
        if self.use_inversion:
            flags.append("inversion")
        if self.use_svd:
            flags.append("svd")
        if self.use_blend:
            blend_parts = []
            if self.adaptive_alpha:
                blend_parts.append(f"α=[{self.alpha_min}~{self.alpha_max}], TSR-guided")
            else:
                blend_parts.append(f"α={self.alpha}")
            if self.beta > 0:
                blend_parts.append(f"β={self.beta}")
            flags.append(f"blend({', '.join(blend_parts)})")
        if self.feature_inject:
            fi_desc = f"feature_inject(λ={self.fi_lambda}, layers={self.fi_layers}, sched={self.fi_schedule}, mode={self.fi_cache_mode}"
            if self.fi_adaptive_gate:
                fi_desc += f", adaptive(temp={self.fi_adaptive_temp})"
            fi_desc += ")"
            flags.append(fi_desc)
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

        # ── 加载 M_d 查表 (懒加载) ──
        cfg.load_md_scores()
        md = cfg.get_md(sample_id)
        cfg._current_md = md  # 运行时注入，供 _get_latents / _generate_with_fi 使用
        logger.info(f"  [M_d] sample={sample_id}, M_d={md:.2f}")

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
        prompt_embeds = None
        ref_latents_enc = None
        ref_trajectory_from_inversion = None  # 合并模式：反演时同时缓存的轨迹
        fi_ref_features_from_inversion = None  # 合并模式：反演时同时缓存的FI特征
        if cfg.use_inversion:
            # 判断是否需要在反演时同时缓存轨迹/FI特征
            need_trajectory = cfg.feature_inject
            cache_every_n = 1

            # 构造 FI 配置（如果启用 feature_inject，在反演时同时缓存 DiT 特征）
            fi_config_for_inv = None
            if cfg.feature_inject:
                transformer = self.pipe.transformer
                num_layers = len(transformer.blocks) if hasattr(transformer, 'blocks') else 30
                if cfg.fi_layers == "all":
                    target_layers = list(range(num_layers))
                elif cfg.fi_layers == "early":
                    target_layers = list(range(0, num_layers // 3))
                elif cfg.fi_layers == "mid":
                    target_layers = list(range(num_layers // 3, 2 * num_layers // 3))
                elif cfg.fi_layers in ("last", "late"):
                    target_layers = list(range(2 * num_layers // 3, num_layers))
                else:
                    try:
                        target_layers = [int(x.strip()) for x in cfg.fi_layers.split(",")]
                    except ValueError:
                        target_layers = list(range(num_layers // 3, 2 * num_layers // 3))
                fi_config_for_inv = {
                    "target_layers": target_layers,
                    "cache_mode": cfg.fi_cache_mode,
                    "gen_num_steps": cfg.num_inference_steps,
                    "num_layers": num_layers,
                }

            eta_temporal, eta_inv_raw, ref_latents_enc, prompt_embeds, ref_trajectory_from_inversion, fi_ref_features_from_inversion, svd_stats = \
                self._compute_noise_prior(
                    ref_video, caption,
                    cache_trajectory=need_trajectory and not cfg.use_midpoint,
                    cache_every_n=cache_every_n,
                    fi_config=fi_config_for_inv,
                )

            # ── 缓存 prompt_embeds 供 PNA 探测使用 ──
            if prompt_embeds is not None:
                cfg._current_prompt_embeds = prompt_embeds

        # ── Step 3.5: 轨迹/FI特征缓存 ──
        ref_trajectory = None
        fi_ref_features = None
        if cfg.feature_inject:
            if not cfg.use_inversion:
                logger.warning(
                    "  [FI] feature_inject=True 但 use_inversion=False, "
                    "自动启用 inversion 以获取参考特征"
                )

            # 优先复用反演时已缓存的轨迹（合并模式）
            if ref_trajectory_from_inversion is not None:
                ref_trajectory = ref_trajectory_from_inversion
                logger.info(
                    f"  [FI] ✅ 复用反演缓存的轨迹 "
                    f"({len(ref_trajectory)} points), 无需二次反演"
                )
            elif ref_latents_enc is not None and prompt_embeds is not None:
                # 合并模式未启用（如 midpoint 模式），需要单独做反演
                ref_lat = ref_latents_enc
                p_emb = prompt_embeds
                traj_inverter = FlowMatchingInverter(
                    pipe=self.pipe,
                    num_inversion_steps=cfg.inversion_steps,
                    guidance_scale=1.0,
                    device=self.device,
                )
                _, ref_trajectory, _ = traj_inverter.invert_with_trajectory(
                    ref_lat, p_emb, p_emb,
                    cache_every_n=1,
                )
            else:
                # 没做过 inversion，现在做一次 encode + embed + 反演
                ref_norm = normalize_video(ref_video).unsqueeze(0)
                ref_lat = encode_video_to_latents(self.pipe, ref_norm, self.device)
                p_emb = self._encode_prompt(caption)
                traj_inverter = FlowMatchingInverter(
                    pipe=self.pipe,
                    num_inversion_steps=cfg.inversion_steps,
                    guidance_scale=1.0,
                    device=self.device,
                )
                _, ref_trajectory, _ = traj_inverter.invert_with_trajectory(
                    ref_lat, p_emb, p_emb,
                    cache_every_n=1,
                )

            # ── Feature Injection: 优先复用反演时 inline 缓存的特征 ──
            if fi_ref_features_from_inversion is not None and len(fi_ref_features_from_inversion) > 1:
                fi_ref_features = fi_ref_features_from_inversion
                logger.info(
                    f"  [FI] ✅ 复用反演时 inline 缓存的特征 "
                    f"({len([k for k in fi_ref_features if k != '_meta'])} steps)"
                )
            elif ref_trajectory is not None:
                # Fallback: 用反演轨迹事后缓存特征 (多耗时 ~77s)
                logger.info(
                    f"  [FI] inline 缓存不可用，使用反演轨迹事后缓存特征..."
                )
                fi_ref_features = self._cache_fi_ref_features(
                    ref_trajectory, prompt_embeds or self._encode_prompt(caption)
                )
            else:
                logger.warning(
                    "  [FI] ⚠️ feature_inject=True 但无反演轨迹，FI 将不生效"
                )


        # ── Step 4: 生成循环 ──
        num_iters = cfg.i_max if cfg.use_iter else 1
        current_prompt = caption
        prev_video = None
        results = []

        # ── 诊断: 噪声决策状态总结 ──
        _diag_fi = cfg.feature_inject and fi_ref_features is not None
        _diag_svd_blend = cfg.use_blend and cfg.use_svd
        _diag_eta_available = eta_temporal is not None
        logger.info(
            f"  [Noise Decision Summary] "
            f"feature_inject={'ACTIVE' if _diag_fi else 'OFF'}, "
            f"svd_blend={'ENABLED' if _diag_svd_blend else 'DISABLED'}, "
            f"eta_temporal={'AVAILABLE' if _diag_eta_available else 'NONE'}, "
            f"use_blend={cfg.use_blend}"
        )

        for i in range(1, num_iters + 1):
            logger.info(f"  iter {i}/{num_iters}: {current_prompt[:60]}...")

            # 获取噪声
            latents = self._get_latents(eta_temporal, generator, svd_stats=svd_stats)

            # 生成视频（FI / 标准）
            if cfg.feature_inject and fi_ref_features is not None:
                gen_video = self._generate_with_fi(
                    current_prompt, latents, generator,
                    ref_features=fi_ref_features,
                    eta_temporal=eta_temporal,
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
        self, ref_video: torch.Tensor, prompt: str,
        cache_trajectory: bool = False, cache_every_n: int = 1,
        fi_config: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        L1/L2: Inversion + SVD → η_temporal

        流程: V_ref → VAE encode → Flow Inversion → SVD Filtering → η_temporal

        Args:
            ref_video: 参考视频张量
            prompt: 文本描述
            cache_trajectory: 是否在反演时同时缓存轨迹（用于 FI）
            cache_every_n: 轨迹缓存间隔
            fi_config: Feature Injection 配置 (传入则在反演时同时缓存特征)

        Returns:
            (eta_temporal, eta_inv_raw, ref_latents, prompt_embeds, trajectory, fi_ref_features):
            SVD滤波后的噪声, 原始反演噪声, VAE编码latent, prompt embedding, 轨迹字典, FI特征缓存
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

        trajectory = None
        fi_ref_features = None
        if self.config.use_midpoint:
            logger.info("  [Inversion] midpoint (2nd-order)...")
            eta_inv = inverter.invert_midpoint(
                ref_latents, prompt_embeds, prompt_embeds
            )
        elif cache_trajectory:
            # 合并模式：反演 + 轨迹缓存 + (可选)FI特征缓存 一步完成
            logger.info("  [Inversion] euler (1st-order) + trajectory caching...")
            eta_inv, trajectory, fi_ref_features = inverter.invert_with_trajectory(
                ref_latents, prompt_embeds, prompt_embeds,
                cache_every_n=cache_every_n,
                fi_config=fi_config,
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
        svd_stats = None
        if self.config.use_svd:
            svd_config = SVDFilterConfig(
                rho_s=self.config.rho_s,
                rho_m=self.config.rho_m,
                use_fast_svd=self.config.use_fast_svd,
            )
            svd_filter = SVDFilter(config=svd_config)

            logger.info(
                f"  [SVD] ρ_s={self.config.rho_s}, ρ_m={self.config.rho_m}"
            )

            # 使用 return_stats 获取 S_temporal (避免 TSR 计算时重复 SVD)
            eta_temporal, svd_stats = svd_filter.filter(eta_inv, return_stats=True)

            # 如果启用自适应 α, 计算 TSR
            if self.config.adaptive_alpha and svd_stats is not None:
                tsr_result = compute_temporal_signal_reliability(
                    eta_temporal,
                    S_temporal=svd_stats.get("S_temporal"),
                )
                # 将 TSR 结果存入 svd_stats, 供 _get_latents 使用
                svd_stats["tsr"] = tsr_result
                logger.info(
                    f"  [TSR] TCR={tsr_result['tcr']:.4f}, "
                    f"TAC={tsr_result['tac']:.4f}, "
                    f"TSR={tsr_result['tsr']:.4f}"
                )
        else:
            eta_temporal = eta_inv

        logger.info(
            f"  η_temporal: mean={eta_temporal.mean():.4f}, std={eta_temporal.std():.4f}"
        )
        return eta_temporal, eta_inv_raw, ref_latents, prompt_embeds, trajectory, fi_ref_features, svd_stats

    def _get_latents(
        self,
        eta_temporal: Optional[torch.Tensor],
        generator: torch.Generator,
        svd_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        L2: Noise Prior Blending

        基础混合公式 (β=0):
            η = √α · η_temporal + √(1-α) · η_random

        三路混合公式 (β>0, 启用外观分量):
            η = √α · η_temporal + √β · η_spatial + √(1-α-β) · η_random

        其中:
            η_temporal: SVD Stage 2 提取的运动先验 (去外观保运动)
            η_spatial:  SVD Stage 1 分离的外观/内容分量 (运动迁移时丢弃, 完全复原时有用)
            η_random:   纯随机噪声

        当 adaptive_alpha=True 时, α 由 TSR (Temporal Signal Reliability) 自适应决定:
            α_adaptive = α_min + TSR × (α_max - α_min)
        TSR 越高 → η_temporal 中的运动信号越可靠 → α 越大 → 注入越强
        TSR 越低 → η_temporal 中的运动信号不可靠 → α 越小 → 注入越弱甚至关闭
        """
        if eta_temporal is None or not self.config.use_blend:
            logger.info(
                f"  [_get_latents] 返回 None → diffusers 纯随机噪声 "
                f"(eta_temporal={'None' if eta_temporal is None else 'EXISTS'}, "
                f"use_blend={self.config.use_blend})"
            )
            return None  # 让 diffusers 自己采样随机噪声

        eta_random = torch.randn(
            eta_temporal.shape,
            dtype=eta_temporal.dtype,
            device=eta_temporal.device,
            generator=generator,
        )

        # ── Determine α ──
        cfg = self.config
        if cfg.adaptive_alpha:
            # ── PNA 在线门控 (替代 M_d × TSR) ──
            if getattr(cfg, 'pna_probe', False):
                # 用模型一步前向测量 η_temporal 的方向是否有利
                prompt_embeds = getattr(cfg, '_current_prompt_embeds', None)
                if prompt_embeds is not None:
                    pna_score = self._compute_pna_score(
                        eta_temporal, prompt_embeds, generator
                    )
                    # PNA → α: 直接映射 (现在 _compute_pna_score 内部已计算多方案)
                    # 默认使用方案E (temporal_frame_cos 修正版)
                    pna_alpha_max = getattr(cfg, 'pna_alpha_max', 0.006)
                    pna_alpha_min = getattr(cfg, 'pna_alpha_min', 0.0005)
                    alpha = pna_alpha_min + pna_score * (pna_alpha_max - pna_alpha_min)

                    # 缓存 α_eff 供 FI 协同缩放使用
                    cfg._current_alpha_eff = alpha

                    # 输出 TSR-guided α 对比 (方案D)
                    tsr_info = svd_stats.get("tsr", {}) if svd_stats else {}
                    tsr = tsr_info.get("tsr", 0.0)
                    alpha_tsr = pna_alpha_min + tsr * (pna_alpha_max - pna_alpha_min)

                    logger.info(
                        f"  [PNA-α] 选用方案G(连续平滑修正v5): score={pna_score:.4f}, "
                        f"α_eff={alpha:.6f}"
                    )
                    pna_alphas_dict = getattr(cfg, '_pna_alphas', {})
                    logger.info(
                        f"  [PNA-α 对比] A={pna_alphas_dict.get('A', 0):.6f} | "
                        f"B={pna_alphas_dict.get('B', 0):.6f} | "
                        f"C={pna_alphas_dict.get('C', 0):.6f} | "
                        f"E={pna_alphas_dict.get('E', 0):.6f} | "
                        f"F={pna_alphas_dict.get('F', 0):.6f} | "
                        f"G(选用)={alpha:.6f} | "
                        f"D(TSR)={alpha_tsr:.6f}(TSR={tsr:.4f}) | "
                        f"经验={pna_alphas_dict.get('hint', 0):.6f}"
                    )
                else:
                    # fallback: 用 TSR-guided
                    logger.warning("  [PNA] prompt_embeds 不可用, fallback 到 TSR-guided α")
                    tsr_info = svd_stats.get("tsr", {}) if svd_stats else {}
                    tsr = tsr_info.get("tsr", 0.0)
                    md = getattr(cfg, '_current_md', 1.0)
                    f_md_tsr = md * tsr + (1.0 - md) * 0.1 * tsr
                    alpha_fusion = cfg.alpha_min + f_md_tsr * (cfg.alpha_max - cfg.alpha_min)
                    alpha_md_floor = getattr(cfg, 'alpha_md_floor', 0.3)
                    md_for_alpha = max(md, alpha_md_floor)
                    alpha_floor_val = cfg.alpha_floor * md_for_alpha
                    alpha = max(alpha_floor_val, alpha_fusion)
                    cfg._current_alpha_eff = alpha
                    logger.info(
                        f"  [Adaptive α (fallback)] M_d={md:.2f}, TSR={tsr:.4f}, "
                        f"α_eff={alpha:.6f}"
                    )
            else:
                # TSR-guided adaptive α (with optional M_d correction) — 原路径
                tsr_info = svd_stats.get("tsr", {}) if svd_stats else {}
                tsr = tsr_info.get("tsr", 0.0)

                # ── M_d × TSR 融合 ──
                md = getattr(cfg, '_current_md', 1.0)  # 运行时注入，见 run() 方法
                f_md_tsr = md * tsr + (1.0 - md) * 0.1 * tsr
                alpha_fusion = cfg.alpha_min + f_md_tsr * (cfg.alpha_max - cfg.alpha_min)

                # ── α_floor 保底 (含 alpha_md_floor 下限) ──
                alpha_md_floor = getattr(cfg, 'alpha_md_floor', 0.3)
                md_for_alpha = max(md, alpha_md_floor)  # 防止 M_d=0 时 α 完全归零
                alpha_floor_val = cfg.alpha_floor * md_for_alpha
                alpha = max(alpha_floor_val, alpha_fusion)

                # ── 缓存 α_eff 供 FI 协同缩放使用 ──
                cfg._current_alpha_eff = alpha

                logger.info(
                    f"  [Adaptive α] M_d={md:.2f}, alpha_md_floor={alpha_md_floor}, "
                    f"md_for_alpha={md_for_alpha:.2f}, TSR={tsr:.4f}, "
                    f"f(M_d,TSR)={f_md_tsr:.4f}, "
                    f"α_fusion={alpha_fusion:.6f}, α_floor={alpha_floor_val:.6f} "
                    f"→ α_eff={alpha:.6f} "
                    f"(range: [{cfg.alpha_min}, {cfg.alpha_max}])"
                )
        else:
            # Fixed α (原行为)
            alpha = cfg.alpha

        beta = cfg.beta
        remaining = max(0.0, 1.0 - alpha - beta)

        sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=self.device))
        sqrt_beta = torch.sqrt(torch.tensor(beta, device=self.device))
        sqrt_remaining = torch.sqrt(torch.tensor(remaining, device=self.device))

        # ── Three-way blend ──
        eta = sqrt_alpha * eta_temporal + sqrt_remaining * eta_random

        # 如果 β > 0, 加入外观分量 η_spatial
        if beta > 0:
            eta_spatial = svd_stats.get("eta_spatial") if svd_stats else None
            if eta_spatial is not None:
                # ── 关键: η_spatial 量级匹配 ──
                # η_spatial 的 std (~0.9-1.2) 远大于 η_temporal (~0.28-0.41)
                # 如果直接注入，β=0.001 的实际贡献会远大于 α=0.004
                # 解决: renorm η_spatial 使其 std 与 η_temporal 一致，
                # 这样 β 的语义才与 α 对等（"相同系数=相同贡献"）
                spatial_std = eta_spatial.std()
                temporal_std = eta_temporal.std()
                if spatial_std > 1e-6:
                    eta_spatial_matched = eta_spatial * (temporal_std / spatial_std)
                else:
                    eta_spatial_matched = eta_spatial
                logger.info(
                    f"  [Spatial Blend] β={beta:.4f} (√β={sqrt_beta.item():.4f}), "
                    f"η_spatial std={spatial_std:.4f} → renormed to {eta_spatial_matched.std():.4f} "
                    f"(matched η_temporal std={temporal_std:.4f})"
                )
                eta = eta + sqrt_beta * eta_spatial_matched
            else:
                logger.warning(
                    f"  [Spatial Blend] β={beta:.4f} 但 eta_spatial 不可用 "
                    f"(需要 --svd 开启), 忽略外观分量"
                )

        # ── 诊断: Blend 效果 ──
        # 1. 混合后分布
        logger.info(
            f"  [Blend] α={alpha:.4f} (√α={sqrt_alpha.item():.4f}), "
            f"β={beta:.4f} (√β={sqrt_beta.item():.4f}), "
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

    def _compute_schedule(
        self,
        num_steps: int,
        max_value: float,
        schedule_type: str,
    ) -> List[float]:
        """
        计算每步的调度值 (用于 FI 的 λ 调度等)。

        Args:
            num_steps: 总步数
            max_value: 峰值
            schedule_type: 调度类型 (constant / middle_peak / warmup_decay / cosine_decay)

        Returns:
            List[float]: 长度为 num_steps 的调度值列表
        """
        if schedule_type == "constant":
            values = [max_value] * num_steps
        elif schedule_type == "middle_peak":
            values = [
                max_value * math.sin(math.pi * i / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
        elif schedule_type == "warmup_decay":
            warmup_steps = max(int(0.2 * num_steps), 1)
            values = []
            for i in range(warmup_steps):
                values.append(max_value * (0.5 + 0.5 * i / max(warmup_steps - 1, 1)))
            remaining = num_steps - warmup_steps
            for i in range(remaining):
                values.append(
                    max_value * math.cos(math.pi / 2 * i / max(remaining - 1, 1))
                )
        elif schedule_type == "cosine_decay":
            values = [
                max_value * math.cos(math.pi / 2 * i / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
        else:
            logger.warning(f"  [Schedule] Unknown type '{schedule_type}', using middle_peak")
            return self._compute_schedule(num_steps, max_value, "middle_peak")

        logger.info(
            f"  [Schedule] type={schedule_type}, max={max_value}, steps={num_steps}, "
            f"first 5: {[f'{v:.4f}' for v in values[:5]]}, "
            f"last 5: {[f'{v:.4f}' for v in values[-5:]]}"
        )
        return values

    def _compute_quality_scale(
        self,
        eta_temporal: Optional[torch.Tensor],
    ) -> float:
        """
        计算质量缩放因子 (用于 FI 质量门控)。

        基于 η_temporal 的帧间余弦相似度 (motion coherence)
        计算一个 0~1 的缩放因子:
            - motion_coherence 高 → scale → 1.0 (完整引导)
            - motion_coherence 低 → scale → 0.1 (保留微弱引导)

        Args:
            eta_temporal: SVD 滤波后的噪声

        Returns:
            scale: 0~1 的缩放因子
        """
        if eta_temporal is None:
            return 1.0

        # ── 计算 motion coherence (帧间余弦相似度) ──
        eta_gate = eta_temporal
        if eta_gate.dim() == 5:
            if eta_gate.shape[2] > eta_gate.shape[1]:
                eta_gate = eta_gate.permute(0, 2, 1, 3, 4)
            num_frames_gate = eta_gate.shape[2]
        elif eta_gate.dim() == 4:
            num_frames_gate = eta_gate.shape[1]
            eta_gate = eta_gate.unsqueeze(0)
        else:
            return 1.0

        if num_frames_gate < 2:
            return 1.0

        frame_cos_sims = []
        for f in range(num_frames_gate - 1):
            f1 = eta_gate[0, :, f, :, :].flatten()
            f2 = eta_gate[0, :, f + 1, :, :].flatten()
            cos = torch.nn.functional.cosine_similarity(
                f1.unsqueeze(0), f2.unsqueeze(0)
            ).item()
            frame_cos_sims.append(cos)

        mean_cos = sum(frame_cos_sims) / len(frame_cos_sims)

        # ── 软门控: sigmoid 映射 ──
        min_scale = 0.1
        threshold = self.config.fi_quality_threshold
        k = 20.0
        exp_arg = -k * (mean_cos - threshold)
        exp_arg = max(min(exp_arg, 500.0), -500.0)
        sigmoid_val = 1.0 / (1.0 + math.exp(exp_arg))
        scale = min_scale + (1.0 - min_scale) * sigmoid_val

        logger.info(
            f"  [Quality Scale] mean_cos={mean_cos:.4f}, "
            f"threshold={threshold}, scale={scale:.4f} "
            f"({'strong' if scale > 0.7 else 'weak' if scale > 0.3 else 'minimal'})"
        )

        return scale

    @torch.no_grad()
    def _compute_pna_score(
        self,
        eta_temporal: torch.Tensor,
        prompt_embeds: torch.Tensor,
        generator: torch.Generator,
    ) -> float:
        """
        PNA (Prompt-Noise Alignment): 在线测量 η_temporal 方向对模型是否有利。

        核心思想 (类似 FI 的 Adaptive Gate，但在 L2 层面):
            1. 用混合噪声 (η_mixed) 跑一步模型前向
            2. 用纯随机噪声 (η_random) 跑一步模型前向
            3. 比较两者在 mid 层的特征差异
            4. 差异方向与 η_temporal 的对齐度 → PNA score

        PNA 高 → η_temporal 让模型预测向"正确"方向偏移 → α 可以大
        PNA 低 → η_temporal 让模型预测向"错误"方向偏移 → α 应该小

        Args:
            eta_temporal: SVD 滤波后的噪声
            prompt_embeds: prompt embedding
            generator: 随机数生成器

        Returns:
            pna_score: 0~1 的对齐度分数
        """
        cfg = self.config
        transformer = self.pipe.transformer

        # ── 构造两种噪声 ──
        alpha_test = 0.004  # 试探 α，用基线值
        eta_random = torch.randn(
            eta_temporal.shape,
            dtype=eta_temporal.dtype,
            device=eta_temporal.device,
            generator=generator,
        )

        # 混合噪声
        sqrt_a = math.sqrt(alpha_test)
        sqrt_1ma = math.sqrt(1.0 - alpha_test)
        eta_mixed = sqrt_a * eta_temporal + sqrt_1ma * eta_random

        # ── 确定探测步 ──
        t_probe = cfg.pna_probe_step  # 接近 1.0 = 纯噪声
        t_tensor = torch.full(
            (eta_temporal.shape[0],), t_probe,
            device=self.device, dtype=eta_temporal.dtype
        )

        # ── 用 mid 层 hook 捕获特征 ──
        num_layers = len(transformer.blocks) if hasattr(transformer, 'blocks') else 30
        probe_layer = num_layers // 2  # mid 层

        captured = {}

        def probe_hook(module, input, output):
            if isinstance(output, tuple):
                captured['feat'] = output[0].detach()
            else:
                captured['feat'] = output.detach()

        handle = transformer.blocks[probe_layer].register_forward_hook(probe_hook)

        try:
            # ── 前向1: 混合噪声 ──
            captured.clear()
            _ = transformer(
                hidden_states=eta_mixed,
                timestep=t_tensor,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            feat_mixed = captured.get('feat')
            if feat_mixed is None:
                logger.warning("  [PNA] 无法捕获混合噪声特征, 跳过 PNA")
                return 0.5

            # ── 前向2: 纯随机噪声 ──
            captured.clear()
            _ = transformer(
                hidden_states=eta_random,
                timestep=t_tensor,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            feat_random = captured.get('feat')
            if feat_random is None:
                logger.warning("  [PNA] 无法捕获随机噪声特征, 跳过 PNA")
                return 0.5

        finally:
            handle.remove()

        # ── 计算特征差异 ──
        delta_feat = feat_mixed - feat_random  # η_temporal 带来的特征偏移

        # ── 指标1: 差异范数归一化 (η_temporal 对模型的影响强度) ──
        delta_norm = delta_feat.norm().item()
        feat_norm = feat_random.norm().item()
        relative_impact = delta_norm / max(feat_norm, 1e-8)

        # ── 指标2: 差异方向的一致性 (各帧/各空间位置的偏移方向是否一致) ──
        # 如果 η_temporal 有利 → 模型各处偏移方向一致 → cos_sim 高
        # 如果 η_temporal 有害/噪声 → 偏移方向混乱 → cos_sim 低
        # 计算: 把特征展平为 (num_elements,) 的向量，然后对半分两段算 cos
        delta_flat = delta_feat.flatten()
        n = delta_flat.shape[0]
        half = n // 2
        if half > 0:
            cos_consistency = torch.nn.functional.cosine_similarity(
                delta_flat[:half].unsqueeze(0),
                delta_flat[half:2*half].unsqueeze(0)
            ).item()
        else:
            cos_consistency = 0.0

        # ── 指标3: 帧间差异的一致性 (temporal coherence of impact) ──
        # 将 delta_feat 按帧拆分，计算相邻帧特征偏移方向的余弦相似度
        # 运动型视频 → 各帧偏移方向一致 → 帧间cos高
        # 静态/噪声型视频 → 各帧偏移方向随机 → 帧间cos低
        frame_consistency = 0.0
        if delta_feat.dim() == 5 and delta_feat.shape[2] >= 2:
            # shape: [1, C, F, H, W]
            nf = delta_feat.shape[2]
            frame_cos_list = []
            for f in range(nf - 1):
                f1 = delta_feat[0, :, f, :, :].flatten()
                f2 = delta_feat[0, :, f+1, :, :].flatten()
                fc = torch.nn.functional.cosine_similarity(
                    f1.unsqueeze(0), f2.unsqueeze(0)
                ).item()
                frame_cos_list.append(fc)
            frame_consistency = sum(frame_cos_list) / len(frame_cos_list) if frame_cos_list else 0.0
        elif delta_feat.dim() == 4 and delta_feat.shape[1] >= 2:
            # shape: [C, F, H, W]
            nf = delta_feat.shape[1]
            frame_cos_list = []
            for f in range(nf - 1):
                f1 = delta_feat[:, f, :, :].flatten()
                f2 = delta_feat[:, f+1, :, :].flatten()
                fc = torch.nn.functional.cosine_similarity(
                    f1.unsqueeze(0), f2.unsqueeze(0)
                ).item()
                frame_cos_list.append(fc)
            frame_consistency = sum(frame_cos_list) / len(frame_cos_list) if frame_cos_list else 0.0

        # ── 指标4: delta_feat 与 prompt_embeds 的方向对齐度 ──
        # 直接测量 η_temporal 引起的特征偏移是否与文本语义方向一致
        prompt_alignment = 0.0
        try:
            # prompt_embeds: [batch, seq_len, dim] 或 [batch, dim]
            if prompt_embeds.dim() == 3:
                prompt_dir = prompt_embeds[0].mean(dim=0)  # [dim]
            else:
                prompt_dir = prompt_embeds[0]  # [dim]
            # delta_feat 太大，取 channel 平均作为方向
            if delta_feat.dim() == 5:
                delta_dir = delta_feat[0].mean(dim=(1, 2, 3))  # [C]
            elif delta_feat.dim() == 4:
                delta_dir = delta_feat.mean(dim=(1, 2, 3))  # [C]
            else:
                delta_dir = delta_feat.flatten()[:prompt_dir.shape[0]]
            # 如果维度不匹配，截取/填充
            min_dim = min(delta_dir.shape[0], prompt_dir.shape[0])
            prompt_alignment = torch.nn.functional.cosine_similarity(
                delta_dir[:min_dim].unsqueeze(0).float(),
                prompt_dir[:min_dim].unsqueeze(0).float()
            ).item()
        except Exception:
            pass

        # ── 指标5: η_temporal 本身的帧间余弦 (即 TSR 的 mean_cos 副本) ──
        temporal_frame_cos = 0.0
        if eta_temporal is not None:
            et = eta_temporal
            if et.dim() == 5 and et.shape[2] >= 2:
                nf = et.shape[2]
                tc_list = []
                for f in range(nf - 1):
                    f1 = et[0, :, f, :, :].flatten()
                    f2 = et[0, :, f+1, :, :].flatten()
                    tc = torch.nn.functional.cosine_similarity(
                        f1.unsqueeze(0), f2.unsqueeze(0)
                    ).item()
                    tc_list.append(tc)
                temporal_frame_cos = sum(tc_list) / len(tc_list) if tc_list else 0.0

        # ── PNA score = impact × consistency ──
        # impact 高 + consistency 正 → η_temporal 有利 → PNA 高
        # impact 高 + consistency 负 → η_temporal 有害 → PNA 低
        # impact 低 → η_temporal 影响小 → PNA 中等
        pna_raw = relative_impact * (0.5 + 0.5 * cos_consistency)  # consistency 映射到 [0, 1]

        # ── 映射策略对比：一次输出所有方案 ──
        pna_alpha_max = getattr(cfg, 'pna_alpha_max', 0.006)
        pna_alpha_min = getattr(cfg, 'pna_alpha_min', 0.0005)
        alpha_ref = getattr(cfg, 'fi_alpha_ref', 0.004)
        fi_lambda = getattr(cfg, 'fi_lambda', 0.05)

        # 方案A: 旧 sigmoid (k=500, x0=0.003)
        score_A = 1.0 / (1.0 + math.exp(-500.0 * (pna_raw - 0.003)))
        score_A = max(0.0, min(1.0, score_A))
        alpha_A = pna_alpha_min + score_A * (pna_alpha_max - pna_alpha_min)

        # 方案B: 新 sigmoid (k=800, x0=0.008) — 中心对准典型值，陡度提升区分度
        exp_B = max(min(-800.0 * (pna_raw - 0.008), 500.0), -500.0)
        score_B = 1.0 / (1.0 + math.exp(exp_B))
        score_B = max(0.0, min(1.0, score_B))
        alpha_B = pna_alpha_min + score_B * (pna_alpha_max - pna_alpha_min)

        # 方案C: linear mapping — pna_raw → [0, 1], 基于观察到的范围 [0.005, 0.01]
        pna_raw_lo, pna_raw_hi = 0.005, 0.010
        score_C = max(0.0, min(1.0, (pna_raw - pna_raw_lo) / (pna_raw_hi - pna_raw_lo)))
        alpha_C = pna_alpha_min + score_C * (pna_alpha_max - pna_alpha_min)

        # 方案D: TSR-only (传统 TSR-guided α, 不用 PNA)
        # 从 svd_stats 取 TSR（后续在 _get_latents 中取）
        score_D_placeholder = -1.0  # 占位，在 _get_latents 中用实际 TSR

        # 方案E: 分段 temporal 修正 (v5 — 调高0.20~0.50区间)
        # 基于v4 L2-only实验: 0.20~0.50区间modifier=0.10太低(73/111严重退化)
        # v5调整: 0.20~0.50区间从0.10提到0.20, 避免α直接掉到下限
        #   cos < 0.05 (快运动):  modifier=0.85 → α有机会到0.004~0.005
        #   cos 0.05~0.20 (慢镜头): modifier=0.25 → α≈0.001~0.002
        #   cos 0.20~0.50 (静态):   modifier=0.20 → α≈0.0008~0.001 (v4是0.10)
        #   cos > 0.50 (纯静态):    modifier=0.05 → α≈0.0005 (下限)
        if temporal_frame_cos < 0.05:
            temporal_modifier_E = 0.85
        elif temporal_frame_cos < 0.20:
            temporal_modifier_E = 0.25
        elif temporal_frame_cos < 0.50:
            temporal_modifier_E = 0.20  # v4=0.10, 73/111退化→调高
        else:
            temporal_modifier_E = 0.05
        pna_raw_E = relative_impact * temporal_modifier_E
        score_E = max(0.0, min(1.0, (pna_raw_E - pna_raw_lo * 0.5) / (pna_raw_hi - pna_raw_lo * 0.5)))
        alpha_E = pna_alpha_min + score_E * (pna_alpha_max - pna_alpha_min)

        # 方案F: 分段 temporal 修正 (v5 — 调高0.15~0.60区间)
        # 基于v4 L2-only实验: 0.15~0.60区间modifier太低
        # v5调整: 全面上调中间区间modifier
        #   cos < 0.05 (快运动):  modifier=0.90
        #   cos 0.05~0.15 (慢镜头偏动): modifier=0.50 (v4=0.40)
        #   cos 0.15~0.30 (慢镜头偏静): modifier=0.25 (v4=0.15)
        #   cos 0.30~0.60 (静态):       modifier=0.15 (v4=0.07)
        #   cos > 0.60 (纯静态):        modifier=0.05 (v4=0.03)
        if temporal_frame_cos < 0.05:
            temporal_modifier_F = 0.90
        elif temporal_frame_cos < 0.15:
            temporal_modifier_F = 0.50  # v4=0.40
        elif temporal_frame_cos < 0.30:
            temporal_modifier_F = 0.25  # v4=0.15
        elif temporal_frame_cos < 0.60:
            temporal_modifier_F = 0.15  # v4=0.07
        else:
            temporal_modifier_F = 0.05  # v4=0.03
        pna_raw_F = relative_impact * temporal_modifier_F
        score_F = max(0.0, min(1.0, (pna_raw_F - pna_raw_lo * 0.3) / (pna_raw_hi - pna_raw_lo * 0.3)))
        alpha_F = pna_alpha_min + score_F * (pna_alpha_max - pna_alpha_min)

        # 方案G: 连续平滑修正 — 用 (1-cos)^power 替代分段, 更平滑
        # 优势: 无边界跳变, 自动适配所有 cos 值
        # v5.2: 默认cap=0.35/power=1.5
        temporal_modifier_G = max(0.05, min(0.35, (1.0 - temporal_frame_cos) ** 1.5))
        pna_raw_G = relative_impact * temporal_modifier_G
        score_G = max(0.0, min(1.0, (pna_raw_G - pna_raw_lo * 0.5) / (pna_raw_hi - pna_raw_lo * 0.5)))
        alpha_G = pna_alpha_min + score_G * (pna_alpha_max - pna_alpha_min)

        # ── Coupling 对比: 正向 vs 反向 ──
        # 正向: α_eff / α_ref (α大→λ大，当前实现)
        # 反向: α_ref / α_eff (α大→λ小，SVD已注入运动则FI减少)
        couple_pos_A = max(0.1, min(2.0, alpha_A / alpha_ref))
        couple_neg_A = max(0.3, min(2.0, alpha_ref / max(alpha_A, 1e-8)))
        couple_pos_B = max(0.1, min(2.0, alpha_B / alpha_ref))
        couple_neg_B = max(0.3, min(2.0, alpha_ref / max(alpha_B, 1e-8)))
        couple_pos_C = max(0.1, min(2.0, alpha_C / alpha_ref))
        couple_neg_C = max(0.3, min(2.0, alpha_ref / max(alpha_C, 1e-8)))
        couple_pos_E = max(0.1, min(2.0, alpha_E / alpha_ref))
        couple_neg_E = max(0.3, min(2.0, alpha_ref / max(alpha_E, 1e-8)))
        couple_pos_F = max(0.1, min(2.0, alpha_F / alpha_ref))
        couple_neg_F = max(0.3, min(2.0, alpha_ref / max(alpha_F, 1e-8)))
        couple_pos_G = max(0.1, min(2.0, alpha_G / alpha_ref))
        couple_neg_G = max(0.3, min(2.0, alpha_ref / max(alpha_G, 1e-8)))
        lambda_pos_A = fi_lambda * couple_pos_A
        lambda_neg_A = fi_lambda * couple_neg_A
        lambda_pos_B = fi_lambda * couple_pos_B
        lambda_neg_B = fi_lambda * couple_neg_B
        lambda_pos_C = fi_lambda * couple_pos_C
        lambda_neg_C = fi_lambda * couple_neg_C
        lambda_pos_E = fi_lambda * couple_pos_E
        lambda_neg_E = fi_lambda * couple_neg_E
        lambda_pos_F = fi_lambda * couple_pos_F
        lambda_neg_F = fi_lambda * couple_neg_F
        lambda_pos_G = fi_lambda * couple_pos_G
        lambda_neg_G = fi_lambda * couple_neg_G

        # ── 场景分类启发式 (基于6-18消融实验经验) ──
        # mean_cos < 0.05 → 物体运动型 (如32-动物: α≈0.004, λ≈0.05)
        # mean_cos 0.05~0.10 → 中间型 (如50-异常: α≈0.001, λ≈0.05)
        # mean_cos 0.10~0.20 → 慢镜头场景 (如80-蒲公英: α≈0.002, λ≈0.04)
        # mean_cos > 0.20 → 静态/航拍场景 (如73-仓库/111-航拍: α≈0.001, λ≈0.03)
        if temporal_frame_cos < 0.05:
            scene_hint = '物体运动型(α≈0.003~0.005,λ≈0.05)'
            hint_alpha, hint_lambda = 0.004, 0.05
        elif temporal_frame_cos < 0.10:
            scene_hint = '中间型(α≈0.001~0.003,λ≈0.04~0.05)'
            hint_alpha, hint_lambda = 0.002, 0.045
        elif temporal_frame_cos < 0.20:
            scene_hint = '慢镜头场景(α≈0.001~0.002,λ≈0.03~0.04)'
            hint_alpha, hint_lambda = 0.0015, 0.035
        else:
            scene_hint = '静态/航拍场景(α≈0.0005~0.001,λ≈0.02~0.03)'
            hint_alpha, hint_lambda = 0.0008, 0.025

        # ── 全量诊断日志 ──
        logger.info(
            f"  [PNA] probe_t={t_probe:.2f}, α_test={alpha_test}, "
            f"layer={probe_layer}, "
            f"relative_impact={relative_impact:.6f}, "
            f"cos_consistency={cos_consistency:.4f}, "
            f"pna_raw={pna_raw:.6f}"
        )
        logger.info(
            f"  [PNA 新指标] frame_consistency={frame_consistency:.4f}, "
            f"prompt_alignment={prompt_alignment:.4f}, "
            f"temporal_frame_cos={temporal_frame_cos:.4f}"
        )
        logger.info(
            f"  [PNA 场景推断] mean_cos={temporal_frame_cos:.4f} → {scene_hint}"
        )
        logger.info(
            f"  [PNA 映射对比] α_range=[{pna_alpha_min}, {pna_alpha_max}], α_ref={alpha_ref}, λ_max={fi_lambda}"
        )
        logger.info(
            f"  [PNA-A] sigmoid(k=500,x0=0.003): score={score_A:.4f} → α={alpha_A:.6f} | "
            f"couple+scale={couple_pos_A:.3f} λ_eff={lambda_pos_A:.4f} | "
            f"couple-scale={couple_neg_A:.3f} λ_eff={lambda_neg_A:.4f}"
        )
        logger.info(
            f"  [PNA-B] sigmoid(k=800,x0=0.008): score={score_B:.4f} → α={alpha_B:.6f} | "
            f"couple+scale={max(0.1, alpha_B / alpha_ref):.3f} λ_eff={fi_lambda * max(0.1, alpha_B / alpha_ref):.4f} | "
            f"couple-scale={max(0.1, alpha_ref / max(alpha_B, 1e-8)):.3f} λ_eff={fi_lambda * max(0.1, alpha_ref / max(alpha_B, 1e-8)):.4f}"
        )
        logger.info(
            f"  [PNA-C] linear({pna_raw_lo}~{pna_raw_hi}):    score={score_C:.4f} → α={alpha_C:.6f} | "
            f"couple+scale={couple_pos_C:.3f} λ_eff={lambda_pos_C:.4f} | "
            f"couple-scale={couple_neg_C:.3f} λ_eff={lambda_neg_C:.4f}"
        )
        # ── 分段区间诊断 (v5) ──
        if temporal_frame_cos < 0.05:
            piece_label_E = 'cos<0.05→快运动(mod=0.85)'
            piece_label_F = 'cos<0.05→快运动(mod=0.90)'
        elif temporal_frame_cos < 0.15:
            piece_label_E = '0.05≤cos<0.20→慢镜头(mod=0.25)'
            piece_label_F = '0.05≤cos<0.15→慢镜头偏动(mod=0.50)'
        elif temporal_frame_cos < 0.20:
            piece_label_E = '0.05≤cos<0.20→慢镜头(mod=0.25)'
            piece_label_F = '0.15≤cos<0.30→慢镜头偏静(mod=0.25)'
        elif temporal_frame_cos < 0.30:
            piece_label_E = '0.20≤cos<0.50→静态(mod=0.20)'
            piece_label_F = '0.15≤cos<0.30→慢镜头偏静(mod=0.25)'
        elif temporal_frame_cos < 0.50:
            piece_label_E = '0.20≤cos<0.50→静态(mod=0.20)'
            piece_label_F = '0.30≤cos<0.60→静态(mod=0.15)'
        elif temporal_frame_cos < 0.60:
            piece_label_E = 'cos≥0.50→纯静态(mod=0.05)'
            piece_label_F = '0.30≤cos<0.60→静态(mod=0.15)'
        else:
            piece_label_E = 'cos≥0.50→纯静态(mod=0.05)'
            piece_label_F = 'cos≥0.60→纯静态(mod=0.05)'

        logger.info(
            f"  [PNA-E] 分段temporal修正(v5): score={score_E:.4f} → α={alpha_E:.6f} | "
            f"{piece_label_E} | "
            f"pna_raw_E={pna_raw_E:.6f} | "
            f"couple+scale={couple_pos_E:.3f} λ_eff={lambda_pos_E:.4f} | "
            f"couple-scale={couple_neg_E:.3f} λ_eff={lambda_neg_E:.4f}"
        )
        logger.info(
            f"  [PNA-F] 分段temporal修正(v5): score={score_F:.4f} → α={alpha_F:.6f} | "
            f"{piece_label_F} | "
            f"pna_raw_F={pna_raw_F:.6f} | "
            f"couple+scale={couple_pos_F:.3f} λ_eff={lambda_pos_F:.4f} | "
            f"couple-scale={couple_neg_F:.3f} λ_eff={lambda_neg_F:.4f}"
        )
        logger.info(
            f"  [PNA-G] 连续平滑修正((1-cos)^1.5, cap=0.35): score={score_G:.4f} → α={alpha_G:.6f} | "
            f"mod={temporal_modifier_G:.4f} | "
            f"pna_raw_G={pna_raw_G:.6f} | "
            f"couple+scale={couple_pos_G:.3f} λ_eff={lambda_pos_G:.4f} | "
            f"couple-scale={couple_neg_G:.3f} λ_eff={lambda_neg_G:.4f}"
        )
        logger.info(
            f"  [PNA 经验参考] 场景建议 α≈{hint_alpha:.4f}, λ≈{hint_lambda:.4f}"
        )
        # ── 参数扫描: 一次输出多种(cap, power)组合的α值 ──
        # 目的: 从日志直接判断最佳参数，避免反复实验
        sweep_configs = [
            # (cap, power, label)
            (0.20, 1.0, 'G1:cap=0.20,pw=1.0'),
            (0.25, 1.2, 'G2:cap=0.25,pw=1.2'),
            (0.30, 1.3, 'G3:cap=0.30,pw=1.3'),
            (0.35, 1.5, 'G4:cap=0.35,pw=1.5(默认)'),
            (0.40, 1.5, 'G5:cap=0.40,pw=1.5'),
            (0.50, 1.5, 'G6:cap=0.50,pw=1.5'),
            (0.60, 1.2, 'G7:cap=0.60,pw=1.2(v5.1)'),
            (0.60, 0.8, 'G8:cap=0.60,pw=0.8(v5.0)'),
        ]
        logger.info(
            f"  [参数扫描] cos={temporal_frame_cos:.4f}, relative_impact={relative_impact:.6f}, "
            f"经验α≈{hint_alpha:.4f}"
        )
        logger.info(
            f"  [参数扫描表] {'方案':<28} | {'modifier':>8} | {'pna_raw':>10} | {'score':>6} | {'α':>10} | {'与经验偏差':>10}"
        )
        logger.info(
            f"  [参数扫描表] {'─'*28}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*6}─┼─{'─'*10}─┼─{'─'*10}"
        )
        best_sweep = None
        best_sweep_dev = float('inf')
        for cap, power, label in sweep_configs:
            sw_mod = max(0.05, min(cap, (1.0 - temporal_frame_cos) ** power))
            sw_raw = relative_impact * sw_mod
            sw_score = max(0.0, min(1.0, (sw_raw - pna_raw_lo * 0.5) / (pna_raw_hi - pna_raw_lo * 0.5)))
            sw_alpha = pna_alpha_min + sw_score * (pna_alpha_max - pna_alpha_min)
            sw_dev = abs(sw_alpha - hint_alpha)
            marker = ' ★' if sw_dev < best_sweep_dev else ''
            if sw_dev < best_sweep_dev:
                best_sweep_dev = sw_dev
                best_sweep = label
            logger.info(
                f"  [参数扫描表] {label:<28} | {sw_mod:>8.4f} | {sw_raw:>10.6f} | {sw_score:>6.4f} | {sw_alpha:>10.6f} | {sw_alpha-hint_alpha:>+10.4f}{marker}"
            )
        logger.info(
            f"  [参数扫描最佳] {best_sweep}(偏差={best_sweep_dev:.4f})"
        )
        # 方案E的α也加入对比
        logger.info(
            f"  [参数扫描对比] 方案E(分段v5): α={alpha_E:.6f}(偏差={abs(alpha_E-hint_alpha):.4f}) | "
            f"方案G(默认): α={alpha_G:.6f}(偏差={abs(alpha_G-hint_alpha):.4f}) | "
            f"扫描最佳: {best_sweep}(偏差={best_sweep_dev:.4f})"
        )
        logger.info(
            f"  [PNA 方案与经验α的偏差] "
            f"A:|{alpha_A-hint_alpha:+.4f}| "
            f"B:|{alpha_B-hint_alpha:+.4f}| "
            f"C:|{alpha_C-hint_alpha:+.4f}| "
            f"E:|{alpha_E-hint_alpha:+.4f}| "
            f"F:|{alpha_F-hint_alpha:+.4f}| "
            f"G:|{alpha_G-hint_alpha:+.4f}| "
            f"→ 越接近0越好"
        )
        # ── 最佳方案判定 ──
        deviations = {
            'A': abs(alpha_A - hint_alpha),
            'B': abs(alpha_B - hint_alpha),
            'C': abs(alpha_C - hint_alpha),
            'E': abs(alpha_E - hint_alpha),
            'F': abs(alpha_F - hint_alpha),
            'G': abs(alpha_G - hint_alpha),
        }
        best_scheme = min(deviations, key=deviations.get)
        logger.info(
            f"  [PNA 最佳方案] {best_scheme}(偏差最小={deviations[best_scheme]:.4f}) "
            f"vs 选用G(偏差={deviations['G']:.4f}) "
            f"{'✓ 一致' if best_scheme == 'G' else '✗ 不一致, 考虑切换'}"
        )
        # ── 反向Coupling 下的 λ_eff 对比表 ──
        logger.info(
            f"  [PNA λ_eff 对比(反向Coupling)] "
            f"A:λ={lambda_neg_A:.4f} | "
            f"B:λ={lambda_neg_B:.4f} | "
            f"C:λ={lambda_neg_C:.4f} | "
            f"E:λ={lambda_neg_E:.4f} | "
            f"F:λ={lambda_neg_F:.4f} | "
            f"G(选用):λ={lambda_neg_G:.4f} | "
            f"经验:λ≈{hint_lambda:.4f}"
        )
        # ── No-Coupling 下的 α 直接对比 (关键: 纯L2效果) ──
        logger.info(
            f"  [PNA 纯L2 α 对比(无Coupling)] "
            f"A:α={alpha_A:.6f} | "
            f"B:α={alpha_B:.6f} | "
            f"C:α={alpha_C:.6f} | "
            f"E:α={alpha_E:.6f} | "
            f"F:α={alpha_F:.6f} | "
            f"G(选用):α={alpha_G:.6f} | "
            f"经验:α≈{hint_alpha:.6f}"
        )

        # ── 默认使用方案G (连续平滑修正) 作为主 score ──
        # v5切换: 方案E在0.20~0.50区间仍可能过低, 方案F的边界跳变不可控
        # 方案G用(1-cos)^0.8连续映射, 无边界跳变, 中间区间给更高modifier
        # 同时缓存所有方案的 α 供 FI 层使用
        pna_score = score_G

        # 缓存多方案 α 到 cfg，供 _generate_with_fi 做诊断
        cfg._pna_alphas = {
            'A': alpha_A, 'B': alpha_B, 'C': alpha_C, 'E': alpha_E, 'F': alpha_F, 'G': alpha_G,
            'hint': hint_alpha,
        }
        cfg._pna_raw = pna_raw
        cfg._pna_frame_consistency = frame_consistency
        cfg._pna_prompt_alignment = prompt_alignment
        cfg._pna_temporal_frame_cos = temporal_frame_cos

        return pna_score

    def _cache_fi_ref_features(
        self,
        ref_trajectory: Dict[float, torch.Tensor],
        prompt_embeds: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Feature Injection: 在反演过程中缓存 DiT 每步的中间特征。

        核心思路:
            不在 latent 空间做方向修正 (VDA), 而是缓存反演过程中 DiT 的中间表示,
            在生成时以残差方式注入到对应层。

        实现方式:
            沿反演轨迹的若干关键 t 值, 重新前向传播 DiT, 通过 hook 捕获中间特征。
            这比在反演时直接 hook 更灵活, 因为可以选择性地只缓存关键步。

        Args:
            ref_trajectory: 反演轨迹 {t_value: z_ref_tensor(cpu)}
            prompt_embeds: prompt embedding (用于 DiT 前向)

        Returns:
            ref_features: dict {step_index: {layer_idx: tensor}}
        """
        cfg = self.config
        num_steps = cfg.num_inference_steps

        logger.info("  ═══════════════════════════════════════════════")
        logger.info("  [FI] 开始缓存参考特征")
        logger.info("  ═══════════════════════════════════════════════")

        # 确定要缓存哪些步的特征
        # 选择与生成步数相同的关键 t 值
        traj_keys = sorted(ref_trajectory.keys())

        # 映射: 生成 step_index → ref_trajectory 的 t 值
        # 生成: step_index 完成后 t = 1 - (step_index+1)/N
        # 反演: 同一进度对应 t_traj = 1 - t_progress
        dt_gen = 1.0 / num_steps

        # 只缓存与生成步对应的点 (减少显存)
        cache_points = {}  # step_index → t_traj_key
        for step_idx in range(num_steps):
            t_progress = 1.0 - (step_idx + 1) / num_steps
            t_traj = 1.0 - t_progress
            # 找最近的 trajectory key
            nearest_t = min(traj_keys, key=lambda t: abs(t - t_traj))
            cache_points[step_idx] = nearest_t

        logger.info(
            f"  [FI] 缓存 {len(cache_points)} 个关键步的特征 "
            f"(对应 {num_steps} 步生成)"
        )

        # 解析注入层配置
        transformer = self.pipe.transformer
        num_layers = len(transformer.blocks) if hasattr(transformer, 'blocks') else 30

        if cfg.fi_layers == "all":
            target_layers = list(range(num_layers))
        elif cfg.fi_layers == "early":
            target_layers = list(range(0, num_layers // 3))
        elif cfg.fi_layers == "mid":
            target_layers = list(range(num_layers // 3, 2 * num_layers // 3))
        elif cfg.fi_layers in ("last", "late"):  # late 是 last 的别名
            target_layers = list(range(2 * num_layers // 3, num_layers))
        else:
            # 逗号分隔的层号
            try:
                target_layers = [int(x.strip()) for x in cfg.fi_layers.split(",")]
            except ValueError:
                logger.warning(f"  [FI] 无法解析 fi_layers='{cfg.fi_layers}', 使用 'mid'")
                target_layers = list(range(num_layers // 3, 2 * num_layers // 3))

        logger.info(
            f"  [FI] 注入层: {target_layers} ({len(target_layers)}/{num_layers} 层)"
        )

        # 缓存参考特征
        ref_features = {}  # {step_index: {layer_idx: feature_tensor(cpu)}}

        # Hook 用于捕获中间特征
        captured_features = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # 捕获输出
                if isinstance(output, tuple):
                    captured_features[layer_idx] = output[0].detach().cpu()
                else:
                    captured_features[layer_idx] = output.detach().cpu()
            return hook_fn

        # 注册 hook
        hooks = []
        blocks = transformer.blocks if hasattr(transformer, 'blocks') else []
        for layer_idx in target_layers:
            if layer_idx < len(blocks):
                block = blocks[layer_idx]
                # Hook 在 block 的前向传播之后
                # Wan2.1 DiT block 结构: self-attn → cross-attn → ffn
                # 我们 hook cross-attn 输出 (如果 cache_mode=attention)
                # 或者 block 整体输出 (如果 cache_mode=hidden)
                if cfg.fi_cache_mode == "attention" and hasattr(block, 'cross_attn'):
                    h = block.cross_attn.register_forward_hook(make_hook(layer_idx))
                elif cfg.fi_cache_mode == "mlp" and hasattr(block, 'ffn'):
                    h = block.ffn.register_forward_hook(make_hook(layer_idx))
                else:
                    # fallback: hook 整个 block
                    h = block.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

        try:
            # 沿轨迹前向传播缓存特征
            for step_idx in sorted(cache_points.keys()):
                t_traj_key = cache_points[step_idx]
                z_ref = ref_trajectory[t_traj_key].to(
                    device=self.device, dtype=self.dtype
                )

                # 构造 timestep tensor
                # 反演 t 值: t_traj_key 是反演坐标 (t=1=数据, t=0=噪声)
                # WanPipeline 的 timestep: t=1=噪声, t=0=数据
                # 所以对应 pipeline timestep = t_traj_key
                t_tensor = torch.full(
                    (z_ref.shape[0],), t_traj_key, device=self.device, dtype=z_ref.dtype
                )

                captured_features.clear()

                # DiT 前向传播
                with torch.no_grad():
                    _ = transformer(
                        hidden_states=z_ref,
                        timestep=t_tensor,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )

                # 保存捕获的特征
                ref_features[step_idx] = {}
                for layer_idx, feat in captured_features.items():
                    ref_features[step_idx][layer_idx] = feat

                if step_idx % 5 == 0:
                    logger.info(
                        f"    [FI Cache] step {step_idx}: "
                        f"t_traj={t_traj_key:.3f}, "
                        f"cached {len(captured_features)} layers"
                    )

        finally:
            # 移除所有 hook
            for h in hooks:
                h.remove()

        total_cached = sum(len(v) for v in ref_features.values())
        logger.info(
            f"  [FI] 参考特征缓存完成: "
            f"{len(ref_features)} steps × {len(target_layers)} layers = {total_cached} tensors"
        )

        # 保存元信息
        ref_features["_meta"] = {
            "target_layers": target_layers,
            "num_layers": num_layers,
            "cache_mode": cfg.fi_cache_mode,
            "num_steps": num_steps,
        }

        return ref_features

    @torch.no_grad()
    def _generate_with_fi(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        ref_features: Dict[str, Any],
        eta_temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L3 V3: Feature Injection (FI) 生成。

        核心思想:
            在生成过程中, 通过 hook 在 DiT 每步前向传播时, 将参考特征以残差方式注入。

        数学:
            h_injected = h_current + λ * (h_ref - h_current)
                       = (1-λ) * h_current + λ * h_ref

        优势:
            - 不修改 ODE 积分路径 (latent 不变), 只修改 DiT 的中间表示
            - 特征空间语义对齐比 latent 空间方向对齐更鲁棒
            - 类似 ControlNet 的零训练注入, 但不需要训练

        Args:
            prompt: 生成 prompt
            latents: 初始噪声 (可含 SVD prior)
            generator: 随机数生成器
            ref_features: 参考特征缓存 {step_index: {layer_idx: tensor}}
            eta_temporal: SVD 滤波后的噪声 (用于质量门控)
        """
        cfg = self.config
        num_steps = cfg.num_inference_steps

        logger.info("  ═══════════════════════════════════════════════")
        logger.info("  [FI] 开始 Feature Injection 生成")
        logger.info("  ═══════════════════════════════════════════════")

        # 提取元信息
        meta = ref_features.get("_meta", {})
        target_layers = meta.get("target_layers", [])

        # ── L2-L3 协同: 获取当前 α_eff ──
        # v5策略: 默认不开启Coupling (v4实验证明Coupling整体无效)
        # L2(SVD)和L3(FI)独立工作, 各自适应门控
        # Coupling仅作为诊断日志输出, 不影响实际 λ
        alpha_coupling_scale = 1.0  # 默认不缩放
        pna_alphas = getattr(cfg, '_pna_alphas', {})
        alpha_eff = getattr(cfg, '_current_alpha_eff', None)
        alpha_ref = getattr(cfg, 'fi_alpha_ref', 0.004)

        if getattr(cfg, 'fi_alpha_coupling', False) and cfg.adaptive_alpha:
            if alpha_eff is not None:
                # 反向 Coupling: α 高 → FI λ 小 (互补)
                alpha_coupling_scale = max(0.3, min(2.0, alpha_ref / max(alpha_eff, 1e-8)))
                # 正向 Coupling 对比 (旧逻辑): α 高 → FI λ 大
                couple_pos = max(0.1, alpha_eff / alpha_ref)

                logger.info(
                    f"  [FI-α Coupling] α_eff={alpha_eff:.6f}, α_ref={alpha_ref:.4f}"
                )
                logger.info(
                    f"  [FI-α Coupling 选用] 反向(互补): scale={alpha_coupling_scale:.4f} "
                    f"(λ_max: {cfg.fi_lambda:.4f}→{cfg.fi_lambda * alpha_coupling_scale:.4f})"
                )
                logger.info(
                    f"  [FI-α Coupling 对比] 正向(同向): scale={couple_pos:.4f} "
                    f"(λ_max: {cfg.fi_lambda:.4f}→{cfg.fi_lambda * couple_pos:.4f}) | "
                    f"无Coupling: scale=1.0000 "
                    f"(λ_max: {cfg.fi_lambda:.4f})"
                )

                # ── 多方案 α 对应的 Coupling 诊断 ──
                if pna_alphas:
                    hint_alpha = pna_alphas.get('hint', 0)
                    logger.info(
                        f"  [FI-α 多方案 Coupling 诊断] (反向=α_ref/α, 正向=α/α_ref) "
                        f"经验参考: α_hint={hint_alpha:.4f}"
                    )
                    for scheme, a_val in pna_alphas.items():
                        if a_val > 0 and scheme != 'hint':
                            neg_s = max(0.3, min(3.0, alpha_ref / a_val))
                            pos_s = max(0.1, a_val / alpha_ref)
                            # 经验参考 λ
                            if hint_alpha > 0:
                                hint_neg_s = max(0.3, min(3.0, alpha_ref / hint_alpha))
                                hint_lambda_neg = cfg.fi_lambda * hint_neg_s
                            else:
                                hint_lambda_neg = -1
                            logger.info(
                                f"    方案{scheme}: α={a_val:.6f}(偏差={a_val-hint_alpha:+.4f}) → "
                                f"反向scale={neg_s:.3f} λ_eff={cfg.fi_lambda * neg_s:.4f} | "
                                f"正向scale={pos_s:.3f} λ_eff={cfg.fi_lambda * pos_s:.4f} | "
                                f"经验λ_eff≈{hint_lambda_neg:.4f}"
                            )
        else:
            # ── L2-only + 无Coupling 诊断 ──
            hint_alpha_l2 = pna_alphas.get('hint', 0)
            temporal_cos_l2 = getattr(cfg, '_pna_temporal_frame_cos', -1)
            frame_cons_l2 = getattr(cfg, '_pna_frame_consistency', -1)
            prompt_align_l2 = getattr(cfg, '_pna_prompt_alignment', -1)
            if alpha_eff is not None:
                logger.info(
                    f"  [L2+FI 独立门控诊断] Coupling=OFF, FI λ固定={cfg.fi_lambda:.4f}"
                )
                logger.info(
                    f"  [L2+FI 独立门控] α_eff={alpha_eff:.6f}, "
                    f"经验α≈{hint_alpha_l2:.4f}, "
                    f"偏差={alpha_eff - hint_alpha_l2:+.4f}"
                )
                if temporal_cos_l2 >= 0:
                    logger.info(
                        f"  [L2+FI 独立门控] temporal_frame_cos={temporal_cos_l2:.4f}, "
                        f"frame_consistency={frame_cons_l2:.4f}, "
                        f"prompt_alignment={prompt_align_l2:.4f}"
                    )
                # 输出所有方案的 α 对比
                if pna_alphas:
                    logger.info(
                        f"  [L2+FI α 全方案对比] "
                        f"A={pna_alphas.get('A', 0):.6f} | "
                        f"B={pna_alphas.get('B', 0):.6f} | "
                        f"C={pna_alphas.get('C', 0):.6f} | "
                        f"E={pna_alphas.get('E', 0):.6f} | "
                        f"F={pna_alphas.get('F', 0):.6f} | "
                        f"G(选用)={alpha_eff:.6f} | "
                        f"经验={hint_alpha_l2:.6f}"
                    )
                # 假设 Coupling 开启时的 λ_eff 对比（虚拟计算）
                couple_neg_l2 = max(0.3, min(2.0, alpha_ref / max(alpha_eff, 1e-8)))
                logger.info(
                    f"  [L2+FI vs Coupling 虚拟对比] "
                    f"当前(无Coupling): λ_eff={cfg.fi_lambda:.4f}(固定) | "
                    f"若开启反向Coupling: λ_eff={cfg.fi_lambda * couple_neg_l2:.4f}(scale={couple_neg_l2:.3f})"
                )
                # v4 L2-only 结果参考
                temporal_cos_val = temporal_cos_l2 if temporal_cos_l2 >= 0 else 0
                if temporal_cos_val < 0.10:
                    l2_ref = "v4 L2-only: 32(cos=0.02) X-CLIP=0.812, 50(cos=0.08) X-CLIP=0.770"
                elif temporal_cos_val < 0.30:
                    l2_ref = "v4 L2-only: 80(cos=0.14) X-CLIP=0.822"
                else:
                    l2_ref = "v4 L2-only: 73(cos=0.32) X-CLIP=0.807, 111(cos=0.55) X-CLIP=0.814"
                logger.info(
                    f"  [v4 L2-only 参考值] {l2_ref}"
                )

        # ── 预计算 λ 调度 (含 α 协同缩放) ──
        effective_fi_lambda = cfg.fi_lambda * alpha_coupling_scale
        lambda_values = self._compute_schedule(
            num_steps, effective_fi_lambda, cfg.fi_schedule
        )

        # ── 质量门控 ──
        quality_scale = 1.0
        if cfg.fi_quality_gate:
            quality_scale = self._compute_quality_scale(eta_temporal)
            # M_d 修正 Quality Scale (可选, 默认关闭)
            # 仅当 fi_md_gate=True 时启用: QS_eff = QS * max(M_d, fi_qs_md_floor)
            # 消融实验: 先只改 L2 SVD 门控, 不改 L3 FI 门控
            if getattr(cfg, 'fi_md_gate', False):
                md = getattr(cfg, '_current_md', 1.0)
                fi_qs_floor = getattr(cfg, 'fi_qs_md_floor', 0.5)
                md_for_qs = max(md, fi_qs_floor)
                if md_for_qs < 1.0:
                    quality_scale_original = quality_scale
                    quality_scale = quality_scale * md_for_qs
                    logger.info(
                        f"  [Quality Scale × M_d] QS_orig={quality_scale_original:.4f}, "
                        f"M_d={md:.2f}, floor={fi_qs_floor} → M_d_eff={md_for_qs:.2f}, "
                        f"QS_eff={quality_scale:.4f}"
                    )
            if quality_scale < 1e-6:
                logger.info(f"  [FI] 质量门控 scale≈0, 跳过 FI, 走标准生成")
                return self._generate(prompt, latents, generator, negative_prompt)

        logger.info(
            f"  [FI] λ_max={cfg.fi_lambda}"
            f"{'→'+f'{effective_fi_lambda:.4f}' if alpha_coupling_scale < 1.0 else ''}, "
            f"schedule={cfg.fi_schedule}, "
            f"quality_scale={quality_scale:.4f}, "
            f"layers={target_layers}, cache_mode={cfg.fi_cache_mode}"
        )
        if cfg.fi_adaptive_gate:
            logger.info(
                f"  [FI] 自适应门控: ON, temp={cfg.fi_adaptive_temp} "
                f"(特征越接近参考→注入越少)"
            )

        # ── FI 统计 ──
        fi_stats = {
            "steps_with_ref": 0,
            "steps_no_ref": 0,
            "total_injection_norm": 0.0,
            "per_step": [],
        }

        # ── 注册注入 hook ──
        transformer = self.pipe.transformer
        blocks = transformer.blocks if hasattr(transformer, 'blocks') else []

        # 当前步的参考特征和 λ (在 callback 中更新)
        current_ref = [{}]  # {layer_idx: feature_tensor}
        current_lambda = [0.0]
        injection_stats_per_step = [{}]  # 每步注入统计

        # ── EMA 特征平滑: 跨步参考特征时序平滑 ──
        ema_decay = 0.7  # 70% 来自上一步平滑值, 30% 来自当前步原始值
        ema_ref_prev = [None]  # 上一步的 EMA 平滑特征 {layer_idx: tensor}

        def make_injection_hook(layer_idx):
            """创建注入 hook: h_injected = (1-λ)*h_current + λ*h_ref"""
            def hook_fn(module, input, output):
                if layer_idx not in current_ref[0]:
                    return output

                h_ref = current_ref[0][layer_idx]
                lam = current_lambda[0]

                if lam < 1e-8:
                    return output

                # 获取当前输出
                if isinstance(output, tuple):
                    h_current = output[0]
                else:
                    h_current = output

                # 确保形状匹配
                h_ref_dev = h_ref.to(device=h_current.device, dtype=h_current.dtype)

                if h_ref_dev.shape != h_current.shape:
                    # 形状不匹配时跳过 (可能是 batch 维度差异)
                    if step_idx_ref[0] % 5 == 0:
                        logger.debug(
                            f"      [FI Hook layer {layer_idx}] 形状不匹配: "
                            f"ref={h_ref_dev.shape} vs current={h_current.shape}, 跳过"
                        )
                    return output

                # ── 自适应门控: 基于特征对齐度调节 λ ──
                lam_eff = lam
                if cfg.fi_adaptive_gate and lam > 1e-8:
                    # 计算当前特征与参考特征的余弦相似度
                    h_cur_flat = h_current.flatten()
                    h_ref_flat = h_ref_dev.flatten()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        h_cur_flat.unsqueeze(0), h_ref_flat.unsqueeze(0)
                    ).item()
                    # cos_sim 高 → 特征已对齐 → 不需要注入 → gate 小
                    # cos_sim 低 → 特征偏离 → 需要注入 → gate 大
                    # gate = 1 - sigmoid(temp * (cos_sim - 0.5))
                    # 当 cos_sim=0.5 时 gate=0.5, cos_sim=1 时 gate→0, cos_sim=0 时 gate→1
                    gate = 1.0 - torch.sigmoid(
                        torch.tensor(cfg.fi_adaptive_temp * (cos_sim - 0.5))
                    ).item()
                    lam_eff = lam * gate

                    # v5: 门控统计 (每5步输出一次, info级别)
                    if step_idx_ref[0] % 5 == 0 and layer_idx == target_layers[0]:
                        logger.info(
                            f"    [FI Adaptive step={step_idx_ref[0]}] "
                            f"layer={layer_idx}, cos={cos_sim:.4f}, "
                            f"gate={gate:.4f}, λ={lam:.5f}→{lam_eff:.5f}"
                        )

                # 残差注入: h_injected = (1-λ_eff)*h_current + λ_eff*h_ref
                h_injected = (1.0 - lam_eff) * h_current + lam_eff * h_ref_dev

                # 记录注入统计
                injection_norm = (h_injected - h_current).norm().item()
                injection_stats_per_step[0][layer_idx] = injection_norm

                if isinstance(output, tuple):
                    return (h_injected,) + output[1:]
                return h_injected

            return hook_fn

        # 注册 hook
        hooks = []
        step_idx_ref = [0]  # 用于在 hook 中引用当前步

        for layer_idx in target_layers:
            if layer_idx < len(blocks):
                block = blocks[layer_idx]
                if cfg.fi_cache_mode == "attention" and hasattr(block, 'cross_attn'):
                    h = block.cross_attn.register_forward_hook(make_injection_hook(layer_idx))
                elif cfg.fi_cache_mode == "mlp" and hasattr(block, 'ffn'):
                    h = block.ffn.register_forward_hook(make_injection_hook(layer_idx))
                else:
                    h = block.register_forward_hook(make_injection_hook(layer_idx))
                hooks.append(h)

        try:
            # ── FI callback: 每步注入前更新参考特征和 λ ──
            def fi_callback(pipe, step_index, timestep, callback_kwargs):
                """FI callback: 更新注入 hook 的参数。"""
                # 更新当前步索引
                step_idx_ref[0] = step_index

                # 计算当前步的 λ
                lam = lambda_values[step_index] * quality_scale
                current_lambda[0] = lam

                # v5: 步级详细日志 (每5步输出一次)
                if step_index % 5 == 0 or step_index == num_steps - 1:
                    logger.info(
                        f"    [FI step={step_index}/{num_steps}] "
                        f"λ_sched={lambda_values[step_index]:.5f} * qs={quality_scale:.4f} = λ_eff={lam:.5f} | "
                        f"ref_layers={list(current_ref[0].keys()) if current_ref[0] else 'none'}"
                    )

                # 获取当前步的参考特征
                step_features = ref_features.get(step_index, {})
                current_ref[0] = {}

                for layer_idx in target_layers:
                    if layer_idx in step_features:
                        h_ref_raw = step_features[layer_idx].to(self.device)

                        # ── EMA 特征平滑 ──
                        # 第一步: 直接使用原始参考特征
                        # 后续步: h_ref_smooth = ema_decay * prev + (1 - ema_decay) * current
                        if ema_ref_prev[0] is not None and layer_idx in ema_ref_prev[0]:
                            h_ref_prev = ema_ref_prev[0][layer_idx]
                            # 确保形状匹配
                            if h_ref_prev.shape == h_ref_raw.shape:
                                h_ref_smooth = ema_decay * h_ref_prev + (1.0 - ema_decay) * h_ref_raw
                            else:
                                h_ref_smooth = h_ref_raw  # 形状不匹配时回退
                        else:
                            h_ref_smooth = h_ref_raw

                        current_ref[0][layer_idx] = h_ref_smooth

                # 更新 EMA 缓存 (detach 避免计算图增长)
                if current_ref[0]:
                    ema_ref_prev[0] = {
                        layer_idx: feat.detach()
                        for layer_idx, feat in current_ref[0].items()
                    }

                # ── 更新 FI 步级统计 ──
                if current_ref[0]:
                    fi_stats['steps_with_ref'] += 1
                else:
                    fi_stats['steps_no_ref'] += 1

                # 汇总本步 hook 写入的注入 norm (在重置之前)
                if injection_stats_per_step[0]:
                    step_total_norm = sum(injection_stats_per_step[0].values())
                    fi_stats['total_injection_norm'] += step_total_norm
                    fi_stats['per_step'].append(dict(injection_stats_per_step[0]))
                else:
                    fi_stats['per_step'].append({})

                # 重置注入统计
                injection_stats_per_step[0] = {}

                # v5: 注入统计日志 (每10步输出一次)
                if step_index % 10 == 0 and step_index > 0 and fi_stats['per_step']:
                    last_stats = fi_stats['per_step'][-1]
                    total_norm = sum(last_stats.values()) if last_stats else 0
                    logger.info(
                        f"    [FI step={step_index} 累计] "
                        f"total_norm={fi_stats['total_injection_norm']:.4f}, "
                        f"last_step_norm={total_norm:.4f}, "
                        f"steps_with_ref={fi_stats['steps_with_ref']}/{step_index}"
                    )

                # callback 不修改 latents
                return callback_kwargs

            # ── 构建生成参数 ──
            kwargs = {
                "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "guidance_scale": cfg.guidance_scale,
            "num_inference_steps": num_steps,
                "generator": generator,
                "output_type": "pt",
                "callback_on_step_end": fi_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
            if latents is not None:
                kwargs["latents"] = latents

            output = self.pipe(**kwargs)

        finally:
            # 移除所有 hook
            for h in hooks:
                h.remove()

        # ── FI 统计总结 (v5: 更详细) ──
        logger.info("  ═══════════════════════════════════════════════")
        logger.info(f"  [FI] 生成完成总结")
        logger.info(
            f"    λ schedule: first 5={[f'{l:.4f}' for l in lambda_values[:5]]}, "
            f"last 5={[f'{l:.4f}' for l in lambda_values[-5:]]}"
        )
        logger.info(
            f"    Coupling: scale={alpha_coupling_scale:.4f}, "
            f"effective_λ_max={effective_fi_lambda:.4f}"
        )
        logger.info(
            f"    Quality scale: {quality_scale:.4f}"
        )
        logger.info(
            f"    Steps: with_ref={fi_stats['steps_with_ref']}/{num_steps}, "
            f"no_ref={fi_stats['steps_no_ref']}/{num_steps}"
        )
        logger.info(
            f"    Total injection norm: {fi_stats['total_injection_norm']:.4f}"
        )
        # v5: α_eff 回顾
        if alpha_eff is not None:
            hint_a = pna_alphas.get('hint', 0) if pna_alphas else 0
            logger.info(
                f"    α_eff={alpha_eff:.6f}, 经验α≈{hint_a:.4f}, "
                f"偏差={alpha_eff - hint_a:+.4f}"
            )
        logger.info("  ═══════════════════════════════════════════════")

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


