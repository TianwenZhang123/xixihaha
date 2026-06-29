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

import json
import time
import shutil
import math
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field

import torch

from .distributed import setup_single_gpu, load_model_single_gpu
from .flow_matching import FlowMatchingInverter, encode_video_to_latents
from .svd_filter import SVDFilter, SVDFilterConfig
from .video_utils import (
    load_video, save_video_tensor, normalize_video, denormalize_video,
    create_vertical_composite,
)
from .vlm_client import create_vlm_client
from .prompt_decompose import create_prompt_decomposer

logger = logging.getLogger(__name__)

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
    alpha: float = 0.004           # SVD 混合权重 (v5 fixed, 推荐 0.004)
                                  # η = √α·η_temporal + √(1-α)·η_random
    rho_s: float = 0.1            # 空间SVD阈值 (去内容)
    rho_m: float = 0.9            # 时间SVD阈值 (保运动)
    svd_motion_filter: bool = False   # 方向3b: 运动方向一致性过滤
    svd_progressive: bool = False     # 方向2: 渐进多尺度SVD
    fi_sparse_ratio: float = 0.0       # 方向3: 通道选择性FI, 0=关闭(全通道), 0.5=只注入50%最重要通道
    prompt_decompose: bool = False     # L1: 结构化分解+CLIP择优
    llm_api_key: str = ""              # LLM API key
    llm_api_base: str = "https://token-plan-cn.xiaomimimo.com/v1"
    llm_model: str = "mimo-v2.5-pro"
    # svd_alternate: bool = False     # 方向5: 交替注入 — 已注释
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
    fi_quality_k: float = 20.0            # 质量门控 sigmoid 斜率 (越大越陡峭)
    fi_quality_min_scale: float = 0.1     # 质量门控最低注入比例 (保留的最小引导量)
    fi_quality_skip_threshold: Optional[float] = None  # 硬跳过阈值: mean_cos > 此值时完全跳过 FI (None=不启用)
    fi_quality_skip_svd: bool = False  # 方向A: 跳过FI时同时关闭SVD blend (默认False)
    fi_max_injection_norm: Optional[float] = None  # 方向B: Total injection norm 硬上限 (None=不启用)
    fi_norm_decay_min: float = 0.3  # 方向B: norm超限后最小衰减系数
    fi_ag_gate_high: Optional[float] = None  # 方向C: AG gate 上限 (默认None=无上限)
    fi_cache_mode: str = "attention"      # 缓存什么特征:
    #   attention: cross-attention 输出 (语义对齐, 推荐)
    #   hidden: 完整 hidden_states (信息丰富但维度大)
    #   mlp: MLP 输出 (更高级语义)

    # ── 迭代优化参数 ──
    i_max: int = 10               # 迭代轮数

    # ── VLM ──
    vlm_provider: str = "local"
    vlm_model_path: str = "models/Qwen2.5-VL-7B-Instruct"

    # ── SVD 双向门控 (Floor + CAP) ──
    pna_std_gate: bool = True          # 门控开关 (opt-out: --no_pna_std_gate 关闭)
    pna_std_low: float = 0.32          # Floor: η_std 低于此值 → 抬升 α 到 floor_alpha
    pna_std_floor_alpha: float = 0.006 # floor 值: 弱信号时 α 至少为此值
    pna_std_high: float = 0.45         # CAP:  η_std 高于此值 → 降低 α 到 cap_alpha
    pna_std_cap_alpha: float = 0.002   # cap 值:  强信号时 α 至多为此值

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
            flags.append(f"blend(α={self.alpha})")
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

        # ── L1: Prompt 结构化分解 + CLIP 择优 ──
        if getattr(cfg, 'prompt_decompose', False):
            api_key = getattr(cfg, 'llm_api_key', '') or os.environ.get('LLM_API_KEY', '')
            if not api_key:
                logger.warning("  [PromptDecompose] 无 API key, 跳过 (set --llm_api_key or LLM_API_KEY)")
            else:
                try:
                    decomposer = create_prompt_decomposer(
                        api_key=api_key,
                        api_base=getattr(cfg, 'llm_api_base', 'https://token-plan-cn.xiaomimimo.com/v1'),
                        model=getattr(cfg, 'llm_model', 'mimo-v2.5-pro'),
                        device=self.device,
                    )
                    caption = decomposer.optimize(caption, video_path)
                    # 保存优化后的 prompt
                    (out / "optimized_prompt.txt").write_text(caption, encoding="utf-8")
                except Exception as e:
                    logger.warning(f"  [PromptDecompose] 失败: {e}, 使用原始 caption")

        # ── Step 3: 计算噪声先验 (如果启用) ──
        eta_temporal = None
        prompt_embeds = None
        ref_latents_enc = None
        ref_trajectory_from_inversion = None  # 合并模式：反演时同时缓存的轨迹
        fi_ref_features_from_inversion = None  # 合并模式：反演时同时缓存的FI特征
        svd_stats = None  # SVD 统计信息, 非SVD模式为None
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
                    cache_trajectory=need_trajectory,
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
                # 需要单独做反演缓存轨迹
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
                    ref_trajectory,
                    prompt_embeds if prompt_embeds is not None else self._encode_prompt(caption)
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
            latents = self._get_latents(
                eta_temporal, generator,
                svd_stats=svd_stats,
                fi_ref_features=fi_ref_features,
            )

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

        # ── PNA v4 诊断快照 (固定α策略) ──
        pna_diag = getattr(cfg, '_pna_diag', None)
        if pna_diag is not None:
            logger.info(
                f"  [SAMPLE-PNA-v4] ★★★ PNA-v4 DIAGNOSTIC FOR sample_{sample_id} ★★★"
            )
            logger.info(
                f"  [SAMPLE-PNA-v4] category={pna_diag['pna_category']}, "
                f"α_final={pna_diag['alpha_eff']:.6f}, "
                f"std_gate={'FLOOR' if pna_diag['std_gate_active'] else 'off'}"
            )
            logger.info(
                f"  [SAMPLE-PNA-v4] η_std={pna_diag.get('eta_std', 0):.4f}, "
                f"frame_cos={pna_diag.get('temporal_frame_cos', 0):.4f}"
            )
            # 预测: α vs fixed=0.004 的关系 → 后续对比XC验证
            alpha_diff = pna_diag['alpha_eff'] - 0.004
            if alpha_diff > 0.001:
                pred_tag = "α_UP→expect_WORSE(over-inject)"
            elif alpha_diff < -0.001:
                pred_tag = "α_DOWN→expect_BETTER(less-inject)"
            else:
                pred_tag = "α≈fixed→expect_SIMILAR"
            logger.info(f"  [SAMPLE-PNA] prediction: {pred_tag} (vs fixed_α=0.004)")
        return metadata

    # ─────────────────────────────────────────────────────────────
    # 内部方法：各改动点的实现
    # ─────────────────────────────────────────────────────────────
    # 方向3b: 运动方向一致性过滤
    # ─────────────────────────────────────────────────────────────

    def _apply_motion_filter(
        self, eta_temporal: torch.Tensor, Vh_m: torch.Tensor
    ) -> torch.Tensor:
        squeeze_batch = False
        if eta_temporal.dim() == 5:
            eta_temporal = eta_temporal[0]
            squeeze_batch = True

        C, F, H, W = eta_temporal.shape
        primary_pattern = Vh_m[0].abs()  # (F,) 主运动模式的帧重要性

        # 自适应阈值: 低于均值 50% 的帧视为噪声帧
        threshold = primary_pattern.mean() * 0.5
        frame_weight = torch.where(
            primary_pattern > threshold,
            torch.ones(F, device=eta_temporal.device, dtype=eta_temporal.dtype),
            primary_pattern / (threshold + 1e-8),
        ).clamp(0.0, 1.0)

        # 应用帧级权重
        eta_filtered = eta_temporal * frame_weight.view(1, F, 1, 1)

        suppressed = (frame_weight < 0.99).sum().item()
        logger.info(
            f"  [MotionFilter] primary_pattern std={primary_pattern.std():.4f}, "
            f"threshold={threshold:.4f}, suppressed={suppressed}/{F} frames, "
            f"η_std: {eta_temporal.std():.4f}→{eta_filtered.std():.4f}"
        )

        if squeeze_batch:
            eta_filtered = eta_filtered.unsqueeze(0)
        return eta_filtered

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
        if cache_trajectory:
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


            # 使用 return_stats 获取 S_temporal
            eta_temporal, svd_stats = svd_filter.filter(eta_inv, return_stats=True)

            logger.info(
                f"  [SVD] η_temporal std={eta_temporal.std():.4f}"
            )

            # ── 方向2: 渐进多尺度SVD ──
            if getattr(self.config, 'svd_progressive', False):
                eta_temporal_prog = svd_filter.filter_progressive(eta_inv)
                eta_temporal = eta_temporal_prog

            # ── 方向3b: 运动方向一致性过滤 ──
            if getattr(self.config, 'svd_motion_filter', False) and svd_stats is not None:
                Vh_m = svd_stats.get("Vh_temporal")
                if Vh_m is not None:
                    eta_temporal = self._apply_motion_filter(eta_temporal, Vh_m)
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
        fi_ref_features: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        L2: Noise Prior Blending

        混合公式: η = √α · η_temporal + √(1-α) · η_random

        其中:
            η_temporal: SVD Stage 2 提取的运动先验 (去外观保运动)
            η_random:   纯随机噪声

        v5 fixed 策略: 固定 α = cfg.alpha (默认 0.004)
        双向门控自动调节:
          Floor: η_std 过低 → 抬升 α 防止欠注入
          CAP:   η_std过高 → 降低 α 防止过注入
        opt-out: --no_pna_std_gate 关闭全部门控
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

        # ── Determine α (v5 fixed: 固定 α + 双向门控) ──
        cfg = self.config
        alpha = cfg.alpha
        # ── 双向门控: Floor(防欠注入) + CAP(防过注入) ──
        # opt-out: --no_pna_std_gate 即可关闭全部
        if getattr(cfg, 'pna_std_gate', False) and eta_temporal is not None:
            eta_std = eta_temporal.std().item()
            std_low = getattr(cfg, 'pna_std_low', 0.32)
            floor_alpha = getattr(cfg, 'pna_std_floor_alpha', 0.006)
            std_high = getattr(cfg, 'pna_std_high', 0.45)
            cap_alpha = getattr(cfg, 'pna_std_cap_alpha', 0.002)

            if eta_std < std_low and alpha < floor_alpha:
                # Floor: 弱信号 → 抬升 α 防欠注入
                logger.info(
                    f"  [SVD-Floor] η_std={eta_std:.4f} < {std_low}, "
                    f"α {alpha:.6f} → {floor_alpha:.6f}"
                )
                alpha = floor_alpha
            elif eta_std > std_high and alpha > cap_alpha:
                # CAP: 强信号 → 降低 α 防过注入
                logger.info(
                    f"  [SVD-CAP] η_std={eta_std:.4f} > {std_high}, "
                    f"α {alpha:.6f} → {cap_alpha:.6f}"
                )
                alpha = cap_alpha

        # ── 方向A: SVD联动跳过 ──
        # 当 mean_cos > skip_threshold 时, 不仅跳过 FI, 还强制关闭 SVD blend
        svd_skip = getattr(cfg, 'fi_quality_skip_svd', False)
        if svd_skip and eta_temporal is not None:
            mean_cos = self._compute_mean_cos(eta_temporal)
            skip_threshold = getattr(cfg, 'fi_quality_skip_threshold', None)
            if mean_cos is not None and skip_threshold is not None and mean_cos > skip_threshold:
                logger.info(
                    f"  [SVD-Skip] mean_cos={mean_cos:.4f} > skip_threshold={skip_threshold}, "
                    f"SVD blend DISABLED (α {alpha:.6f} → 0.0000)"
                )
                alpha = 0.0

        remaining = max(0.0, 1.0 - alpha)

        sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=self.device))
        sqrt_remaining = torch.sqrt(torch.tensor(remaining, device=self.device))

        # ── 方向5: 交替注入 (帧级 interleave) ──
        # 方向5 交替注入已注释 (9case 均值-1.2%, 对困难case有效但非普适)
        # use_alternate = getattr(cfg, 'svd_alternate', False)
        # if use_alternate and eta_temporal.dim() >= 4: ... (见 git history)
        # ── Two-way blend: η_temporal + η_random ──
        eta = sqrt_alpha * eta_temporal + sqrt_remaining * eta_random

        # ── 诊断: Blend 效果 ──
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

    def _compute_mean_cos(self, eta_temporal: torch.Tensor) -> Optional[float]:
        """计算 η_temporal 的帧间余弦相似度均值。返回 None 表示无法计算。"""
        eta_gate = eta_temporal
        if eta_gate.dim() == 5:
            if eta_gate.shape[2] > eta_gate.shape[1]:
                eta_gate = eta_gate.permute(0, 2, 1, 3, 4)
            num_frames_gate = eta_gate.shape[2]
        elif eta_gate.dim() == 4:
            num_frames_gate = eta_gate.shape[1]
            eta_gate = eta_gate.unsqueeze(0)
        else:
            return None

        if num_frames_gate < 2:
            return None

        frame_cos_sims = []
        for f in range(num_frames_gate - 1):
            f1 = eta_gate[0, :, f, :, :].flatten()
            f2 = eta_gate[0, :, f + 1, :, :].flatten()
            cos = torch.nn.functional.cosine_similarity(
                f1.unsqueeze(0), f2.unsqueeze(0)
            ).item()
            frame_cos_sims.append(cos)

        return sum(frame_cos_sims) / len(frame_cos_sims)

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
        mean_cos = self._compute_mean_cos(eta_temporal)
        if mean_cos is None:
            return 1.0

        # ── 硬跳过门控: 运动单调的样本不注入 ──
        # 实证发现: mean_cos > skip_threshold 的 case 注入后大概率受害
        # 因为高 mean_cos 表示 η_temporal 帧间高度相似, 携带的引导信号贫乏且冗余
        skip_threshold = getattr(self.config, 'fi_quality_skip_threshold', None)
        if skip_threshold is not None and mean_cos > skip_threshold:
            logger.info(
                f"  [Quality Scale] mean_cos={mean_cos:.4f} > skip_threshold={skip_threshold}, "
                f"FI SKIPPED (motion-too-coherent)"
            )
            return 0.0  # 完全跳过 FI

        # ── 软门控: sigmoid 映射 ──
        min_scale = self.config.fi_quality_min_scale
        threshold = self.config.fi_quality_threshold
        k = self.config.fi_quality_k
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

        # ── FI 独立门控: λ 由 fi_lambda + schedule + quality_scale 决定, 与 SVD α 完全解耦 ──
        logger.info(
            f"  [FI 独立门控] λ_max={cfg.fi_lambda:.4f}, "
            f"quality_gate={'ON' if cfg.fi_quality_gate else 'OFF'}, "
            f"adaptive_gate={'ON' if cfg.fi_adaptive_gate else 'OFF'}"
        )

        # ── 预计算 λ 调度 ──
        lambda_values = self._compute_schedule(
            num_steps, cfg.fi_lambda, cfg.fi_schedule
        )

        # ── 质量门控 ──
        quality_scale = 1.0
        if cfg.fi_quality_gate:
            quality_scale = self._compute_quality_scale(eta_temporal)
            if quality_scale < 1e-6:
                logger.info(f"  [FI] 质量门控 scale≈0, 跳过 FI, 走标准生成")
                return self._generate(prompt, latents, generator)

        logger.info(
            f"  [FI] λ_max={cfg.fi_lambda:.4f}, "
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

        # ── 方向B: Norm 硬上限衰减 ──
        max_norm = getattr(cfg, 'fi_max_injection_norm', None)
        norm_decay_min = getattr(cfg, 'fi_norm_decay_min', 0.3)
        running_norm = 0.0

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
                nonlocal running_norm
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
                    # ── 方向C: AG gate 上限 ──
                    gate_high = getattr(cfg, 'fi_ag_gate_high', None)
                    gate_raw = gate  # 保存原始gate用于日志
                    if gate_high is not None and gate > gate_high:
                        gate = gate_high
                    lam_eff = lam * gate

                # ── 方向B: Norm 硬上限衰减 ─_
                decay = 1.0
                norm_triggered = False
                if max_norm is not None and running_norm > max_norm:
                    decay = max(norm_decay_min, 1.0 - (running_norm - max_norm) / max_norm)
                    lam_eff *= decay
                    norm_triggered = True

                # v5: 门控统计 (每5步输出一次, info级别)
                    if step_idx_ref[0] % 5 == 0 and layer_idx == target_layers[0]:
                        parts = [
                            f"[FI step={step_idx_ref[0]}]",
                            f"L={layer_idx}",
                            f"cos={cos_sim:.4f}",
                        ]
                        # 方向C日志：gate截断
                        if gate_high is not None and gate_raw != gate:
                            parts.append(f"gate={gate_raw:.3f}→{gate:.3f}(cap)")
                        else:
                            parts.append(f"gate={gate:.3f}")
                        # 方向B日志：norm衰减
                        if norm_triggered:
                            parts.append(f"norm={running_norm:.0f}/{max_norm:.0f} decay={decay:.3f}")
                        parts.append(f"λ={lam:.5f}→{lam_eff:.5f}")
                        logger.info("    " + " ".join(parts))

                # 残差注入: h_injected = (1-λ_eff)*h_current + λ_eff*h_ref
                h_injected = (1.0 - lam_eff) * h_current + lam_eff * h_ref_dev

                # ── 方向3: 通道选择性FI ──
                sparse_ratio = getattr(cfg, 'fi_sparse_ratio', 0.0)
                if sparse_ratio > 0:
                    diff = h_ref_dev - h_current  # (B, N, D)
                    # 通道重要性 = |diff| 在 token 维度上的均值
                    ch_importance = diff.abs().mean(dim=1)  # (B, D)
                    top_k = max(1, int(ch_importance.shape[-1] * sparse_ratio))
                    _, top_idx = ch_importance.topk(top_k, dim=-1)  # (B, K)
                    mask = torch.zeros_like(h_current)
                    mask.scatter_(-1, top_idx.unsqueeze(1).expand(-1, h_current.shape[1], -1), 1.0)
                    h_injected = h_current + lam_eff * mask * diff

                # 记录注入统计
                injection_norm = (h_injected - h_current).norm().item()
                injection_stats_per_step[0][layer_idx] = injection_norm
                if max_norm is not None:
                    running_norm += injection_norm

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
            f"    λ_max={cfg.fi_lambda:.4f} (独立门控)"
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


