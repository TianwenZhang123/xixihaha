#!/usr/bin/env python3
"""
P-Flow Runner - 通过命令行 flag 控制各改动点。

用法:
    # 纯 baseline (caption + 一次生成)
    python run.py --video /path/to/ref.mp4 --caption "a cat walking"

    # 启用噪声先验 (inversion + svd + blend)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend

    # 启用迭代优化 (10轮VLM反馈)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --iter 10

    # 完整 P-Flow (噪声先验 + 迭代优化)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend --iter 10 --composite

    # 用中点法替代Euler
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend --midpoint

    # 批量处理
    python run.py --data_dir /path/to/videos --caption_dir /path/to/captions --inversion --svd --blend --iter 10

改动点 Flag 说明:
    --inversion      启用 Flow Matching Inversion (从参考视频反演噪声)
    --svd            启用 SVD 两阶段滤波 (空间去内容 + 时间保运动)
    --blend          启用噪声混合 (η = sqrt(α)*η_temporal + sqrt(1-α)*η_random)
    --iter N         启用迭代VLM优化 (N轮反馈循环)
    --midpoint       使用二阶中点法ODE求解器 (替代默认Euler)
    --composite      启用三面板垂直拼接 (ref|prev|current 送VLM对比)

快捷组合:
    --noise_prior  等价于 --inversion --svd --blend
    --full         等价于 --inversion --svd --blend --iter 10 --composite
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import PFlowPipeline, PFlowConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="P-Flow: 通过 flag 控制各改动点的视频生成管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 输入 ──
    p.add_argument("--video", type=str, help="单个参考视频路径")
    p.add_argument("--caption", type=str, default="", help="初始 caption")
    p.add_argument("--data_dir", type=str, help="批量模式: 视频目录")
    p.add_argument("--caption_dir", type=str, help="批量模式: caption目录")
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--sample_ids", type=int, nargs="+", help="指定样本ID")
    p.add_argument("--limit", type=int, help="最多处理N个样本")

    # ── 改动点开关 ──
    p.add_argument("--inversion", action="store_true", help="启用 Flow Matching Inversion")
    p.add_argument("--svd", action="store_true", help="启用 SVD 滤波")
    p.add_argument("--blend", action="store_true", help="启用噪声混合")
    p.add_argument("--iter", type=int, default=0, help="迭代轮数 (0=不迭代)")
    p.add_argument("--midpoint", action="store_true", help="使用中点法ODE求解器")
    p.add_argument("--composite", action="store_true", help="启用垂直拼接对比")

    # ── 快捷组合 ──
    p.add_argument("--noise_prior", action="store_true", help="快捷: --inversion --svd --blend")
    p.add_argument("--full", action="store_true", help="快捷: 全部启用 (iter=10)")

    # ── 参数调节 ──
    p.add_argument("--alpha", type=float, default=0.003, help="噪声混合权重 (推荐 0.001~0.01, P-Flow论文用 0.001)")
    p.add_argument("--rho_s", type=float, default=0.1, help="空间SVD阈值")
    p.add_argument("--rho_m", type=float, default=0.9, help="时间SVD阈值")
    p.add_argument("--steps", type=int, default=30, help="推理步数")
    p.add_argument("--guidance", type=float, default=5.0, help="CFG scale")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--inversion_steps", type=int, default=50, help="反演ODE步数 (30=快速, 50=标准)")
    p.add_argument("--no_fast_svd", action="store_true", help="禁用 randomized SVD (使用精确SVD)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=15)

    # ── Quality-Gated Alpha (方案 B) ──
    p.add_argument("--quality_gated_alpha", action="store_true",
                   help="启用 per-sample adaptive alpha (根据SVD方向质量动态调节注入量)")
    p.add_argument("--qga_base_alpha", type=float, default=0.004,
                   help="Quality-Gated Alpha 的基础 alpha")
    p.add_argument("--qga_low_mult", type=float, default=0.25,
                   help="quality=0 时的 alpha 倍率")
    p.add_argument("--qga_high_mult", type=float, default=2.5,
                   help="quality=1 时的 alpha 倍率")

    # ── 方向 C: 频域噪声重塑 (叠加在 alpha blend 上) ──
    p.add_argument("--freq_reshape", action="store_true",
                   help="启用频域噪声重塑 (对 η_random 预处理后再 alpha 混合, 需配合 --blend --alpha)")
    p.add_argument("--freq_reshape_beta", type=float, default=1.0,
                   help="频域重塑强度: 0=不重塑(纯随机), 1=完全匹配频谱 (推荐0.5~1.0)")

    # ── 方向 D: Std-Gated Adaptive Alpha (SGA) ──
    p.add_argument("--adaptive_alpha", action="store_true",
                   help="启用 per-sample adaptive alpha (根据 η_temporal std 动态调节注入量)")
    p.add_argument("--sga_target_std", type=float, default=0.30,
                   help="SGA 目标标准差: effective_alpha = alpha * (target_std / actual_std)")
    p.add_argument("--sga_alpha_min", type=float, default=0.001,
                   help="SGA alpha 下界 (防止完全不注入)")
    p.add_argument("--sga_alpha_max", type=float, default=0.010,
                   help="SGA alpha 上界 (防止过度注入导致 catastrophic failure)")

    # ── 方向 E: Prompt-Orthogonal Decomposition Injection (PODI) ──
    p.add_argument("--podi", action="store_true",
                   help="启用 PODI: 只注入 η_temporal 中与 prompt 语义对齐的分量，过滤正交/冲突信号")
    p.add_argument("--podi_alpha", type=float, default=0.004,
                   help="PODI 注入强度 (默认与 baseline alpha 对齐做公平对比; 后续可尝试更大值 0.008~0.02)")
    p.add_argument("--podi_min_alignment", type=float, default=0.01,
                   help="最小对齐度阈值: alignment < 此值则放弃注入 (推荐 0.01~0.05)")
    p.add_argument("--podi_proj_mode", type=str, default="mean_pool",
                   choices=["mean_pool", "last_token", "weighted"],
                   help="prompt embedding → latent 的投影方式 (mean_pool/last_token/weighted)")

    # ── 方向 F: Channel-Energy Gated Injection (CEGI) ──
    p.add_argument("--cegi", action="store_true",
                   help="启用 CEGI: 只在 temporal energy 最高的 top-k channel 集中注入 prior")
    p.add_argument("--cegi_top_k", type=int, default=4,
                   help="注入的 channel 数量 (默认 4/16=25%%; 推荐搜索 2~8)")
    p.add_argument("--cegi_alpha", type=float, default=0.02,
                   help="选中 channel 的注入强度 (默认 0.02, 是 baseline 的 5x; 推荐搜索 0.01~0.05)")
    p.add_argument("--cegi_residual_alpha", type=float, default=0.0,
                   help="未选中 channel 的注入强度 (默认 0=纯随机; 可设 0.002 保留微弱 prior)")

    # ── 方向 G: Multi-Scale Temporal Decomposition Injection (MSTDI) ──
    p.add_argument("--mstdi", action="store_true",
                   help="启用 MSTDI: 多尺度时序分解注入, 低频集中注入 temporal prior")
    p.add_argument("--mstdi_levels", type=int, default=3,
                   help="金字塔层数 (默认 3: 1/4, 1/2, 原始; 推荐 2~4)")
    p.add_argument("--mstdi_alpha_base", type=float, default=0.05,
                   help="最粗尺度 alpha (默认 0.05; 推荐 0.02~0.10)")
    p.add_argument("--mstdi_alpha_decay", type=float, default=0.25,
                   help="每层 alpha 衰减倍率 (默认 0.25: L0=0.05 → L1=0.0125 → L2≈0.003)")

    # ── 方向 H: Temporal Phase Injection (TPI) ──
    p.add_argument("--tpi", action="store_true",
                   help="启用 TPI: 只注入 temporal 的时间相位, 保留 random 幅度")
    p.add_argument("--tpi_gamma", type=float, default=0.5,
                   help="相位插值强度 (0=纯random, 1=纯temporal; 默认 0.5; 推荐 0.3~0.7)")
    p.add_argument("--tpi_freq_min", type=int, default=1,
                   help="注入起始频率 bin (默认 1 跳过 DC; 0=包括 DC)")
    p.add_argument("--tpi_freq_max", type=int, default=-1,
                   help="注入结束频率 bin (默认 -1=所有; 可设 5 只注入低频)")

    # ── 方向 I: Orthogonal Complement Suppression (OCS) ──
    p.add_argument("--ocs", action="store_true",
                   help="启用 OCS: 抑制 η_random 中与 temporal 主方向正交的分量")
    p.add_argument("--ocs_top_k", type=int, default=3,
                   help="SVD 保留的主成分数 (默认 3; 推荐 2~5)")
    p.add_argument("--ocs_suppress_ratio", type=float, default=0.5,
                   help="正交补抑制比例 (0=不抑制, 1=完全去除; 默认 0.5; 推荐 0.3~0.7)")

    # ── 灰盒: Latent Trajectory Soft Anchor (旧方案) ──
    p.add_argument("--trajectory_anchor", action="store_true",
                   help="启用旧方案轨迹锚定 (position lerp, 已证明失败, 保留用于对比)")
    p.add_argument("--anchor_beta_max", type=float, default=0.3,
                   help="最大锚定强度 β_max (推荐搜索 0.1~0.5; 0.3 为中等强度)")
    p.add_argument("--anchor_schedule", type=str, default="cosine_decay",
                   choices=["cosine_decay", "linear_decay", "constant", "warmup_decay"],
                   help="β 退火策略: 控制锚定力度如何随去噪步骤衰减")
    p.add_argument("--anchor_cache_every_n", type=int, default=1,
                   help="inversion 轨迹缓存间隔 (1=全缓存; 2=隔一步; 用于节省显存)")
    p.add_argument("--no_anchor_quality_gate", action="store_true",
                   help="禁用轨迹质量门控 (默认启用)")
    p.add_argument("--anchor_quality_threshold", type=float, default=0.05,
                   help="η_temporal 帧间 cos 阈值: mean_cos < 此值则跳过 anchor (默认 0.05)")
    p.add_argument("--anchor_cos_threshold", type=float, default=0.2,
                   help=">0 时启用 cos-proportional β 模式 (旧方案2; 设 0 禁用)")

    # ── L3 V2: Velocity Direction Anchor (VDA) ──
    p.add_argument("--velocity_anchor", action="store_true",
                   help="启用 VDA (Velocity Direction Anchor): 用参考轨迹速度方向微调去噪过程, 不要求起点对齐")
    p.add_argument("--vda_mode", type=str, default="v1",
                   choices=["v1", "v2"],
                   help="VDA 版本: v1=原始(motion_coherence gate), v2=角度自适应(angle gate, 推荐)")
    p.add_argument("--vda_gamma", type=float, default=0.03,
                   help="VDA 方向引导强度 γ (推荐搜索 0.01~0.10; 越大参考速度影响越强)")
    p.add_argument("--vda_schedule", type=str, default="middle_peak",
                   choices=["constant", "middle_peak", "warmup_decay", "cosine_decay"],
                   help="γ 调度策略: middle_peak=中间最强 (推荐), constant=恒定, cosine_decay=前强后弱")
    p.add_argument("--vda_no_perp_only", action="store_true",
                   help="禁用'只注入正交分量'模式, 改为混合注入 (正交+微弱平行分量)")
    p.add_argument("--vda_parallel_weight", type=float, default=0.1,
                   help="vda_no_perp_only 时平行分量权重 (默认 0.1=微弱; 推荐 0.05~0.2)")
    p.add_argument("--vda_no_quality_gate", action="store_true",
                   help="禁用 VDA 质量门控")
    p.add_argument("--vda_hard_gate", action="store_true",
                   help="VDA 使用硬门控 (低于阈值完全跳过), 默认为软门控 (sigmoid 缩放)")
    p.add_argument("--vda_angle_threshold", type=float, default=110.0,
                   help="VDA v2 角度自适应阈值 (度): angle(v_ref,v_gen)>此值时降权 (推荐 100~120)")
    p.add_argument("--vda_norm_clamp", type=float, default=0.0,
                   help="每步 Δz 范数上限 = clamp * ‖z‖ (推荐 0.05~0.10; 0=不限制)")
    p.add_argument("--vda_start_step", type=int, default=1,
                   help="VDA 起始步 (默认 1=跳过第一步; 0=从第一步开始)")
    p.add_argument("--vda_end_step", type=int, default=-1,
                   help="VDA 结束步 (默认 -1=到最后一步)")

    # ── 模型路径 ──
    p.add_argument("--model_path", type=str, default="models/Wan2.1-T2V-1.3B-Diffusers",
                   help="Wan2.1 T2V 模型路径 (默认: 项目内 models/ 目录)")
    p.add_argument("--vlm_path", type=str, default="models/Qwen2.5-VL-7B-Instruct",
                   help="VLM 模型路径 (默认: 项目内 models/ 目录)")
    p.add_argument("--vlm_provider", type=str, default="local", choices=["local", "dashscope", "mock"])

    # ── 负面 Prompt ──
    p.add_argument("--negative_prompt", type=str, default="",
                   help="全局自定义负面 prompt (替代默认硬编码)")
    p.add_argument("--negative_prompt_dir", type=str, default="",
                   help="按样本加载负面 prompt 的目录 (含 {id}.txt, 优先级高于 --negative_prompt)")

    # ── 执行控制 ──
    p.add_argument("--resume", action="store_true", help="跳过已有输出")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def build_config(args) -> PFlowConfig:
    """从命令行参数构建配置。"""
    # 处理快捷组合
    if args.full:
        args.inversion = True
        args.svd = True
        args.blend = True
        args.composite = True
        if args.iter == 0:
            args.iter = 10

    if args.noise_prior:
        args.inversion = True
        args.svd = True
        args.blend = True

    return PFlowConfig(
        t2v_path=args.model_path,
        dtype="bfloat16",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        use_inversion=args.inversion,
        use_svd=args.svd,
        use_blend=args.blend,
        use_iter=args.iter > 0,
        use_midpoint=args.midpoint,
        use_composite=args.composite,
        alpha=args.alpha,
        rho_s=args.rho_s,
        rho_m=args.rho_m,
        inversion_steps=args.inversion_steps,
        use_fast_svd=not args.no_fast_svd,
        i_max=args.iter if args.iter > 0 else 1,
        vlm_provider=args.vlm_provider,
        vlm_model_path=args.vlm_path,
        negative_prompt=args.negative_prompt,
        negative_prompt_file=args.negative_prompt_dir,
        seed=args.seed,
        # Quality-Gated Alpha
        quality_gated_alpha=args.quality_gated_alpha,
        qga_base_alpha=args.qga_base_alpha,
        qga_low_mult=args.qga_low_mult,
        qga_high_mult=args.qga_high_mult,
        # 方向 C: 频域噪声重塑
        freq_reshape=args.freq_reshape,
        freq_reshape_beta=args.freq_reshape_beta,
        # 方向 D: Std-Gated Adaptive Alpha
        adaptive_alpha=args.adaptive_alpha,
        sga_target_std=args.sga_target_std,
        sga_alpha_min=args.sga_alpha_min,
        sga_alpha_max=args.sga_alpha_max,
        # 方向 E: PODI
        podi=args.podi,
        podi_alpha=args.podi_alpha,
        podi_min_alignment=args.podi_min_alignment,
        podi_proj_mode=args.podi_proj_mode,
        # 方向 F: CEGI
        cegi=args.cegi,
        cegi_top_k=args.cegi_top_k,
        cegi_alpha=args.cegi_alpha,
        cegi_residual_alpha=args.cegi_residual_alpha,
        # 方向 G: MSTDI
        mstdi=args.mstdi,
        mstdi_levels=args.mstdi_levels,
        mstdi_alpha_base=args.mstdi_alpha_base,
        mstdi_alpha_decay=args.mstdi_alpha_decay,
        # 方向 H: TPI
        tpi=args.tpi,
        tpi_gamma=args.tpi_gamma,
        tpi_freq_min=args.tpi_freq_min,
        tpi_freq_max=args.tpi_freq_max,
        # 方向 I: OCS
        ocs=args.ocs,
        ocs_top_k=args.ocs_top_k,
        ocs_suppress_ratio=args.ocs_suppress_ratio,
        # 灰盒: Trajectory Anchor
        trajectory_anchor=args.trajectory_anchor,
        anchor_beta_max=args.anchor_beta_max,
        anchor_schedule=args.anchor_schedule,
        anchor_cache_every_n=args.anchor_cache_every_n,
        anchor_quality_gate=not args.no_anchor_quality_gate,
        anchor_quality_threshold=args.anchor_quality_threshold,
        anchor_cos_threshold=args.anchor_cos_threshold,
        # L3 V2: Velocity Direction Anchor
        velocity_anchor=args.velocity_anchor,
        vda_mode=args.vda_mode,
        vda_gamma=args.vda_gamma,
        vda_schedule=args.vda_schedule,
        vda_use_perp_only=not args.vda_no_perp_only,
        vda_parallel_weight=args.vda_parallel_weight,
        vda_quality_gate=not args.vda_no_quality_gate,
        vda_quality_scale=not args.vda_hard_gate,
        vda_angle_threshold=args.vda_angle_threshold,
        vda_norm_clamp=args.vda_norm_clamp,
        vda_start_step=args.vda_start_step,
        vda_end_step=args.vda_end_step,
    )


def run_single(pipeline, args):
    """单视频模式。"""
    if not args.video:
        print("错误: 单视频模式需要 --video 参数")
        sys.exit(1)

    result = pipeline.run(
        video_path=args.video,
        output_dir=args.output_dir,
        caption=args.caption,
        sample_id=0,
    )
    print(f"\n完成: {result['output']}")
    print(f"  实验: {result['experiment']}")
    print(f"  耗时: {result['time_seconds']:.1f}s")
    print(f"  flags: {result['flags']}")


def run_batch(pipeline, args):
    """批量模式。"""
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)

    # 发现样本
    videos = sorted(data_path.glob("*.mp4"), key=lambda p: int(p.stem))
    if args.sample_ids:
        id_set = set(args.sample_ids)
        videos = [v for v in videos if int(v.stem) in id_set]
    if args.limit:
        videos = videos[:args.limit]

    print(f"找到 {len(videos)} 个样本")
    print(f"实验: {pipeline.config.experiment_name()}")
    print(f"Flags: {pipeline.config.active_flags() or ['baseline']}")
    print()

    for idx, vp in enumerate(videos, 1):
        sample_id = int(vp.stem)
        sample_out = Path(args.output_dir) / f"sample_{sample_id}"

        # Resume
        if args.resume and (sample_out / f"{sample_id}.mp4").exists():
            print(f"  [{idx}/{len(videos)}] 跳过 {sample_id} (已存在)")
            continue

        # 加载 caption
        caption = ""
        if args.caption_dir:
            cap_file = Path(args.caption_dir) / f"{sample_id}.txt"
            if cap_file.exists():
                caption = cap_file.read_text(encoding="utf-8").strip()

        print(f"  [{idx}/{len(videos)}] 处理 {sample_id}...")
        pipeline.run(
            video_path=str(vp),
            output_dir=str(sample_out),
            caption=caption,
            sample_id=sample_id,
        )

    print(f"\n批量完成! 输出: {args.output_dir}")


def main():
    args = parse_args()

    # ── 日志配置: 同时输出到终端 + 自动保存到文件 ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    # 确保输出目录存在
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_log.txt"

    # 设置 root logger: 终端 + 文件双输出
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)

    # 终端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler (追加模式，多次运行日志累积)
    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"日志将保存到: {log_file.resolve()}")

    config = build_config(args)

    # 打印配置摘要
    flags = config.active_flags()
    print(f"P-Flow | {config.experiment_name()}")
    print(f"  Flags: {flags or ['baseline (无改动)']}")
    if config.use_blend:
        if config.adaptive_alpha:
            print(f"  [SGA] adaptive alpha: base={config.alpha}, target_std={config.sga_target_std}, "
                  f"range=[{config.sga_alpha_min}, {config.sga_alpha_max}]")
            print(f"  rho_s={config.rho_s}, rho_m={config.rho_m}")
        elif config.freq_reshape:
            print(f"  η_random 频域重塑(β={config.freq_reshape_beta}) + alpha blend(α={config.alpha})")
            print(f"  rho_s={config.rho_s}, rho_m={config.rho_m}")
        else:
            print(f"  alpha={config.alpha}, rho_s={config.rho_s}, rho_m={config.rho_m}")
    if config.trajectory_anchor:
        print(f"  [Trajectory Anchor] β_max={config.anchor_beta_max}, "
              f"schedule={config.anchor_schedule}, cache_every_n={config.anchor_cache_every_n}")
    if config.velocity_anchor:
        vda_info = f"  [VDA {config.vda_mode}] γ={config.vda_gamma}, schedule={config.vda_schedule}"
        if config.vda_mode == "v2":
            vda_info += f", angle_threshold={config.vda_angle_threshold}°"
        print(vda_info)
    if config.use_iter:
        print(f"  iterations={config.i_max}")
    print()

    pipeline = PFlowPipeline(config)

    if args.data_dir:
        run_batch(pipeline, args)
    else:
        run_single(pipeline, args)


if __name__ == "__main__":
    main()
