#!/usr/bin/env python3
"""
P-Flow Runner - 通过命令行 flag 控制各改动点。

用法:
    # 纯 baseline (caption + 一次生成)
    python run.py --data_dir data/videos --caption_dir data/captions_qwen

    # L2: 启用噪声先验 (inversion + svd + blend)
    python run.py --data_dir data/videos --caption_dir data/captions_qwen --inversion --svd --blend --alpha 0.004

    # L2+L3: 噪声先验 + Feature Injection (当前最强配置)
    python run.py --data_dir data/videos --caption_dir data/captions_qwen \
        --inversion --svd --blend --alpha 0.004 --svd_mode v1 \
        --feature_inject --fi_layers mid --fi_lambda 0.05

改动点 Flag 说明:
    --inversion       L2: 启用 Flow Matching Inversion (从参考视频反演噪声)
    --svd             L2: 启用 SVD 两阶段滤波 (空间去内容 + 时间保运动)
    --blend           L2: 启用噪声混合 (η = sqrt(α)*η_temporal + sqrt(1-α)*η_random)
    --feature_inject  L3: 启用 Feature Injection (DiT特征空间注入)
    --iter N          L1: 启用迭代VLM优化 (N轮反馈循环)

快捷组合:
    --noise_prior  等价于 --inversion --svd --blend
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
    p.add_argument("--full", action="store_true", help="快捷: --inversion --svd --blend --feature_inject")

    # ── 参数调节 ──
    p.add_argument("--alpha", type=float, default=0.004, help="SVD 噪声混合权重 (v5 fixed, 推荐 0.004)")
    p.add_argument("--rho_s", type=float, default=0.1, help="空间SVD阈值")
    p.add_argument("--rho_m", type=float, default=0.9, help="时间SVD阈值")
    p.add_argument("--steps", type=int, default=30, help="推理步数")
    p.add_argument("--guidance", type=float, default=5.0, help="CFG scale")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--inversion_steps", type=int, default=50, help="反演ODE步数 (30=快速, 50=标准)")
    p.add_argument("--no_fast_svd", action="store_true", help="禁用 randomized SVD (使用精确SVD)")
    p.add_argument("--svd_motion_filter", action="store_true", help="方向3b: 运动方向一致性过滤")
    p.add_argument("--svd_alternate", action="store_true", help="方向5: 交替注入 (帧级 temporal/random 交替)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=15)

    # ── [已废弃] 旧方案参数已全部移除，详见 pipeline.py PFlowConfig ──

    # ── L3 V3: Feature Injection (FI) ──
    p.add_argument("--feature_inject", action="store_true",
                   help="启用 Feature Injection: 在 DiT 特征空间注入参考信息 (独立于 VDA)")
    p.add_argument("--fi_layers", type=str, default="mid",
                   help="FI 注入层: 'all'/'mid'/'last'/逗号分隔层号 (默认 mid=中间1/3层)")
    p.add_argument("--fi_lambda", type=float, default=0.1,
                   help="FI 注入强度 λ (推荐 0.01~0.3; 0=无注入, 1=完全替换)")
    p.add_argument("--fi_schedule", type=str, default="middle_peak",
                   choices=["constant", "middle_peak", "warmup_decay", "cosine_decay"],
                   help="FI λ 调度策略")
    p.add_argument("--fi_no_quality_gate", action="store_true",
                   help="禁用 FI 质量门控")
    p.add_argument("--fi_quality_threshold", type=float, default=0.05,
                   help="FI 质量门控阈值 (mean_cos < threshold → 弱引导; 默认 0.05; 设为 0.00 可基本跳过门控)")
    p.add_argument("--fi_quality_k", type=float, default=20.0,
                   help="FI 质量门控 sigmoid 斜率 (越大越陡峭; 默认 20.0)")
    p.add_argument("--fi_quality_min_scale", type=float, default=0.1,
                   help="FI 质量门控最低注入比例 (默认 0.1, 即最少保留 10%% 注入)")
    p.add_argument("--fi_quality_skip_threshold", type=float, default=None,
                   help="FI 硬跳过阈值: mean_cos > 此值时完全跳过 FI (默认 None 不启用; 推荐 0.08)")
    p.add_argument("--fi_quality_skip_svd", action="store_true",
                   help="方向A: 跳过FI时同时关闭SVD blend (默认False)")
    p.add_argument("--fi_max_injection_norm", type=float, default=None,
                   help="方向B: Total injection norm 硬上限 (默认 None 不启用; 推荐 10000)")
    p.add_argument("--fi_norm_decay_min", type=float, default=0.3,
                   help="方向B: norm超限后最小衰减系数 (默认 0.3)")
    p.add_argument("--fi_ag_gate_high", type=float, default=None,
                   help="方向C: AG gate 上限 (默认 None 无上限; 推荐 0.40)")
    p.add_argument("--fi_cache_mode", type=str, default="attention",
                   choices=["attention", "hidden", "mlp"],
                   help="FI 缓存特征类型: attention=cross-attn输出(推荐), hidden=block输出, mlp=ffn输出")
    p.add_argument("--fi_no_adaptive_gate", action="store_true",
                   help="禁用 FI 自适应门控 (默认开启: 特征越接近参考→注入越少)")
    p.add_argument("--fi_adaptive_temp", type=float, default=5.0,
                   help="FI 自适应门控温度 (越大越敏感, 推荐 3~10; 默认 5.0)")

    # ── SVD 双向门控 (Floor + CAP) ──
    p.add_argument("--no_pna_std_gate", action="store_true",
                   help="禁用双向门控 (默认启用: Floor+CAP)")
    p.add_argument("--pna_std_low", type=float, default=0.32,
                   help="Floor 低阈值: η_std < 此值 → 抬升 α (默认 0.32)")
    p.add_argument("--pna_std_floor_alpha", type=float, default=0.006,
                   help="Floor 值: 弱信号时 α 至少为此值 (默认 0.006)")
    p.add_argument("--pna_std_high", type=float, default=0.45,
                   help="CAP 高阈值: η_std > 此值 → 降低 α (默认 0.45)")
    p.add_argument("--pna_std_cap_alpha", type=float, default=0.002,
                   help="CAP 值: 强信号时 α 至多为此值 (默认 0.002)")

    # ── 模型路径 ──
    p.add_argument("--model_path", type=str, default="models/Wan2.1-T2V-1.3B-Diffusers",
                   help="Wan2.1 T2V 模型路径 (默认: 项目内 models/ 目录)")
    p.add_argument("--vlm_path", type=str, default="models/Qwen2.5-VL-7B-Instruct",
                   help="VLM 模型路径 (默认: 项目内 models/ 目录)")
    p.add_argument("--vlm_provider", type=str, default="local", choices=["local", "dashscope", "mock"])

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
        args.feature_inject = True

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
        svd_motion_filter=args.svd_motion_filter,
        svd_alternate=args.svd_alternate,
        i_max=args.iter if args.iter > 0 else 1,
        vlm_provider=args.vlm_provider,
        vlm_model_path=args.vlm_path,
                   seed=args.seed,
        # L3: Feature Injection
        feature_inject=args.feature_inject,
        fi_layers=args.fi_layers,
        fi_lambda=args.fi_lambda,
        fi_schedule=args.fi_schedule,
        fi_quality_gate=not args.fi_no_quality_gate,
        fi_quality_threshold=args.fi_quality_threshold,
        fi_quality_k=args.fi_quality_k,
        fi_quality_min_scale=args.fi_quality_min_scale,
        fi_quality_skip_threshold=args.fi_quality_skip_threshold,
        fi_cache_mode=args.fi_cache_mode,
        fi_adaptive_gate=not args.fi_no_adaptive_gate,
        fi_adaptive_temp=args.fi_adaptive_temp,
        # 方向A/B/C 新增参数
        fi_quality_skip_svd=args.fi_quality_skip_svd,
        fi_max_injection_norm=args.fi_max_injection_norm,
        fi_norm_decay_min=args.fi_norm_decay_min,
        fi_ag_gate_high=args.fi_ag_gate_high,
        # SVD 双向门控 (Floor + CAP)
        pna_std_gate=not args.no_pna_std_gate,
        pna_std_low=args.pna_std_low,
        pna_std_floor_alpha=args.pna_std_floor_alpha,
        pna_std_high=args.pna_std_high,
        pna_std_cap_alpha=args.pna_std_cap_alpha,
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
        print(f"  alpha={config.alpha}, rho_s={config.rho_s}, rho_m={config.rho_m}")
    if config.feature_inject:
        fi_adapt_str = f", adaptive(temp={config.fi_adaptive_temp})" if config.fi_adaptive_gate else ""
        print(f"  [FI] λ={config.fi_lambda}, layers={config.fi_layers}, schedule={config.fi_schedule}, mode={config.fi_cache_mode}{fi_adapt_str}")
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
