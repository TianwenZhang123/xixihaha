#!/usr/bin/env python3
"""
P-Flow Runner — 默认使用 configs/default.toml 中的最优配置。

用法:
    # 最优配置 (ABC + 自适应渐进 SVD)
    python run.py --data_dir data/videos --caption_dir data/captions_qwen

    # 覆盖某个参数
    python run.py --data_dir data/videos --caption_dir data/captions_qwen --no-svd

    # 纯 baseline
    python run.py --data_dir data/videos --caption_dir data/captions_qwen --no-svd

    # 指定输出目录
    python run.py --data_dir data/videos --caption_dir data/captions_qwen --output_dir outputs/my_exp

配置文件: configs/default.toml
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import PFlowPipeline, PFlowConfig


def _load_config() -> dict:
    """从 configs/default.toml 加载默认配置。"""
    cfg_path = Path(__file__).parent / "configs" / "default.toml"
    if not cfg_path.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except ImportError:
        try:
            # Python < 3.11 使用 tomli
            import tomli
            return tomli.loads(cfg_path.read_text(encoding="utf-8"))
        except ImportError:
            import json
            raise RuntimeError("需要 tomli 包: pip install tomli")


def _cfg(cfg: dict, *keys, default=None):
    """安全地从嵌套字典取值."""
    for k in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(k, {})
        else:
            return default
    return cfg if cfg != {} else default


def parse_args():
    """解析命令行参数，默认值从 configs/default.toml 加载。"""
    cfg = _load_config()

    p = argparse.ArgumentParser(
        description="P-Flow — 默认使用 configs/default.toml 最优配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 输入 (不设默认值, 要求显式传入) ──
    p.add_argument("--video", type=str, help="单个参考视频路径")
    p.add_argument("--caption", type=str, default="", help="初始 caption")
    p.add_argument("--data_dir", type=str, help="批量模式: 视频目录")
    p.add_argument("--caption_dir", type=str, help="批量模式: caption目录")
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--sample_ids", type=int, nargs="+", help="指定样本ID")
    p.add_argument("--limit", type=int, help="最多处理N个样本")

    # ── 基础参数 ──
    p.add_argument("--model_path", type=str,
                   default=_cfg(cfg, "paths", "model_path", default="models/Wan2.1-T2V-1.3B-Diffusers"))
    p.add_argument("--vlm_path", type=str,
                   default=_cfg(cfg, "paths", "vlm_path", default="models/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--vlm_provider", type=str,
                   default=_cfg(cfg, "paths", "vlm_provider", default="local"),
                   choices=["local", "dashscope", "mock"])
    p.add_argument("--height", type=int, default=_cfg(cfg, "video", "height", default=480))
    p.add_argument("--width", type=int, default=_cfg(cfg, "video", "width", default=832))
    p.add_argument("--num_frames", type=int, default=_cfg(cfg, "video", "num_frames", default=81))
    p.add_argument("--fps", type=int, default=_cfg(cfg, "video", "fps", default=15))
    p.add_argument("--steps", type=int, default=_cfg(cfg, "inference", "steps", default=30))
    p.add_argument("--guidance", type=float, default=_cfg(cfg, "inference", "guidance", default=5.0))
    p.add_argument("--seed", type=int, default=_cfg(cfg, "inference", "seed", default=42))
    p.add_argument("--inversion_steps", type=int,
                   default=_cfg(cfg, "inference", "inversion_steps", default=50))
    p.add_argument("--no_fast_svd", action="store_true",
                   help=f"禁用 randomized SVD (默认: {_cfg(cfg, 'inference', 'use_fast_svd', default=True)})")
    p.add_argument("--iter", type=int, default=0, help="迭代轮数 (0=不迭代)")
    p.add_argument("--composite", action="store_true", help="启用垂直拼接对比")

    # ── L2: 噪声先验 ──
    p.add_argument("--svd", action="store_true", dest="svd",
                   default=_cfg(cfg, "noise_prior", "svd", default=False))
    p.add_argument("--no-svd", action="store_false", dest="svd",
                   help="禁用 SVD (含 inversion + two-stage SVD + blend)")
    p.add_argument("--alpha", type=float,
                   default=_cfg(cfg, "noise_prior", "alpha", default=0.004))
    p.add_argument("--rho_s", type=float,
                   default=_cfg(cfg, "noise_prior", "rho_s", default=0.1))
    p.add_argument("--rho_m", type=float,
                   default=_cfg(cfg, "noise_prior", "rho_m", default=0.9))

    # ── L2: sigmoid 自适应 α ──
    p.add_argument("--no-pna-std-gate", action="store_true",
                   help="禁用 sigmoid 门控")
    p.add_argument("--pna_std_eta0", type=float,
                   default=_cfg(cfg, "std_gate", "eta0", default=0.38))
    p.add_argument("--pna_std_kappa", type=float,
                   default=_cfg(cfg, "std_gate", "kappa", default=20.0))

    # ── L2: 渐进 SVD ──
    p.add_argument("--svd-progressive", action="store_true", dest="svd_progressive",
                   default=_cfg(cfg, "progressive_svd", "enabled", default=False))
    p.add_argument("--no-svd-progressive", action="store_false", dest="svd_progressive",
                   help="禁用渐进多尺度 SVD")

    # ── L3: FI ──
    p.add_argument("--feature-inject", action="store_true", dest="feature_inject",
                   default=_cfg(cfg, "fi", "enabled", default=False))
    p.add_argument("--no-feature-inject", action="store_false", dest="feature_inject",
                   help="禁用 Feature Injection")
    p.add_argument("--fi_layers", type=str,
                   default=_cfg(cfg, "fi", "layers", default="mid"))
    p.add_argument("--fi_lambda", type=float,
                   default=_cfg(cfg, "fi", "lambda", default=0.1))
    p.add_argument("--fi_schedule", type=str,
                   default=_cfg(cfg, "fi", "schedule", default="middle_peak"),
                   choices=["constant", "middle_peak", "warmup_decay", "cosine_decay"])
    p.add_argument("--fi_cache_mode", type=str,
                   default=_cfg(cfg, "fi", "cache_mode", default="attention"),
                   choices=["attention", "hidden", "mlp"])
    p.add_argument("--fi_max_cache_step", type=int,
                   default=_cfg(cfg, "fi", "max_cache_step", default=30),
                   help="FI 缓存步数上限 (30=全, 14B建议23)")

    # ── L3: FI 门控 ──
    p.add_argument("--fi_no_quality_gate", action="store_true",
                   help="禁用 FI 质量门控")
    p.add_argument("--fi_quality_k", type=float,
                   default=_cfg(cfg, "fi", "quality_gate", "k", default=20.0))

    # ── L3: FI 自适应门控 ──
    p.add_argument("--fi_no_adaptive_gate", action="store_true",
                   help="禁用 FI 自适应门控")
    p.add_argument("--fi_adaptive_temp", type=float,
                   default=_cfg(cfg, "fi", "adaptive_gate", "temp", default=5.0))
    p.add_argument("--fi_ag_gate_high", type=float,
                   default=_cfg(cfg, "fi", "adaptive_gate", "high", default=None))



    # ── 执行控制 ──
    p.add_argument("--resume", action="store_true", help="跳过已有输出")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def build_config(args) -> PFlowConfig:
    """从命令行参数构建配置。"""
    return PFlowConfig(
        t2v_path=args.model_path,
        dtype="bfloat16",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        use_svd=args.svd,
        use_iter=args.iter > 0,
        use_composite=args.composite,
        alpha=args.alpha,
        rho_s=args.rho_s,
        rho_m=args.rho_m,
        inversion_steps=args.inversion_steps,
        use_fast_svd=not args.no_fast_svd,
        svd_progressive=args.svd_progressive,
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
        fi_quality_k=args.fi_quality_k,
        fi_cache_mode=args.fi_cache_mode,
        fi_max_cache_step=args.fi_max_cache_step,
        fi_adaptive_gate=not args.fi_no_adaptive_gate,
        fi_adaptive_temp=args.fi_adaptive_temp,
        fi_ag_gate_high=args.fi_ag_gate_high,
        # SVD sigmoid 自适应 α
        pna_std_gate=not args.no_pna_std_gate,
        pna_std_eta0=args.pna_std_eta0,
        pna_std_kappa=args.pna_std_kappa,
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
    if config.use_svd:
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
