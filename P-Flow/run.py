#!/usr/bin/env python3
"""
P-Flow Runner - 通过命令行 flag 控制各改动点。

用法:
    # 纯 baseline (caption + 一次生成)
    python run.py --video /path/to/ref.mp4 --caption "a cat walking"

    # 启用噪声先验 (inversion + svd + blend)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend

    # 启用 Velocity Matching (Δe embedding 注入)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --velocity

    # 噪声先验 + Velocity (推荐最强组合)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend --velocity

    # 启用迭代优化 (10轮VLM反馈)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --iter 10

    # 完整 P-Flow (所有改动点)
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend --velocity --iter 10 --composite

    # 用中点法替代Euler
    python run.py --video /path/to/ref.mp4 --caption "a cat" --inversion --svd --blend --midpoint

    # 批量处理
    python run.py --data_dir /path/to/videos --caption_dir /path/to/captions --inversion --svd --blend --iter 10

改动点 Flag 说明:
    --inversion      启用 Flow Matching Inversion (从参考视频反演噪声)
    --svd            启用 SVD 两阶段滤波 (空间去内容 + 时间保运动)
    --blend          启用噪声混合 (η = sqrt(α)*η_temporal + sqrt(1-α)*η_random)
    --velocity       启用 Velocity Field Matching (Δe embedding 注入, 需 --inversion)
    --iter N         启用迭代VLM优化 (N轮反馈循环)
    --midpoint       使用二阶中点法ODE求解器 (替代默认Euler)
    --composite      启用三面板垂直拼接 (ref|prev|current 送VLM对比)

快捷组合:
    --noise_prior  等价于 --inversion --svd --blend
    --velocity_full 等价于 --inversion --svd --blend --velocity
    --full         等价于 --inversion --svd --blend --velocity --iter 10 --composite
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
    p.add_argument("--velocity", action="store_true", help="启用 Velocity Field Matching (Δe, 需 --inversion)")
    p.add_argument("--iter", type=int, default=0, help="迭代轮数 (0=不迭代)")
    p.add_argument("--midpoint", action="store_true", help="使用中点法ODE求解器")
    p.add_argument("--composite", action="store_true", help="启用垂直拼接对比")

    # ── 快捷组合 ──
    p.add_argument("--noise_prior", action="store_true", help="快捷: --inversion --svd --blend")
    p.add_argument("--velocity_full", action="store_true", help="快捷: --inversion --svd --blend --velocity")
    p.add_argument("--full", action="store_true", help="快捷: 全部启用 (iter=10)")

    # ── 参数调节 ──
    p.add_argument("--alpha", type=float, default=0.001, help="噪声混合权重")
    p.add_argument("--rho_s", type=float, default=0.1, help="空间SVD阈值")
    p.add_argument("--rho_m", type=float, default=0.9, help="时间SVD阈值")
    p.add_argument("--embed_strength", type=float, default=0.005, help="Δe 注入强度")
    p.add_argument("--velocity_steps", type=int, default=30, help="Velocity matching 优化步数")
    p.add_argument("--velocity_lr", type=float, default=1e-3, help="Velocity matching 学习率")
    p.add_argument("--velocity_K", type=int, default=4, help="每步采样的时间步数量 (stratified)")
    p.add_argument("--velocity_motion_weight", type=float, default=1.0, help="运动区域加权强度 (0=关闭, 1=全开)")
    p.add_argument("--steps", type=int, default=30, help="推理步数")
    p.add_argument("--guidance", type=float, default=5.0, help="CFG scale")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=15)

    # ── 模型路径 ──
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--vlm_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct")
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
        args.velocity = True
        args.composite = True
        if args.iter == 0:
            args.iter = 10

    if args.velocity_full:
        args.inversion = True
        args.svd = True
        args.blend = True
        args.velocity = True

    if args.noise_prior:
        args.inversion = True
        args.svd = True
        args.blend = True

    # velocity 依赖 inversion
    if args.velocity and not args.inversion:
        print("警告: --velocity 需要 --inversion，自动启用 --inversion")
        args.inversion = True

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
        use_velocity=args.velocity,
        use_iter=args.iter > 0,
        use_midpoint=args.midpoint,
        use_composite=args.composite,
        alpha=args.alpha,
        rho_s=args.rho_s,
        rho_m=args.rho_m,
        embed_strength=args.embed_strength,
        velocity_steps=args.velocity_steps,
        velocity_lr=args.velocity_lr,
        velocity_K=args.velocity_K,
        velocity_motion_weight=args.velocity_motion_weight,
        inversion_steps=50,
        i_max=args.iter if args.iter > 0 else 1,
        vlm_provider=args.vlm_provider,
        vlm_model_path=args.vlm_path,
        seed=args.seed,
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = build_config(args)

    # 打印配置摘要
    flags = config.active_flags()
    print(f"P-Flow | {config.experiment_name()}")
    print(f"  Flags: {flags or ['baseline (无改动)']}")
    if config.use_blend:
        print(f"  alpha={config.alpha}, rho_s={config.rho_s}, rho_m={config.rho_m}")
    if config.use_velocity:
        print(f"  velocity: steps={config.velocity_steps}, lr={config.velocity_lr}, embed_strength={config.embed_strength}")
        print(f"  velocity v2: K={config.velocity_K}, motion_weight={config.velocity_motion_weight}")
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
