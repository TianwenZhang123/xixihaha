"""
VMAD Motion Asset Application Script.

将提取的运动资产应用到新内容，生成具有相同运动模式的新视频。

用法:
    # 基本应用
    python run_apply.py --asset ./output/dance/asset --content "a white cat" \
        --output ./output/cat_dance

    # 调整运动强度
    python run_apply.py --asset ./output/dance/asset --content "a robot" \
        --output ./output/robot_dance --strength 0.8

    # 批量应用 (多个 content)
    python run_apply.py --asset ./output/dance/asset \
        --content "a white cat" "a robot" "a teddy bear" \
        --output ./output/batch_dance

    # 不使用噪声混合 (仅 delta_e)
    python run_apply.py --asset ./output/dance/asset --content "a cat" \
        --output ./output/cat_dance --no-blend

    # 自定义生成参数
    python run_apply.py --asset ./output/dance/asset --content "a cat" \
        --output ./output/cat_dance --guidance_scale 7.5 --num_inference_steps 50

参考 P-Flow 的 flag 架构:
    apply 模式下可控制是否使用 noise prior blending。
"""

import argparse
import logging
import sys
import os
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import VMADPipeline, VMADConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="VMAD: Apply Motion Asset to New Content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply dance motion to a cat
  python run_apply.py --asset ./output/dance/asset --content "a white cat dancing" \\
      --output ./output/cat_dance

  # Batch apply to multiple subjects
  python run_apply.py --asset ./output/dance/asset \\
      --content "a white cat" "a robot" "a teddy bear" \\
      --output ./output/batch

  # Weaker motion
  python run_apply.py --asset ./output/dance/asset --content "a cat" \\
      --output ./output/cat_dance --strength 0.5
        """,
    )

    # ── 必需参数 ──
    parser.add_argument(
        "--asset", type=str, required=True,
        help="Path to extracted motion asset directory",
    )
    parser.add_argument(
        "--content", type=str, nargs="+", required=True,
        help="Content prompt(s) for new video generation",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory",
    )

    # ── 运动控制 ──
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Motion strength factor (default: 1.0, range: 0.0-2.0)",
    )

    # ── 模块开关 ──
    parser.add_argument("--no-blend", action="store_true", help="Disable noise prior blending")

    # ── 生成参数 ──
    parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale (default: 5.0)")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Inference steps (default: 30)")
    parser.add_argument("--height", type=int, default=None, help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=None, help="Video width (default: 832)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (default: 81)")
    parser.add_argument("--fps", type=int, default=None, help="FPS (default: 15)")

    # ── 模型 ──
    parser.add_argument("--model_path", type=str, default=None, help="Path to Wan2.1 model")

    # ── 其他 ──
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument("--alpha", type=float, default=None, help="Noise blend ratio (default: 0.001)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser.parse_args()


def build_config(args) -> VMADConfig:
    """从命令行参数构建 VMADConfig。"""
    if args.config:
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        config = VMADConfig(**{k: v for k, v in yaml_cfg.items() if hasattr(VMADConfig, k)})
    else:
        config = VMADConfig()

    # Apply 模式不需要 extract 相关模块
    config.use_inversion = False
    config.use_spectral = False
    config.use_velocity = False
    config.use_disentangle = False
    config.use_text_decode = False

    # 开关
    if args.no_blend:
        config.use_blend = False

    # 参数覆盖
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale
    if args.num_inference_steps is not None:
        config.num_inference_steps = args.num_inference_steps
    if args.height is not None:
        config.height = args.height
    if args.width is not None:
        config.width = args.width
    if args.num_frames is not None:
        config.num_frames = args.num_frames
    if args.fps is not None:
        config.fps = args.fps
    if args.seed is not None:
        config.seed = args.seed
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.model_path:
        config.t2v_path = args.model_path

    return config


def main():
    args = parse_args()

    # 日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 检查输入
    if not os.path.isdir(args.asset):
        logging.error(f"Asset directory not found: {args.asset}")
        sys.exit(1)

    # 构建配置
    config = build_config(args)

    # 打印配置摘要
    logging.info("=" * 60)
    logging.info("VMAD Motion Asset Application")
    logging.info("=" * 60)
    logging.info(f"  Asset:      {args.asset}")
    logging.info(f"  Content(s): {args.content}")
    logging.info(f"  Strength:   {args.strength}")
    logging.info(f"  Blend:      {config.use_blend}")
    logging.info(f"  Output:     {args.output}")
    logging.info(f"  Seed:       {config.seed}")
    logging.info("=" * 60)

    # 初始化 pipeline
    pipeline = VMADPipeline(config)

    # 逐个 content 生成
    results = []
    for i, content in enumerate(args.content):
        if len(args.content) > 1:
            out_dir = os.path.join(args.output, f"content_{i:02d}")
        else:
            out_dir = args.output

        logging.info(f"\n[{i+1}/{len(args.content)}] Content: '{content}'")

        result = pipeline.apply(
            content_prompt=content,
            asset_dir=args.asset,
            output_dir=out_dir,
            strength=args.strength,
        )
        results.append(result)

    # 输出结果
    logging.info("")
    logging.info("=" * 60)
    logging.info("Application Complete!")
    logging.info(f"  Generated {len(results)} video(s)")
    for r in results:
        logging.info(f"    -> {r['video_path']}")
    total_time = sum(r["time_seconds"] for r in results)
    logging.info(f"  Total time: {total_time:.1f}s")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
