"""
VMAD Motion Asset Extraction Script.

从参考视频中提取运动资产 (Motion Asset)。

用法:
    # 完整提取 (所有模块)
    python run_extract.py --video input.mp4 --output ./output/dance

    # 快速提取 (跳过 VLM 解码)
    python run_extract.py --video input.mp4 --output ./output/dance --no-text_decode

    # 仅反演 + SVD (不做优化)
    python run_extract.py --video input.mp4 --output ./output/dance \
        --no-velocity --no-disentangle --no-text_decode

    # 使用 midpoint 求解器
    python run_extract.py --video input.mp4 --output ./output/dance --midpoint

    # 自定义参数
    python run_extract.py --video input.mp4 --output ./output/dance \
        --T_m 0.5 --num_opt_steps 200 --lr 5e-4

    # 指定 caption (跳过 VLM 自动生成)
    python run_extract.py --video input.mp4 --output ./output/dance \
        --caption "a person dancing hip-hop"

参考 P-Flow 的 flag 架构:
    每个模块对应一个 --no-xxx 开关，默认全部启用。
"""

import argparse
import logging
import sys
import os
import yaml
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import VMADPipeline, VMADConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="VMAD: Extract Motion Asset from Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction
  python run_extract.py --video dance.mp4 --output ./output/dance_asset

  # Fast mode (skip VLM decode)
  python run_extract.py --video dance.mp4 --output ./output/dance_asset --no-text_decode

  # Ablation: no disentanglement
  python run_extract.py --video dance.mp4 --output ./output/dance_asset --no-disentangle
        """,
    )

    # ── 必需参数 ──
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to reference video file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for extracted asset",
    )

    # ── 可选文本 ──
    parser.add_argument(
        "--caption", type=str, default="",
        help="Manual caption for the video (skip VLM auto-caption if provided)",
    )

    # ── 模块开关 (默认全部启用, 用 --no-xxx 关闭) ──
    parser.add_argument("--no-inversion", action="store_true", help="Disable Flow Matching Inversion")
    parser.add_argument("--no-svd", action="store_true", help="Disable SVD filtering")
    parser.add_argument("--no-blend", action="store_true", help="Disable noise blending")
    parser.add_argument("--no-velocity", action="store_true", help="Disable Velocity Field Matching")
    parser.add_argument("--no-disentangle", action="store_true", help="Disable content disentanglement")
    parser.add_argument("--no-text_decode", action="store_true", help="Disable VLM motion text decoding")
    parser.add_argument("--midpoint", action="store_true", help="Use midpoint solver for inversion")

    # ── 超参数 ──
    parser.add_argument("--inversion_steps", type=int, default=None, help="Inversion ODE steps (default: 50)")
    parser.add_argument("--rho_s", type=float, default=None, help="SVD spatial threshold (default: 0.1)")
    parser.add_argument("--rho_m", type=float, default=None, help="SVD temporal threshold (default: 0.9)")
    parser.add_argument("--alpha", type=float, default=None, help="Noise blend ratio (default: 0.001)")
    parser.add_argument("--T_m", type=float, default=None, help="Velocity matching time range (default: 1.0 for reproduction, 0.3 for motion transfer)")
    parser.add_argument("--num_opt_steps", type=int, default=None, help="Optimization steps (default: 100)")
    parser.add_argument("--lr", type=float, default=None, help="Optimization learning rate (default: 1e-3)")
    parser.add_argument("--lambda_dis", type=float, default=None, help="Disentangle loss weight (default: 0.0 for reproduction, 0.1 for motion transfer)")
    parser.add_argument("--num_augmentations", type=int, default=None, help="Number of augmented prompts (default: 5)")

    # ── 模型 ──
    parser.add_argument("--model_path", type=str, default=None, help="Path to Wan2.1 model")
    parser.add_argument("--vlm_provider", type=str, default=None, choices=["local", "api", "mock"], help="VLM provider")
    parser.add_argument("--vlm_model_path", type=str, default=None, help="Path to VLM model")
    parser.add_argument("--aug_provider", type=str, default=None, choices=["mock", "dashscope", "local"], help="Augmentation provider")

    # ── 视频参数 ──
    parser.add_argument("--height", type=int, default=None, help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=None, help="Video width (default: 832)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (default: 81)")
    parser.add_argument("--fps", type=int, default=None, help="FPS (default: 15)")

    # ── 其他 ──
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument("--no-save_intermediates", action="store_true", help="Don't save intermediate results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser.parse_args()


def build_config(args) -> VMADConfig:
    """从命令行参数构建 VMADConfig。"""
    # 基础配置 (从 YAML 或默认)
    if args.config:
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        config = VMADConfig(**{k: v for k, v in yaml_cfg.items() if hasattr(VMADConfig, k)})
    else:
        config = VMADConfig()

    # 模块开关
    if args.no_inversion:
        config.use_inversion = False
    if args.no_svd:
        config.use_spectral = False
    if args.no_blend:
        config.use_blend = False
    if args.no_velocity:
        config.use_velocity = False
    if args.no_disentangle:
        config.use_disentangle = False
    if args.no_text_decode:
        config.use_text_decode = False
    if args.midpoint:
        config.use_midpoint = True
    if args.no_save_intermediates:
        config.save_intermediates = False

    # 超参数覆盖
    param_map = {
        "inversion_steps": "inversion_steps",
        "rho_s": "rho_s",
        "rho_m": "rho_m",
        "alpha": "alpha",
        "T_m": "T_m",
        "num_opt_steps": "num_opt_steps",
        "lr": "opt_lr",
        "lambda_dis": "lambda_dis",
        "num_augmentations": "num_augmentations",
        "seed": "seed",
        "height": "height",
        "width": "width",
        "num_frames": "num_frames",
        "fps": "fps",
    }
    for arg_name, cfg_name in param_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, cfg_name, val)

    # 模型路径
    if args.model_path:
        config.t2v_path = args.model_path
    if args.vlm_provider:
        config.vlm_provider = args.vlm_provider
    if args.vlm_model_path:
        config.vlm_model_path = args.vlm_model_path
    if args.aug_provider:
        config.augmentation_provider = args.aug_provider

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
    if not os.path.exists(args.video):
        logging.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # 构建配置
    config = build_config(args)

    # 打印配置摘要
    logging.info("=" * 60)
    logging.info("VMAD Motion Asset Extraction")
    logging.info("=" * 60)
    logging.info(f"  Video:      {args.video}")
    logging.info(f"  Output:     {args.output}")
    logging.info(f"  Flags:      {config.active_flags()}")
    logging.info(f"  Experiment: {config.experiment_name()}")
    logging.info(f"  Seed:       {config.seed}")
    logging.info("=" * 60)

    # 运行
    pipeline = VMADPipeline(config)
    result = pipeline.extract(
        video_path=args.video,
        output_dir=args.output,
        caption=args.caption,
    )

    # 输出结果
    logging.info("")
    logging.info("=" * 60)
    logging.info("Extraction Complete!")
    logging.info(f"  Asset saved to: {result['asset_dir']}")
    logging.info(f"  Motion text:    {result['motion_text'][:60]}...")
    logging.info(f"  Time:           {result['time_seconds']:.1f}s")
    logging.info(f"  ||delta_e||:    {result['delta_e_norm']:.4f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
