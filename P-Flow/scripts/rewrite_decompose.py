#!/usr/bin/env python3
"""
结构化分解 + CLIPScore择优 方案：改写video3_captions的24个prompt
不生成视频，只输出优化后的prompt到新目录

用法:
    python scripts/rewrite_decompose.py \
        --input-dir /root/xixihaha/P-Flow/data/video3_captions \
        --video-dir /root/xixihaha/P-Flow/data/video3 \
        --output-dir /root/xixihaha/P-Flow/data/video3_captions_decomposed \
        --llm-api-key YOUR_KEY
"""

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_decompose import create_prompt_decomposer


def main():
    parser = argparse.ArgumentParser(
        description="结构化分解 + CLIPScore择优: 改写video3_captions"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入caption目录 (video3_captions)")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="原始视频目录 (video3)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--sample-ids", type=int, nargs="+",
                        help="只处理指定样本ID (默认全部)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的输出文件")
    parser.add_argument("--llm-api-key", type=str, default="",
                        help="LLM API Key (或环境变量 LLM_API_KEY)")
    parser.add_argument("--llm-api-base", type=str,
                        default="https://token-plan-cn.xiaomimimo.com/v1",
                        help="LLM API Base URL")
    parser.add_argument("--llm-model", type=str, default="mimo-v2.5-pro",
                        help="LLM模型名称")
    parser.add_argument("--n-variants", type=int, default=3,
                        help="每个组件生成的变体数量 (默认3)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="CLIP设备 (默认cuda)")

    args = parser.parse_args()

    # API Key
    api_key = args.llm_api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        logger.error("需要 --llm-api-key 或 LLM_API_KEY 环境变量")
        sys.exit(1)

    # 目录
    input_dir = Path(args.input_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    if not video_dir.exists():
        logger.error(f"视频目录不存在: {video_dir}")
        sys.exit(1)

    # 收集caption文件
    caption_files = sorted(
        input_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    logger.info(f"{'='*60}")
    logger.info(f"结构化分解 + CLIPScore择优 方案")
    logger.info(f"{'='*60}")
    logger.info(f"样本数: {len(caption_files)}")
    logger.info(f"LLM: {args.llm_api_base}/{args.llm_model}")
    logger.info(f"变体数: {args.n_variants}")
    logger.info(f"输入: {input_dir}")
    logger.info(f"视频: {video_dir}")
    logger.info(f"输出: {output_dir}")

    # 创建PromptDecomposer
    decomposer = create_prompt_decomposer(
        api_key=api_key,
        api_base=args.llm_api_base,
        model=args.llm_model,
        device=args.device,
    )

    # 处理
    success = 0
    failed = 0
    skipped = 0

    for idx, cap_file in enumerate(caption_files, 1):
        sample_id = cap_file.stem
        out_file = output_dir / f"{sample_id}.txt"
        video_path = video_dir / f"{sample_id}.mp4"

        # 跳过已存在
        if args.skip_existing and out_file.exists():
            logger.info(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id} (已存在)")
            skipped += 1
            continue

        # 读取原始caption
        original = cap_file.read_text(encoding="utf-8").strip()
        if not original:
            logger.warning(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id} (空文件)")
            skipped += 1
            continue

        # 检查视频
        if not video_path.exists():
            logger.warning(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id} (视频不存在: {video_path})")
            skipped += 1
            continue

        logger.info(f"\n  [{idx}/{len(caption_files)}] 处理 {sample_id}")
        logger.info(f"    原文 ({len(original.split())} words): {original[:80]}...")

        try:
            # 执行优化
            optimized = decomposer.optimize(
                original_prompt=original,
                video_path=str(video_path),
                n_variants=args.n_variants,
            )

            # 保存
            out_file.write_text(optimized + "\n", encoding="utf-8")

            logger.info(f"    优化后 ({len(optimized.split())} words): {optimized[:80]}...")
            success += 1

        except Exception as e:
            logger.error(f"    失败: {e}")
            failed += 1
            # fallback: 保存原文
            out_file.write_text(original + "\n", encoding="utf-8")

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! success={success}, failed={failed}, skipped={skipped}")
    logger.info(f"输出: {output_dir}")


if __name__ == "__main__":
    main()
