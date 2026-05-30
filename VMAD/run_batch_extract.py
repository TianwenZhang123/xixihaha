#!/usr/bin/env python3
"""
VMAD Batch Extraction + Application Script.

批量从 P-Flow 数据集中提取运动资产并应用到新内容，用于评测。

工作流:
    1. 读取 P-Flow/data/selected_200.csv 获取视频列表和 caption
    2. 对每个视频执行 extract (提取运动资产)
    3. 对每个资产执行 apply (应用到指定的新 content)
    4. 输出生成视频供评测脚本使用

用法:
    # 完整批量提取 + 应用 (使用 P-Flow 数据)
    python run_batch_extract.py \
        --video-dir ../P-Flow/data/videos_200 \
        --caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/vmad_full_batch \
        --content "a white cat"

    # 快速模式 (跳过 VLM)
    python run_batch_extract.py \
        --video-dir ../P-Flow/data/videos_200 \
        --caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/vmad_fast_batch \
        --content "a white cat" \
        --no-text_decode

    # 消融: 无解耦
    python run_batch_extract.py \
        --video-dir ../P-Flow/data/videos_200 \
        --caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/ablation_no_dis \
        --content "a white cat" \
        --no-disentangle

    # 多 content 批量 (用于 cross-content 评测)
    python run_batch_extract.py \
        --video-dir ../P-Flow/data/videos_200 \
        --caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/cross_content \
        --content "a white cat" "a robot" "a teddy bear" "an astronaut" "a goldfish" \
        --cross-content

    # 仅提取 (不应用)
    python run_batch_extract.py \
        --video-dir ../P-Flow/data/videos_200 \
        --caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/assets_only \
        --extract-only

    # 限制数量 (调试)
    python run_batch_extract.py \
        --video-dir ../P-Flow/data/videos_200 \
        --caption-dir ../P-Flow/data/captions_qwen \
        --output-dir ./outputs/debug \
        --content "a cat" \
        --limit 5

数据复用:
    直接使用 P-Flow/data/ 下的视频和 caption，无需额外准备数据。
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import VMADPipeline, VMADConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="VMAD Batch Extraction + Application for Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 数据目录 ──
    parser.add_argument("--video-dir", type=Path, required=True,
                        help="Directory containing source videos ({id}.mp4)")
    parser.add_argument("--caption-dir", type=Path, required=True,
                        help="Directory containing captions ({id}.txt) for extraction (e0)")
    parser.add_argument("--apply-caption-dir", type=Path, default=None,
                        help="Directory containing captions for apply phase. "
                             "If not set, uses --caption-dir. Used with --content SELF.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for all results")

    # ── 应用内容 ──
    parser.add_argument("--content", type=str, nargs="+", default=["a white cat"],
                        help="Content prompt(s) for application")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Motion strength for application")

    # ── 模式控制 ──
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract assets, don't apply")
    parser.add_argument("--apply-only", action="store_true",
                        help="Only apply (assets must already exist)")
    parser.add_argument("--cross-content", action="store_true",
                        help="Generate all content variants per asset (for cross-content eval)")

    # ── 模块开关 (同 run_extract.py) ──
    parser.add_argument("--no-inversion", action="store_true")
    parser.add_argument("--no-svd", action="store_true")
    parser.add_argument("--no-blend", action="store_true")
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--no-disentangle", action="store_true")
    parser.add_argument("--no-text_decode", action="store_true")
    parser.add_argument("--no-token_decode", action="store_true")
    parser.add_argument("--midpoint", action="store_true")

    # ── 超参数 ──
    parser.add_argument("--T_m", type=float, default=None)
    parser.add_argument("--num_opt_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_dis", type=float, default=None)
    parser.add_argument("--rho_s", type=float, default=None)
    parser.add_argument("--rho_m", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)

    # ── 模型 ──
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--vlm-provider", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    # ── 其他 ──
    parser.add_argument("--sample-ids", type=str, nargs="+", default=None,
                        help="Only process these specific video IDs (e.g. 7 17 21 31)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N videos (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already processed videos")
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def find_videos(video_dir: Path, caption_dir: Path, limit: int = 0) -> list:
    """Find all video-caption pairs."""
    items = []
    for video_path in sorted(video_dir.glob("*.mp4"),
                             key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        video_id = video_path.stem
        caption_path = caption_dir / f"{video_id}.txt"
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
            items.append({
                "id": video_id,
                "video_path": str(video_path),
                "caption": caption,
            })

    if limit > 0:
        items = items[:limit]
    return items


def build_config(args) -> VMADConfig:
    """Build VMADConfig from args."""
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        config = VMADConfig(**{k: v for k, v in cfg.items() if hasattr(VMADConfig, k)})
    else:
        config = VMADConfig()

    # Flags
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
    if args.no_token_decode:
        config.use_token_decode = False
    if args.midpoint:
        config.use_midpoint = True

    # Params
    param_map = {
        "T_m": "T_m", "num_opt_steps": "num_opt_steps",
        "lr": "opt_lr", "lambda_dis": "lambda_dis",
        "rho_s": "rho_s", "rho_m": "rho_m", "alpha": "alpha",
    }
    for arg_name, cfg_name in param_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, cfg_name, val)

    if args.model_path:
        config.t2v_path = args.model_path
    if args.vlm_provider:
        config.vlm_provider = args.vlm_provider

    config.seed = args.seed
    return config


def main():
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Find videos
    items = find_videos(args.video_dir, args.caption_dir, limit=args.limit)

    # Filter by sample IDs if specified
    if args.sample_ids:
        id_set = set(args.sample_ids)
        items = [item for item in items if item["id"] in id_set]

    if not items:
        logging.error(f"No videos found in {args.video_dir} with captions in {args.caption_dir}")
        sys.exit(1)

    logging.info(f"Found {len(items)} video-caption pairs")

    # Build config
    config = build_config(args)
    logging.info(f"Config flags: {config.active_flags()}")
    logging.info(f"Experiment: {config.experiment_name()}")

    # Output structure
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    generated_dir = out_dir / "generated"
    content_prompts_dir = out_dir / "content_prompts"
    assets_dir.mkdir(exist_ok=True)
    generated_dir.mkdir(exist_ok=True)
    content_prompts_dir.mkdir(exist_ok=True)

    # Initialize pipeline
    pipeline = VMADPipeline(config)

    # ── Phase 1: Batch Extract ──
    extract_results = []
    if not args.apply_only:
        logging.info(f"\n{'='*60}")
        logging.info("Phase 1: Batch Extraction")
        logging.info(f"{'='*60}")

        for idx, item in enumerate(items, 1):
            video_id = item["id"]
            asset_output = str(assets_dir / video_id)

            # Resume check
            if args.resume and (Path(asset_output) / "asset" / "asset.json").exists():
                logging.info(f"[{idx}/{len(items)}] {video_id} - SKIP (already exists)")
                extract_results.append({"id": video_id, "status": "skipped"})
                continue

            logging.info(f"[{idx}/{len(items)}] Extracting {video_id}...")

            try:
                result = pipeline.extract(
                    video_path=item["video_path"],
                    output_dir=asset_output,
                    caption=item["caption"],
                )
                extract_results.append({
                    "id": video_id,
                    "status": "success",
                    "time": result["time_seconds"],
                    "delta_e_norm": result["delta_e_norm"],
                    "motion_text": result.get("motion_text", ""),
                })
                logging.info(
                    f"  Done: {result['time_seconds']:.1f}s, "
                    f"||delta_e||={result['delta_e_norm']:.4f}"
                )
            except Exception as e:
                logging.error(f"  FAILED: {e}")
                extract_results.append({"id": video_id, "status": "failed", "error": str(e)})

        # Save extraction log
        with open(out_dir / "extraction_log.json", "w", encoding="utf-8") as f:
            json.dump(extract_results, f, indent=2, ensure_ascii=False)

    # ── Phase 2: Batch Apply ──
    if not args.extract_only:
        logging.info(f"\n{'='*60}")
        logging.info("Phase 2: Batch Application")
        logging.info(f"  Content(s): {args.content}")
        logging.info(f"  Strength: {args.strength}")
        logging.info(f"{'='*60}")

        apply_results = []

        for idx, item in enumerate(items, 1):
            video_id = item["id"]
            asset_path = str(assets_dir / video_id / "asset")

            if not Path(asset_path).exists():
                logging.warning(f"[{idx}/{len(items)}] {video_id} - No asset, skipping apply")
                continue

            if args.cross_content:
                # Cross-content mode: generate all variants
                for ci, content in enumerate(args.content):
                    apply_output = str(out_dir / "cross_content" / video_id / f"content_{ci:02d}")

                    if args.resume and Path(apply_output).joinpath("generated.mp4").exists():
                        continue

                    try:
                        result = pipeline.apply(
                            content_prompt=content,
                            asset_dir=asset_path,
                            output_dir=apply_output,
                            strength=args.strength,
                        )
                        apply_results.append({
                            "id": video_id, "content_idx": ci,
                            "content": content, "status": "success",
                        })
                    except Exception as e:
                        logging.error(f"  Apply failed ({video_id}, content_{ci}): {e}")
                        apply_results.append({
                            "id": video_id, "content_idx": ci,
                            "content": content, "status": "failed", "error": str(e),
                        })
            else:
                # Standard mode: use first content, output as {id}.mp4
                # "SELF" means use the video's own caption for reproduction validation
                if args.content[0] == "SELF":
                    if args.apply_caption_dir:
                        apply_cap_path = args.apply_caption_dir / f"{video_id}.txt"
                        content = apply_cap_path.read_text(encoding="utf-8").strip()
                    else:
                        content = item["caption"]
                else:
                    content = args.content[0]
                gen_video_path = generated_dir / f"{video_id}.mp4"

                if args.resume and gen_video_path.exists():
                    logging.info(f"[{idx}/{len(items)}] {video_id} - SKIP apply (exists)")
                    continue

                logging.info(f"[{idx}/{len(items)}] Applying to '{content[:40]}' ...")

                try:
                    apply_output = str(out_dir / "apply_tmp" / video_id)
                    result = pipeline.apply(
                        content_prompt=content,
                        asset_dir=asset_path,
                        output_dir=apply_output,
                        strength=args.strength,
                    )

                    # Copy/move generated video to flat structure for eval
                    import shutil
                    src_video = Path(apply_output) / "generated.mp4"
                    if src_video.exists():
                        shutil.copy2(str(src_video), str(gen_video_path))

                    # Save content prompt for eval
                    (content_prompts_dir / f"{video_id}.txt").write_text(
                        content, encoding="utf-8"
                    )

                    apply_results.append({
                        "id": video_id, "content": content,
                        "status": "success", "time": result["time_seconds"],
                    })
                    logging.info(f"  Done: {result['time_seconds']:.1f}s -> {gen_video_path}")

                except Exception as e:
                    logging.error(f"  Apply failed ({video_id}): {e}")
                    apply_results.append({
                        "id": video_id, "content": content,
                        "status": "failed", "error": str(e),
                    })

        # Save application log
        with open(out_dir / "application_log.json", "w", encoding="utf-8") as f:
            json.dump(apply_results, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    logging.info(f"\n{'='*60}")
    logging.info("Batch Processing Complete!")
    logging.info(f"  Output: {out_dir}")
    logging.info(f"  Assets: {assets_dir}")
    logging.info(f"  Generated: {generated_dir}")
    if not args.extract_only:
        n_success = sum(1 for r in apply_results if r.get("status") == "success")
        logging.info(f"  Applied: {n_success}/{len(items)}")
    logging.info(f"{'='*60}")

    # Save batch config for reproducibility
    batch_info = {
        "video_dir": str(args.video_dir),
        "caption_dir": str(args.caption_dir),
        "output_dir": str(args.output_dir),
        "content": args.content,
        "strength": args.strength,
        "flags": config.active_flags(),
        "experiment": config.experiment_name(),
        "num_videos": len(items),
        "seed": args.seed,
    }
    with open(out_dir / "batch_config.json", "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
