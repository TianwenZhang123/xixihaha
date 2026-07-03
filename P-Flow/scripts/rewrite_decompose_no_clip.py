#!/usr/bin/env python3
"""
结构化分解方案（纯LLM版，无CLIP评分）
不需要视频文件，只用LLM分解+变体生成+组装

用法:
    python scripts/rewrite_decompose_no_clip.py \
        --input-dir /root/xixihaha/P-Flow/data/video3_captions \
        --output-dir /root/xixihaha/P-Flow/data/video3_captions_decomposed \
        --llm-api-key YOUR_KEY
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志文件名带时间戳
log_file = LOG_DIR / f"rewrite_decompose_noclip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 配置日志：同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # 控制台
        logging.FileHandler(log_file, encoding="utf-8"),  # 文件
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file}")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_decompose import LLMClient


def optimize_prompt(llm: LLMClient, original: str, n_variants: int = 3) -> str:
    """纯LLM优化（无CLIP评分）"""
    from src.prompt_decompose import (
        DECOMPOSE_SYSTEM, DECOMPOSE_USER,
        VARIANT_SYSTEM, VARIANT_USER,
        ASSEMBLE_SYSTEM, ASSEMBLE_USER,
    )
    import re

    # Step 1: 分解
    try:
        resp = llm._call(
            DECOMPOSE_SYSTEM,
            DECOMPOSE_USER.format(caption=original),
            response_format={"type": "json_object"},
        )
        # 解析JSON
        try:
            components = json.loads(resp)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', resp, re.DOTALL)
            if match:
                components = json.loads(match.group())
            else:
                logger.warning(f"  分解JSON解析失败，返回原文")
                return original
    except Exception as e:
        logger.warning(f"  分解失败: {e}")
        return original

    logger.info(f"    分解为: {list(components.keys())}")

    # Step 2: 每个组件生成变体，选最长最具体的（无CLIP时的启发式）
    COMPONENT_DESCRIPTIONS = {
        "subject": "the main subjects/objects and their appearance",
        "scene": "the background environment, setting, and atmosphere",
        "motion": "actions, movements, gestures, speed, direction",
        "camera": "camera shot type, angle, and movement",
        "style": "visual style, color palette, lighting, mood",
    }

    best_components = {}
    for comp_name, comp_desc in COMPONENT_DESCRIPTIONS.items():
        current_text = components.get(comp_name, "not specified")
        if not current_text or current_text.lower() in ("not specified", "none", ""):
            best_components[comp_name] = current_text
            continue

        # 生成变体
        try:
            system = VARIANT_SYSTEM.format(
                n_variants=n_variants,
                component_name=comp_name,
                component_description=comp_desc,
            )
            user = VARIANT_USER.format(
                full_caption=original,
                component_name=comp_name,
                current_text=current_text,
                n_variants=n_variants,
            )
            resp = llm._call(system, user)
            try:
                variants = json.loads(resp)
            except json.JSONDecodeError:
                match = re.search(r'\[.*\]', resp, re.DOTALL)
                variants = json.loads(match.group()) if match else [resp]

            all_candidates = [current_text] + variants
            # 启发式选择：选最长的（通常最具体）
            best_text = max(all_candidates, key=lambda x: len(x.split()))
            logger.info(f"    {comp_name}: 选 \"{best_text[:50]}...\"")
        except Exception as e:
            logger.warning(f"    {comp_name} 变体失败: {e}")
            best_text = current_text

        best_components[comp_name] = best_text

    # Step 3: 组装
    try:
        final = llm._call(
            ASSEMBLE_SYSTEM,
            ASSEMBLE_USER.format(
                subject=best_components.get("subject", ""),
                scene=best_components.get("scene", ""),
                motion=best_components.get("motion", ""),
                camera=best_components.get("camera", ""),
                style=best_components.get("style", ""),
            ),
            temperature=0.3,
            max_tokens=512,
        )
    except Exception as e:
        logger.warning(f"    组装失败: {e}")
        parts = [v for k, v in best_components.items() if v and v != "not specified"]
        final = ". ".join(parts)

    return final


def main():
    parser = argparse.ArgumentParser(
        description="结构化分解方案（纯LLM版，无CLIP评分）"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-api-base", type=str,
                        default="https://token-plan-cn.xiaomimimo.com/v1")
    parser.add_argument("--llm-model", type=str, default="mimo-v2.5-pro")
    parser.add_argument("--n-variants", type=int, default=3)

    args = parser.parse_args()

    api_key = args.llm_api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        logger.error("需要 --llm-api-key 或 LLM_API_KEY")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    caption_files = sorted(
        input_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    logger.info(f"{'='*60}")
    logger.info(f"结构化分解方案（纯LLM版）")
    logger.info(f"样本数: {len(caption_files)}")
    logger.info(f"LLM: {args.llm_api_base}/{args.llm_model}")

    llm = LLMClient(
        api_key=api_key,
        api_base=args.llm_api_base,
        model=args.llm_model,
    )

    success = 0
    failed = 0
    skipped = 0
    results = []  # 收集详细结果

    for idx, cap_file in enumerate(caption_files, 1):
        sample_id = cap_file.stem
        out_file = output_dir / f"{sample_id}.txt"

        if args.skip_existing and out_file.exists():
            logger.info(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id}")
            skipped += 1
            results.append({"sample_id": sample_id, "status": "skipped", "reason": "exists"})
            continue

        original = cap_file.read_text(encoding="utf-8").strip()
        if not original:
            skipped += 1
            results.append({"sample_id": sample_id, "status": "skipped", "reason": "empty"})
            continue

        logger.info(f"\n  [{idx}/{len(caption_files)}] {sample_id}")
        logger.info(f"    原文: {original[:80]}...")

        try:
            optimized = optimize_prompt(llm, original, args.n_variants)
            out_file.write_text(optimized + "\n", encoding="utf-8")
            logger.info(f"    优化: {optimized[:80]}...")
            success += 1
            results.append({
                "sample_id": sample_id,
                "status": "success",
                "original": original,
                "optimized": optimized,
                "orig_words": len(original.split()),
                "opt_words": len(optimized.split()),
            })
        except Exception as e:
            logger.error(f"    失败: {e}")
            out_file.write_text(original + "\n", encoding="utf-8")
            failed += 1
            results.append({
                "sample_id": sample_id,
                "status": "failed",
                "error": str(e),
                "original": original,
            })

    logger.info(f"\n{'='*60}")
    logger.info(f"完成! success={success}, failed={failed}, skipped={skipped}")
    logger.info(f"输出: {output_dir}")

    # 保存详细JSON日志
    json_log = output_dir / "rewrite_log.json"
    log_data = {
        "version": "decompose_no_clip",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_api_base": args.llm_api_base,
            "llm_model": args.llm_model,
            "n_variants": args.n_variants,
        },
        "summary": {
            "total": len(caption_files),
            "success": success,
            "failed": failed,
            "skipped": skipped,
        },
        "results": results,
    }
    json_log.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"详细日志: {json_log}")
    logger.info(f"文本日志: {log_file}")


if __name__ == "__main__":
    main()
