#!/usr/bin/env python3
"""
快速验证 LLM 改写质量（不生成视频，只测 prompt 改写）

用途：
  在跑完整 pipeline 之前，先验证 V4 prompt + temperature/diff check 的效果。
  对比改写结果 vs 手动 ground truth (captions_hybrid)，计算：
  1. 长度保持率（输出/输入 word count ratio）
  2. 编辑距离（改动了多少）
  3. 主体识别正确率（开头是否是正确的主体）
  4. 与 ground truth 的相似度

分阶段验证方案：
  Phase 1: --ablation base     → V4 prompt + temp 0.5 + length/diff check（全部改动）
  Phase 2: --ablation no_check → V4 prompt + temp 0.5，关闭 length/diff check
  Phase 3: --ablation high_temp → V4 prompt + temp 0.7，关闭 length/diff check
  
  对比 Phase 1 vs 2 → 验证 length/diff check 的价值
  对比 Phase 2 vs 3 → 验证 temperature 降低的价值
  Phase 1 vs ground truth → 验证 V4 prompt 整体效果

用法:
    export DASHSCOPE_API_KEY="sk-xxxxx"
    python scripts/verify_rewrite_quality.py \
        --baseline_dir /path/to/baseline \
        --gt_dir data/captions_hybrid \
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \
        --ablation base \
        --output_dir /path/to/verify_output
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 导入改写函数
from scripts.run_hybrid_iter import (
    REWRITE_SYSTEM,
    call_llm,
    _compute_edit_ratio,
)


# ─── 已知的正确主体（用于验证主体识别）───
EXPECTED_SUBJECTS = {
    7: "Two small sailboats",
    17: "White SUV",
    21: "Colorful paper airplanes",
    31: "Giant whale",
    32: "Massive volcanic eruption",  # or "Volcano"
    33: "Two adorable golden retriever puppies",
    34: "Vibrant red and orange autumn leaves",
    43: "Orange and white cat",
    46: "Massive volcanic eruption",
    47: "Bright orange goldfish",
}


def check_subject(text: str, sample_id: int) -> bool:
    """检查改写结果的开头是否包含正确主体"""
    expected = EXPECTED_SUBJECTS.get(sample_id, "")
    if not expected:
        return True  # 没有 ground truth，跳过
    # 检查前 30 个字符是否包含主体关键词
    first_words = text[:80].lower()
    # 取主体的核心名词（最后一个词）
    key_nouns = expected.lower().split()[-1]  # e.g., "whale", "airplanes", "SUV"
    return key_nouns in first_words


def rewrite_with_ablation(caption: str, ablation: str, model: str) -> str:
    """根据 ablation 模式调用不同配置的改写"""
    word_count = len(caption.split())
    user_msg = (
        f"Restructure this VLM caption ({word_count} words). "
        f"First, identify what MOVES in this caption — that is your action subject. "
        f"Then make ONLY 3 changes: "
        f"(1) move the action subject to the first word (delete 'The video captures/shows...' if present), "
        f"(2) find the 1-2 sentences about the subject's motion and add a temporal chain (initially/then/gradually), "
        f"(3) end the last sentence with a key motion/visual word. "
        f"Copy ALL other sentences VERBATIM — do not rephrase, compress, or merge paragraphs. "
        f"Delete only meta-text like 'In summary/Overall/This perspective allows...'. "
        f"Output must be ~{word_count} words (±15%). Do NOT compress.\n\n"
        f"INPUT:\n{caption}\n\n"
        f"OUTPUT:"
    )

    if ablation == "base":
        # 全部改动：temp 0.5 + length/diff check + retry
        for attempt in range(3):
            temp = 0.5 if attempt == 0 else max(0.3, 0.5 - attempt * 0.1)
            result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)
            result_words = len(result.split())
            ratio = result_words / max(word_count, 1)
            edit_ratio = _compute_edit_ratio(caption, result)
            if ratio >= 0.70 and edit_ratio <= 0.50:
                return result
            logger.warning(f"    [retry {attempt+1}] ratio={ratio:.0%}, edit={edit_ratio:.0%}")
        return result

    elif ablation == "no_check":
        # V4 prompt + temp 0.5，但不做验证/重试
        return call_llm(user_msg, REWRITE_SYSTEM, model, temperature=0.5)

    elif ablation == "high_temp":
        # V4 prompt + temp 0.7（原始温度），不做验证
        return call_llm(user_msg, REWRITE_SYSTEM, model, temperature=0.7)

    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def main():
    p = argparse.ArgumentParser(description="验证 LLM 改写质量（快速，不生成视频）")
    p.add_argument("--baseline_dir", type=str, required=True,
                   help="baseline 输出目录（读取 VLM caption）")
    p.add_argument("--gt_dir", type=str, default="data/captions_hybrid",
                   help="手动 ground truth caption 目录")
    p.add_argument("--sample_ids", type=int, nargs="+",
                   default=[7, 17, 21, 31, 32, 33, 34, 43, 46, 47])
    p.add_argument("--ablation", type=str, default="base",
                   choices=["base", "no_check", "high_temp"],
                   help="消融实验模式")
    p.add_argument("--llm_model", type=str, default="qwen-plus")
    p.add_argument("--output_dir", type=str, default="outputs/verify_rewrite")
    args = p.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("需要设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir)
    results = []

    logger.info(f"{'=' * 60}")
    logger.info(f"验证模式: {args.ablation}")
    logger.info(f"{'=' * 60}")

    for sid in args.sample_ids:
        # 读取 baseline VLM caption
        cap_file = Path(args.baseline_dir) / f"sample_{sid}" / "vlm_caption.txt"
        if not cap_file.exists():
            logger.warning(f"  [{sid}] 找不到 baseline caption，跳过")
            continue
        vlm_caption = cap_file.read_text(encoding="utf-8").strip()
        input_words = len(vlm_caption.split())

        # 改写
        logger.info(f"  [{sid}] 改写中... (input: {input_words} words)")
        rewritten = rewrite_with_ablation(vlm_caption, args.ablation, args.llm_model)
        output_words = len(rewritten.split())

        # 保存改写结果
        (out_dir / f"{sid}.txt").write_text(rewritten, encoding="utf-8")

        # ── 计算指标 ──
        # 1. 长度保持率
        length_ratio = output_words / max(input_words, 1)

        # 2. 与输入的编辑距离（改动量）
        edit_from_input = _compute_edit_ratio(vlm_caption, rewritten)

        # 3. 主体识别
        subject_correct = check_subject(rewritten, sid)

        # 4. 与 ground truth 的相似度
        gt_file = gt_dir / f"{sid}.txt"
        gt_similarity = 0.0
        if gt_file.exists():
            gt_text = gt_file.read_text(encoding="utf-8").strip()
            gt_similarity = SequenceMatcher(
                None, rewritten.split(), gt_text.split()
            ).ratio()

        result = {
            "sample_id": sid,
            "input_words": input_words,
            "output_words": output_words,
            "length_ratio": round(length_ratio, 3),
            "edit_from_input": round(edit_from_input, 3),
            "subject_correct": subject_correct,
            "gt_similarity": round(gt_similarity, 3),
            "first_20_words": " ".join(rewritten.split()[:20]),
        }
        results.append(result)

        status = "✓" if subject_correct else "✗"
        logger.info(
            f"    {status} words={output_words}/{input_words} ({length_ratio:.0%}), "
            f"edit={edit_from_input:.0%}, gt_sim={gt_similarity:.0%}"
        )

    # ── 汇总 ──
    if results:
        avg_length = sum(r["length_ratio"] for r in results) / len(results)
        avg_edit = sum(r["edit_from_input"] for r in results) / len(results)
        avg_gt_sim = sum(r["gt_similarity"] for r in results) / len(results)
        subject_acc = sum(1 for r in results if r["subject_correct"]) / len(results)

        print(f"\n{'─' * 70}")
        print(f"  Ablation: {args.ablation}")
        print(f"  Samples: {len(results)}")
        print(f"{'─' * 70}")
        print(f"  Avg Length Ratio:     {avg_length:.1%}  (target: 85-115%)")
        print(f"  Avg Edit Distance:    {avg_edit:.1%}  (target: 20-40%)")
        print(f"  Subject Accuracy:     {subject_acc:.0%}  (target: 100%)")
        print(f"  Avg GT Similarity:    {avg_gt_sim:.1%}  (target: >60%)")
        print(f"{'─' * 70}\n")

        summary = {
            "ablation": args.ablation,
            "model": args.llm_model,
            "num_samples": len(results),
            "avg_length_ratio": round(avg_length, 3),
            "avg_edit_from_input": round(avg_edit, 3),
            "subject_accuracy": round(subject_acc, 3),
            "avg_gt_similarity": round(avg_gt_sim, 3),
            "per_sample": results,
        }
        summary_path = out_dir / f"verify_{args.ablation}.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
