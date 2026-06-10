#!/usr/bin/env python3
"""
Prompt 微调脚本 v8-minimal — 纯结构优化，零内容改动

设计理念：
  Pure L2 (原始 VLM caption + SVD v1) 已取得 CLIP 0.8964, X-CLIP 0.7874。
  问题不在内容，而在结构：
    - 100% 的 caption 以 "The video depicts/features..." 开头，浪费 UMT5 position-0
    - 72% 以 "The overall atmosphere..." 结尾，浪费尾部 token 权重
    - 80% 包含猜测/对冲词（suggesting/possibly/might be），模糊化 attention

  v8-minimal 只做三件事：
    1. 去掉 "The video ..." 开头，直接主体名词开头（利用 position-0 bias）
    2. 去掉猜测词和情感总结句（减少 attention 噪声）
    3. 补一句镜头描述在末尾（利用尾部 token 权重）

  严格红线（不做的事）：
    - 不改任何动词/运动描述 → SVD 继续独占运动通道
    - 不做视觉推断（不加颜色/材质/光照等原文没有的）
    - 不加时序词（initially/then/gradually）
    - 不加新对象/事件

  流程：VLM 原始 caption → LLM 一次结构微调 → 直出
  无需 VLM 验证（因为不做内容推断，不会引入事实错误）

预期效果：
  - CLIP 从 0.8964 微升（主体前置 + 去噪声让 UMT5 编码更聚焦）
  - X-CLIP 维持 0.7874（运动信息不变，SVD 继续有效）
  - 实现 L1+L2 > L2（prompt 微调在 SVD 基础上带来额外增益）

用法:
    python scripts/rewrite_minimal.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/minimal_captions \
        --backend dashscope \
        --model qwen-plus
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt v8-minimal: 纯结构优化，零内容推断
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You clean up VLM video captions for a T2V model (Wan2.1 with UMT5 text encoder). You do NOT add any new information. You only REMOVE noise and RESTRUCTURE for better encoding.

## Your 3 tasks (NOTHING ELSE):

### 1. SUBJECT-FIRST OPENING
Remove "The video depicts/features/captures/shows..." preamble. Start directly with the main subject noun phrase.

Rules:
- Identify what is the PRIMARY SUBJECT (the thing that moves or is most prominent)
- Start the output with that subject's noun phrase (with article)
- Do NOT change the subject's description — copy it exactly from the input
- Convert passive voice to active where the subject allows it naturally

Examples:
- "The video depicts a white SUV driving..." → "A white SUV drives..."
- "The video features two golden retriever puppies playing..." → "Two golden retriever puppies play..."
- "The video captures a cheetah in motion, running..." → "A cheetah runs..."

### 2. REMOVE NOISE (delete these, do not replace them):

Delete ALL of the following patterns wherever they appear:
- Hedging phrases: "suggesting...", "indicating that...", "appears to be...", "might be...", "possibly...", "likely...", "could be...", "perhaps..."
  - Delete the entire clause that starts with these words
  - Example: "carrying luggage or gear, suggesting it might be on a journey or adventure" → "carrying luggage"
- Meta-commentary about the video: "As the frames progress,", "Throughout the video,", "In one part of the video,"
- Overall summary sentences: Any sentence starting with "The overall atmosphere/mood/scene/effect..." — DELETE THE ENTIRE SENTENCE
- Redundant emotional interpretations: "conveying a sense of excitement", "adding to the whimsical feel", "creating a sense of urgency"
  - Only delete these when they are standalone clauses/phrases, NOT when they are part of a motion description

IMPORTANT: When deleting a hedging clause, make sure the remaining sentence is grammatically correct. If deleting leaves a dangling comma or incomplete sentence, clean it up minimally.

### 3. CAMERA CLOSING
Add ONE brief camera/shot description as the final sentence. Choose based on the scene content:
- Static scenes (objects, landscapes): "The camera holds steady in a [wide/medium/close-up] shot."
- Moving subjects (people, animals, vehicles): "The camera [follows/tracks] smoothly from a [low/eye-level/high] angle."
- Dramatic/epic scenes: "The camera captures the full scale from a [wide/elevated/sweeping] perspective."
- Close-ups (small objects, faces): "The camera remains fixed in extreme close-up throughout."

The camera sentence must be SHORT (under 15 words). Do not elaborate.

## ABSOLUTE RULES (violations = failure):

1. **ZERO motion changes**: Every verb, direction, speed, and movement description from the input MUST appear UNCHANGED in your output. If input says "kicks up a cloud of dust behind it" → output must contain "kicks up a cloud of dust behind it" verbatim.

2. **ZERO new information**: Do NOT add colors, materials, textures, lighting details, or any visual property not explicitly stated in the input. Do NOT infer or deduce anything.

3. **ZERO temporal markers**: Do NOT add "initially", "then", "gradually", "as time passes", "subsequently", etc.

4. **Preserve all factual content**: Every object, person, animal, color, spatial relationship, and described detail from the input must remain in the output (unless it was a hedging/noise phrase you're removing).

5. **Length**: Output should be SHORTER than input (you're removing noise, not adding content). Target: 85-95% of input length.

## Process:
1. Identify the primary subject (what moves or is most prominent)
2. Remove the "The video..." preamble, start with subject
3. Scan every sentence — delete hedging phrases and overall summary sentences
4. Verify ALL motion/action descriptions are unchanged
5. Add one camera sentence at the end
6. Verify output is shorter than input

Output ONLY the cleaned caption. No explanations."""

USER_TEMPLATE = """Clean up this VLM caption ({word_count} words). Remove preamble, hedging phrases, and overall-summary sentences. Start with the main subject. Add a brief camera note at the end. Do NOT change any motion/action descriptions. Do NOT add new information. Target ~{target_words} words.

INPUT:
{original_caption}

OUTPUT:"""

# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用后端
# ─────────────────────────────────────────────────────────────────────────────

def call_dashscope(prompt: str, system: str, model: str, api_key: str,
                   temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """调用 DashScope API (阿里云百炼)"""
    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai: pip install openai")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def call_openai_compatible(prompt: str, system: str, model: str,
                           api_base: str, api_key: str = "EMPTY",
                           temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """调用 OpenAI 兼容接口 (vLLM / ollama / etc.)"""
    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai: pip install openai")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_caption(original: str, backend: str, model: str,
                    api_base: str = "", api_key: str = "",
                    temperature: float = 0.3, max_retries: int = 3) -> str:
    """v8-minimal 改写：纯结构清理（去preamble + 去猜测 + 补镜头）"""
    word_count = len(original.split())
    target_words = int(word_count * 0.9)  # 目标90%长度

    user_msg = USER_TEMPLATE.format(
        word_count=word_count,
        target_words=target_words,
        original_caption=original,
    )

    result = None
    for attempt in range(max_retries + 1):
        temp = temperature if attempt == 0 else min(0.5, temperature + attempt * 0.1)

        if backend == "dashscope":
            result = call_dashscope(user_msg, SYSTEM_PROMPT, model, api_key, temp)
        elif backend == "openai":
            result = call_openai_compatible(user_msg, SYSTEM_PROMPT, model, api_base, api_key, temp)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # 清理可能的引号包裹
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        # ── 验证 1: 不能以 "The video" 开头 ──
        if result.lower().startswith(("the video", "this video", "in this video")):
            logger.warning(f"  [重试 {attempt+1}] 仍以 preamble 开头")
            continue

        # ── 验证 2: 长度检查（不能比原文长太多）──
        result_words = len(result.split())
        if result_words > word_count * 1.1:
            logger.warning(f"  [重试 {attempt+1}] 输出过长: {result_words} > {word_count}*1.1")
            continue

        # ── 验证 3: 不能太短（至少原文60%）──
        if result_words < word_count * 0.6:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words} < {word_count}*0.6")
            continue

        # 通过所有验证
        return result

    # 所有重试都失败，返回最后一次结果
    logger.warning(f"  所有重试均未通过验证，使用最后一次结果")
    return result if result else original


def validate_rewrite(original: str, rewritten: str) -> dict:
    """验证改写质量"""
    orig_words = len(original.split())
    new_words = len(rewritten.split())

    issues = []
    if not rewritten.strip():
        issues.append("empty output")
    if rewritten.lower().startswith(("the video", "this video", "in this video")):
        issues.append("still starts with preamble (subject-first violated)")
    if new_words > orig_words * 1.1:
        issues.append(f"longer than input ({new_words} > {orig_words}*1.1)")

    # 检查是否保留了关键运动动词
    motion_verbs = ["running", "walking", "driving", "flying", "swimming",
                    "moving", "riding", "skating", "jumping", "falling",
                    "climbing", "spinning", "rolling", "sliding", "flowing"]
    orig_lower = original.lower()
    rew_lower = rewritten.lower()
    lost_verbs = []
    for verb in motion_verbs:
        if verb in orig_lower and verb not in rew_lower:
            lost_verbs.append(verb)
    if lost_verbs:
        issues.append(f"lost motion verbs: {lost_verbs}")

    return {
        "valid": len(issues) == 0,
        "orig_words": orig_words,
        "new_words": new_words,
        "compression_ratio": new_words / max(orig_words, 1),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prompt 微调 v8-minimal：纯结构优化，零内容改动（配合 SVD v1 使用）"
    )

    # I/O
    parser.add_argument("--input-dir", type=str, required=True,
                        help="原始 VLM caption 目录（包含 {id}.txt 文件）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录（微调后的 {id}.txt）")
    parser.add_argument("--sample-ids", type=int, nargs="+",
                        help="只处理指定样本 ID（默认处理全部）")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的输出文件（断点续跑）")

    # LLM 后端
    parser.add_argument("--backend", type=str, default="dashscope",
                        choices=["dashscope", "openai"],
                        help="LLM 后端: dashscope (阿里云) 或 openai (兼容接口)")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="模型名称 (默认: qwen-plus)")
    parser.add_argument("--api-base", type=str, default="",
                        help="OpenAI 兼容接口地址 (仅 --backend openai 时需要)")
    parser.add_argument("--api-key", type=str, default="",
                        help="API Key (也可通过环境变量 DASHSCOPE_API_KEY 设置)")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM 温度 (默认: 0.3, 低温度确保不引入新内容)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单个样本最大重试次数 (默认: 3)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="请求间隔秒数 (默认: 0.5)")

    args = parser.parse_args()

    # 解析 API Key
    api_key = args.api_key
    if not api_key:
        if args.backend == "dashscope":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if args.backend == "dashscope" and not api_key:
        logger.error("需要设置 DASHSCOPE_API_KEY 环境变量或通过 --api-key 传入")
        sys.exit(1)

    # 准备目录
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    # 收集待处理文件
    caption_files = sorted(
        input_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    if not caption_files:
        logger.error(f"未找到 caption 文件: {input_dir}/*.txt")
        sys.exit(1)

    logger.info(f"{'='*60}")
    logger.info(f"v8-minimal: 纯结构优化（去preamble/去猜测/补镜头）")
    logger.info(f"{'='*60}")
    logger.info(f"待处理: {len(caption_files)} 个样本")
    logger.info(f"后端: {args.backend}, 模型: {args.model}, 温度: {args.temperature}")
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"策略: 不改内容、不加推断、不碰运动 → 配合 SVD v1 使用")

    # 统计
    success = 0
    failed = 0
    skipped = 0
    results = []

    for idx, cap_file in enumerate(caption_files, 1):
        sample_id = cap_file.stem
        out_file = output_dir / f"{sample_id}.txt"

        # 断点续跑
        if args.skip_existing and out_file.exists():
            logger.info(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id} (已存在)")
            skipped += 1
            continue

        original = cap_file.read_text(encoding="utf-8").strip()
        if not original:
            logger.warning(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id} (空文件)")
            skipped += 1
            continue

        # ── LLM 结构微调 ──
        try:
            rewritten = rewrite_caption(
                original=original,
                backend=args.backend,
                model=args.model,
                api_base=args.api_base,
                api_key=api_key,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )

            validation = validate_rewrite(original, rewritten)

            if not validation["valid"]:
                logger.warning(
                    f"  [{idx}/{len(caption_files)}] {sample_id} 验证警告: "
                    f"{validation['issues']}"
                )

            # 保存结果
            out_file.write_text(rewritten + "\n", encoding="utf-8")
            success += 1

            logger.info(
                f"  [{idx}/{len(caption_files)}] {sample_id} ✓ "
                f"({validation['orig_words']}→{validation['new_words']} words, "
                f"x{validation['compression_ratio']:.2f})"
            )
            results.append({
                "sample_id": sample_id,
                "status": "success",
                "orig_words": validation["orig_words"],
                "new_words": validation["new_words"],
                "compression_ratio": validation["compression_ratio"],
                "issues": validation["issues"],
            })

        except Exception as e:
            failed += 1
            logger.error(f"  [{idx}/{len(caption_files)}] {sample_id} ✗ 失败: {e}")
            results.append({"sample_id": sample_id, "status": "failed", "error": str(e)})

        # 请求间隔
        if idx < len(caption_files):
            time.sleep(args.delay)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! 成功={success}, 失败={failed}, 跳过={skipped}, 总计={len(caption_files)}")
    logger.info(f"输出目录: {output_dir}")

    if results:
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            avg_compression = sum(r["compression_ratio"] for r in successful) / len(successful)
            logger.info(f"平均压缩率: {avg_compression:.2f} (目标 0.85-0.95)")

    # 保存处理日志
    log_file = output_dir / "rewrite_log.json"
    log_data = {
        "version": "v8_minimal",
        "strategy": "structure_only_no_content_change",
        "description": "纯结构优化：去preamble(主体前置) + 去猜测词/情感总结 + 补镜头描述。零内容改动，配合 SVD v1 使用。",
        "changes": [
            "1. Subject-first: remove 'The video depicts/features...' preamble",
            "2. Remove noise: delete hedging phrases and overall-summary sentences",
            "3. Camera closing: add brief shot description at end",
        ],
        "red_lines": [
            "No motion/verb changes",
            "No visual inference (no new colors/materials/textures)",
            "No temporal markers",
            "No new objects/events",
        ],
        "backend": args.backend,
        "model": args.model,
        "temperature": args.temperature,
        "total": len(caption_files),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    log_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"处理日志: {log_file}")


if __name__ == "__main__":
    main()
