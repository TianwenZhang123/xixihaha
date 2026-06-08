#!/usr/bin/env python3
"""
Hybrid Prompt 自动改写脚本

将原始 VLM caption 按照融合策略三原则自动改写：
1. 首词=主体名词（利用 UMT5 position 0 权重优势）
2. 注入时序动作链（initially...then...finally...）
3. 保留原始视觉描述词（颜色、材质、光线等不替换）

支持两种 LLM 后端：
- DashScope API（Qwen 系列，适合 AutoDL 服务器）
- OpenAI-compatible API（本地 vLLM / ollama / 任意兼容接口）

用法:
    # 使用 DashScope (需设置 DASHSCOPE_API_KEY)
    python scripts/rewrite_hybrid.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/hybrid_captions \
        --backend dashscope \
        --model qwen-plus

    # 使用本地 Qwen2.5-72B (通过 vLLM 部署)
    python scripts/rewrite_hybrid.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/hybrid_captions \
        --backend openai \
        --api-base http://localhost:8000/v1 \
        --model Qwen2.5-72B-Instruct

    # 只处理指定样本
    python scripts/rewrite_hybrid.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/hybrid_captions \
        --sample-ids 7 17 21 31 32

    # 跳过已存在的文件（断点续跑）
    python scripts/rewrite_hybrid.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/hybrid_captions \
        --skip-existing
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 融合策略 System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You restructure VLM video captions into better T2V generation prompts. Your output must be nearly identical to the input — you only make 3 surgical changes.

## CRITICAL: How to identify the ACTION SUBJECT
The action subject is THE THING THAT MOVES OR ACTS in the video. Ask: "What is performing the main motion?"
- If a whale swims through a cityscape → subject = "Giant whale", NOT "Underwater cityscape"
- If paper airplanes fly through a jungle → subject = "Colorful paper airplanes", NOT "Vibrant jungle environment"
- If a SUV drives on a road → subject = "White SUV", NOT "Scenic mountainous landscape"
- If puppies waddle through snow → subject = "Two adorable golden retriever puppies", NOT "Serene snowy landscape"
- If a cat walks through a garden → subject = "Orange and white cat", NOT "Serene garden"
- If a volcano erupts → subject = "Massive volcanic eruption", NOT "High vantage point"
The subject is NEVER the background/environment/setting. It is always the moving entity.

## The 3 changes you make (NOTHING ELSE):

1. OPENING — Move the action subject to the very first words. Delete "The video shows/captures/depicts/showcases..." and start with the subject noun phrase.

2. ACTION — Find the 1-2 sentences where the subject's motion is described. Convert static verbs to a temporal chain using "initially/then/gradually". Add motion direction if implied but not stated. Do NOT touch any other sentences.

3. ENDING — Make the final phrase end with a vivid motion or visual keyword (e.g., "gentle circular motion", "dust trail billowing behind", "dappled jungle light").

## What you must NOT do:

- Do NOT compress or summarize. If the input is 150 words, output ~150 words.
- Do NOT rephrase visual descriptions. Copy them VERBATIM: "dark brown hulls" stays "dark brown hulls", "glass facades and steel structures" stays "glass facades and steel structures".
- Do NOT merge paragraphs. If input has 3 paragraphs, output has ~3 paragraphs.
- Do NOT add information the original doesn't mention or imply.
- Do NOT add temporal markers to background/lighting/atmosphere sentences.

## Process:
1. Read the input. Identify the action subject (what moves?).
2. Copy the ENTIRE input text.
3. Move the subject to position 0 (delete framework phrase if needed).
4. Find the 1-2 motion sentences → insert temporal chain.
5. Adjust the last phrase to end on a strong visual/motion word.
6. Leave everything else UNTOUCHED.

## Examples:

### Example 1 (whale in underwater city):
INPUT: "The video depicts an underwater cityscape with tall buildings emerging from the water. The buildings have a modern architectural style with glass facades and steel structures. The water is dark blue and rippled, creating a sense of depth and movement. A large whale swims gracefully through the center of the scene. Fish can be seen swimming around the whale, adding to the underwater atmosphere. The lighting is dim, giving the scene a mysterious and serene mood."
OUTPUT: "Giant whale swimming gracefully through an underwater cityscape with tall buildings emerging from the water. The buildings have a modern architectural style with glass facades and steel structures. The water is dark blue and rippled, creating a sense of depth and movement. The whale initially enters from the left side of the frame, then glides steadily rightward through the center of the scene, its tail and fins moving in slow rhythmic undulation. Fish can be seen swimming around the whale, scattering as it passes and adding to the underwater atmosphere. The lighting is dim, giving the scene a mysterious and serene mood, with the whale's massive form creating gentle currents in the dark blue water."
WHY: Subject="whale" (it swims), not "cityscape" (static background). Sentences about buildings/water/lighting copied verbatim. Only the whale's motion sentence was expanded into a temporal chain.

### Example 2 (paper airplanes in jungle — multi-paragraph):
INPUT: "The video showcases a vibrant and lush jungle environment, with dense green foliage covering the ground and towering trees stretching towards the sky. The trees have a mix of thin and thick trunks, some with bark that appears weathered and rugged. The canopy overhead is thick with leaves, allowing only patches of sunlight to filter through and cast dappled shadows on the forest floor below.\\n\\nA variety of colorful paper airplanes, including shades of white, pink, purple, yellow, and green, are seen flying through the air. The planes vary in size and design, some appearing more complex than others. They gracefully glide and spin as they move across the frame, contrasting beautifully against the natural backdrop of the forest.\\n\\nThe scene is peaceful and serene, with the gentle rustling of leaves and the occasional chirping sounds of birds providing a soothing soundtrack to the visual display. The overall atmosphere is one of tranquility and harmony between nature and human creativity in a beautiful jungle setting."
OUTPUT: "Colorful paper airplanes flying through a vibrant and lush jungle environment, with dense green foliage covering the ground and towering trees stretching towards the sky. The trees have a mix of thin and thick trunks, some with bark that appears weathered and rugged. The canopy overhead is thick with leaves, allowing only patches of sunlight to filter through and cast dappled shadows on the forest floor below.\\n\\nA variety of paper airplanes, including shades of white, pink, purple, yellow, and green, initially drift gently into the frame, then gradually accelerate as they glide and spin across the scene. The planes vary in size and design, some appearing more complex than others. They gracefully swoop and spiral as they move, some darting forward quickly while others flutter slowly downward, contrasting beautifully against the natural backdrop of the forest.\\n\\nThe scene is peaceful and serene, with the overall atmosphere one of tranquility and harmony between nature and human creativity, the camera panning left to right following the paper airplanes through the dappled jungle light."
WHY: Subject="paper airplanes" (they fly), not "jungle environment" (static setting). The entire first paragraph about trees/canopy is copied word-for-word. Only the airplanes' motion in paragraph 2 gets temporal markers. Paragraph 3's meta-text trimmed, ended with "dappled jungle light".

### Example 3 (sailboats on coffee — multi-paragraph):
INPUT: "The video captures a unique scene of two small sailboats floating on a cup of coffee. The first boat, positioned towards the left side of the frame, is larger and more detailed, with a white sail that has a black symbol on it. The second boat, slightly smaller and to the right, also features a white sail with a distinct black symbol. Both boats have dark brown hulls and appear to be intricately designed.\\n\\nThe coffee in the cup is dark, providing a stark contrast to the light-colored boats. The camera remains steady throughout the video, providing a clear and unobstructed view of the boats and the coffee. There are no other objects or distractions in the frame, keeping the focus solely on the boats and the coffee.\\n\\nOverall, the video is a creative and visually appealing representation of two sailboats on a cup of coffee, with the dark coffee serving as the 'sea' for the boats to sail on."
OUTPUT: "Two small sailboats floating on a cup of coffee. The first boat, positioned towards the left side of the frame, is larger and more detailed, with a white sail that has a black symbol on it. The second boat, slightly smaller and to the right, also features a white sail with a distinct black symbol. Both boats have dark brown hulls and appear to be intricately designed.\\n\\nAs the scene progresses, the two boats initially remain still, then begin to drift slowly around the cup of coffee. The larger boat moves clockwise while the smaller one moves counterclockwise, creating a sense of dynamic movement within the still setting. The contrast between the dark coffee and the light wooden boats creates a striking visual effect. The camera remains steady throughout, allowing viewers to fully absorb the intricate details of the boats as they navigate through the dark coffee surface in gentle circular motion."
WHY: Subject="Two small sailboats" (they float/drift). Deleted "The video captures a unique scene of" and "Overall..." meta-text. Added temporal chain for the boats' motion. Ended with "gentle circular motion". Visual details (dark brown hulls, white sail, black symbol) all preserved verbatim.

### Example 4 (SUV on mountain road — multi-paragraph):
INPUT: "The video depicts a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. A white SUV is seen driving on a dirt road that winds through the mountains. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.\\n\\nThe SUV's tire tracks are visible on the road, and its headlights illuminate the path ahead. The vehicle moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.\\n\\nThe combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience."
OUTPUT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.\\n\\nThe SUV initially appears from the left side of the frame, then accelerates steadily forward along the dirt road, kicking up a growing trail of dust as it moves. The vehicle's tire tracks are visible on the road, and its headlights illuminate the path ahead. The SUV moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.\\n\\nThe combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience with the dust trail billowing behind the vehicle."
WHY: Subject="White SUV" (it drives), not "scenic mountainous landscape" (static). Landscape/vegetation sentences copied verbatim. SUV motion expanded with temporal chain. Ended with "dust trail billowing behind the vehicle".

Output ONLY the restructured prompt. No explanations."""

USER_TEMPLATE = """Restructure this VLM caption ({word_count} words). First, identify what MOVES in this caption — that is your action subject. Then make ONLY 3 changes: (1) move the action subject to the first word (delete 'The video captures/shows/depicts/features...' if present), (2) find the 1-2 sentences about the subject's motion and add a temporal chain (initially/then/gradually), (3) end the last sentence with a key motion/visual word. Copy ALL other sentences VERBATIM — do not rephrase, compress, or merge paragraphs. Delete only meta-text like 'In summary/Overall/This perspective allows...'. Output must be ~{word_count} words (±15%). Do NOT compress.

CRITICAL CONSTRAINT: You MUST preserve at least 90% of the original words in their exact form. Only add/modify the 3 surgical points above. Every adjective, noun, preposition from the original that is not in the motion sentences MUST appear unchanged in your output.

INPUT:
{original_caption}

OUTPUT:"""

# ─────────────────────────────────────────────────────────────────────────────
# 负面 Prompt 生成 System Prompt
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_SYSTEM_PROMPT = """You are a video generation quality expert. Given a positive prompt describing a desired video, generate a NEGATIVE prompt that tells the video model what to AVOID.

## Rules:
1. The negative prompt should describe visual defects and unwanted artifacts specific to the content.
2. Include both GENERIC quality issues and CONTENT-SPECIFIC issues.
3. Generic issues: blurry, low quality, watermark, text overlay, static, overexposed, underexposed, jittery, flickering.
4. Content-specific issues: identify what could go WRONG for this particular subject/scene.
   - For people: extra fingers, deformed face, unnatural body proportions, inconsistent clothing
   - For vehicles: floating wheels, distorted shape, inconsistent reflections
   - For nature: repetitive textures, unnatural colors, frozen motion
   - For animals: extra limbs, distorted anatomy, unnatural movement
5. Keep it concise: 30-60 words max.
6. Output ONLY the negative prompt, no explanations.
7. Write in English.
8. Do NOT use complete sentences — use comma-separated descriptive phrases."""

NEGATIVE_USER_TEMPLATE = """Positive prompt:

{positive_prompt}

Generate a negative prompt (what to AVOID) tailored to this content. Output ONLY the negative prompt:"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用后端
# ─────────────────────────────────────────────────────────────────────────────

def call_dashscope(prompt: str, system: str, model: str, api_key: str,
                   temperature: float = 0.7, max_tokens: int = 1024) -> str:
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
                           temperature: float = 0.7, max_tokens: int = 1024) -> str:
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

def _compute_edit_ratio(text_a: str, text_b: str) -> float:
    """计算两段文本的 token-level 编辑距离比率 (0~1)。
    使用 SequenceMatcher 的 ratio 取反：1 - similarity = edit_ratio。
    """
    from difflib import SequenceMatcher
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    similarity = SequenceMatcher(None, tokens_a, tokens_b).ratio()
    return 1.0 - similarity


def rewrite_caption(original: str, backend: str, model: str,
                    api_base: str = "", api_key: str = "",
                    temperature: float = 0.5, max_retries: int = 2) -> str:
    """对单个 caption 执行融合策略改写（带 length 验证 + diff check）"""
    word_count = len(original.split())
    user_msg = USER_TEMPLATE.format(
        word_count=word_count,
        original_caption=original,
    )

    result = None
    for attempt in range(max_retries + 1):
        # 首次用指定温度，重试时逐步降低增加保守性
        temp = temperature if attempt == 0 else max(0.3, temperature - attempt * 0.1)

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

        # ── 验证 1: 长度检查（不能压缩超过 30%）──
        result_words = len(result.split())
        ratio = result_words / max(word_count, 1)
        if ratio < 0.70:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words}/{word_count} = {ratio:.0%}")
            continue

        # ── 验证 2: diff check（编辑距离不能超过 35%）──
        edit_ratio = _compute_edit_ratio(original, result)
        if edit_ratio > 0.35:
            logger.warning(f"  [重试 {attempt+1}] 改动过大: edit_ratio={edit_ratio:.0%}")
            continue

        # 通过所有验证
        return result

    # 所有重试都失败，返回最后一次结果
    logger.warning(f"  所有重试均未通过验证，使用最后一次结果")
    return result


def generate_negative_prompt(positive_prompt: str, backend: str, model: str,
                             api_base: str = "", api_key: str = "",
                             temperature: float = 0.5) -> str:
    """基于正向 prompt 生成定制化负面 prompt"""
    user_msg = NEGATIVE_USER_TEMPLATE.format(positive_prompt=positive_prompt)

    if backend == "dashscope":
        result = call_dashscope(user_msg, NEGATIVE_SYSTEM_PROMPT, model, api_key, temperature)
    elif backend == "openai":
        result = call_openai_compatible(user_msg, NEGATIVE_SYSTEM_PROMPT, model, api_base, api_key, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # 清理引号
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    if result.startswith("'") and result.endswith("'"):
        result = result[1:-1]

    return result.strip()


def _is_chinese_text(text: str) -> bool:
    """判断文本是否主要为中文（CJK 字符占比 > 30%）"""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cjk_count / len(text) > 0.3


def _estimate_word_count(text: str) -> int:
    """估算等效英文词数：中文按 ~2字/词 换算，英文直接空格分词"""
    if _is_chinese_text(text):
        # 中文字符数 / 2 ≈ 等效英文词数
        cjk_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        non_cjk_words = len(''.join(ch for ch in text if not ('\u4e00' <= ch <= '\u9fff')).split())
        return cjk_chars // 2 + non_cjk_words
    else:
        return len(text.split())


def validate_rewrite(original: str, rewritten: str) -> dict:
    """验证改写质量（词数比例、非空等）"""
    orig_words = _estimate_word_count(original)
    new_words = len(rewritten.split())  # 输出是英文，直接空格分词
    ratio = new_words / orig_words if orig_words > 0 else 0

    # 跨语言改写（中→英）放宽阈值：0.3 ~ 3.0
    is_cross_lingual = _is_chinese_text(original)
    ratio_low = 0.3 if is_cross_lingual else 0.5
    ratio_high = 3.0 if is_cross_lingual else 2.0

    issues = []
    if not rewritten.strip():
        issues.append("empty output")
    if ratio < ratio_low:
        issues.append(f"too short ({new_words} vs {orig_words} words, ratio={ratio:.2f})")
    if ratio > ratio_high:
        issues.append(f"too long ({new_words} vs {orig_words} words, ratio={ratio:.2f})")
    if rewritten.lower().startswith(("the video", "this video", "in this video", "the scene")):
        issues.append("still starts with preamble (Principle 1 violated)")

    return {
        "valid": len(issues) == 0,
        "orig_words": orig_words,
        "new_words": new_words,
        "ratio": ratio,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Prompt 自动改写：将 baseline VLM caption 改写为融合策略 prompt"
    )

    # I/O
    parser.add_argument("--input-dir", type=str, required=True,
                        help="原始 caption 目录（包含 {id}.txt 文件）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录（改写后的 {id}.txt）")
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
                        help="API Key (也可通过环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 设置)")

    # 负面 prompt
    parser.add_argument("--enable-negative", action="store_true",
                        help="为每个正向 prompt 额外生成配套的负面 prompt")
    parser.add_argument("--negative-output-dir", type=str, default="",
                        help="负面 prompt 输出目录 (默认: output-dir 同级的 _negative 后缀目录)")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="生成温度 (默认: 0.2, 低温度提高保留率)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单个样本最大重试次数 (默认: 3)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="请求间隔秒数，避免限流 (默认: 1.0)")

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
    caption_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    if not caption_files:
        logger.error(f"未找到 caption 文件: {input_dir}/*.txt")
        sys.exit(1)

    # 负面 prompt 目录
    neg_output_dir = None
    if args.enable_negative:
        if args.negative_output_dir:
            neg_output_dir = Path(args.negative_output_dir)
        else:
            neg_output_dir = output_dir.parent / (output_dir.name + "_negative")
        neg_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"待处理: {len(caption_files)} 个样本")
    logger.info(f"后端: {args.backend}, 模型: {args.model}")
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")
    if neg_output_dir:
        logger.info(f"负面 prompt 输出: {neg_output_dir}")

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

        # 重试逻辑
        rewritten = None
        validation = None
        for attempt in range(1, args.max_retries + 1):
            try:
                rewritten = rewrite_caption(
                    original=original,
                    backend=args.backend,
                    model=args.model,
                    api_base=args.api_base,
                    api_key=api_key,
                    temperature=args.temperature,
                )
                validation = validate_rewrite(original, rewritten)

                if validation["valid"]:
                    break
                else:
                    logger.warning(
                        f"  [{idx}/{len(caption_files)}] {sample_id} 验证失败 "
                        f"(attempt {attempt}): {validation['issues']}"
                    )
                    if attempt < args.max_retries:
                        time.sleep(args.delay)

            except Exception as e:
                logger.error(f"  [{idx}/{len(caption_files)}] {sample_id} 调用失败 (attempt {attempt}): {e}")
                if attempt < args.max_retries:
                    time.sleep(args.delay * 2)

        if rewritten and validation and validation["valid"]:
            out_file.write_text(rewritten + "\n", encoding="utf-8")
            success += 1
            logger.info(
                f"  [{idx}/{len(caption_files)}] {sample_id} ✓ "
                f"({validation['orig_words']}→{validation['new_words']} words, "
                f"ratio={validation['ratio']:.2f})"
            )
            results.append({
                "sample_id": sample_id,
                "status": "success",
                "orig_words": validation["orig_words"],
                "new_words": validation["new_words"],
                "ratio": validation["ratio"],
            })

            # 生成负面 prompt (仅在改写成功时)
            if neg_output_dir:
                try:
                    neg_prompt = generate_negative_prompt(
                        positive_prompt=rewritten,
                        backend=args.backend,
                        model=args.model,
                        api_base=args.api_base,
                        api_key=api_key,
                        temperature=0.5,
                    )
                    neg_file = neg_output_dir / f"{sample_id}.txt"
                    neg_file.write_text(neg_prompt + "\n", encoding="utf-8")
                    logger.info(f"    → negative prompt: {neg_prompt[:60]}...")
                    time.sleep(args.delay)
                except Exception as e:
                    logger.warning(f"    → negative prompt 生成失败: {e}")

        elif rewritten:
            # 验证失败但有输出，仍然保存（标记警告）
            out_file.write_text(rewritten + "\n", encoding="utf-8")
            success += 1
            logger.warning(
                f"  [{idx}/{len(caption_files)}] {sample_id} ⚠ 保存但有问题: "
                f"{validation['issues'] if validation else 'unknown'}"
            )
            results.append({
                "sample_id": sample_id,
                "status": "warning",
                "issues": validation["issues"] if validation else [],
            })
        else:
            failed += 1
            logger.error(f"  [{idx}/{len(caption_files)}] {sample_id} ✗ 全部重试失败")
            results.append({"sample_id": sample_id, "status": "failed"})

        # 请求间隔
        if idx < len(caption_files):
            time.sleep(args.delay)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! 成功={success}, 失败={failed}, 跳过={skipped}, 总计={len(caption_files)}")
    logger.info(f"输出目录: {output_dir}")

    # 保存处理日志
    log_file = output_dir / "rewrite_log.json"
    log_data = {
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
