#!/usr/bin/env python3
"""
Hybrid Iterative Pipeline — 一体化脚本

流程：
  1. 读取 baseline 的 VLM caption
  2. LLM 融合策略改写 → iter0 prompt
  3. 循环 N 轮：
     a. 用当前 prompt 调 run.py 生成视频
     b. 调评测脚本算 CLIP/XCLIP
     c. VLM 对比（原始 vs 生成）→ 差异分析
     d. LLM 根据反馈修复 prompt
  4. 汇总所有轮次指标 + baseline 对比

用法:
    cd /root/autodl-tmp/videofake/P-Flow

    export DASHSCOPE_API_KEY="sk-xxxxx"

    python scripts/run_hybrid_iter.py \
        --data_dir /root/autodl-tmp/data/video-200/water_mark_out \
        --baseline_dir /root/autodl-tmp/outputs/baseline \
        --output_dir /root/autodl-tmp/outputs/hybrid_iter \
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \
        --iter 3
"""

import sys
import os
import json
import subprocess
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM 改写
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You restructure VLM video captions into better T2V generation prompts. Your output must be nearly identical to the input — you only make 3 surgical changes.

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
INPUT: "The video showcases a vibrant and lush jungle environment, with dense green foliage covering the ground and towering trees stretching towards the sky. The trees have a mix of thin and thick trunks, some with bark that appears weathered and rugged. The canopy overhead is thick with leaves, allowing only patches of sunlight to filter through and cast dappled shadows on the forest floor below.\n\nA variety of colorful paper airplanes, including shades of white, pink, purple, yellow, and green, are seen flying through the air. The planes vary in size and design, some appearing more complex than others. They gracefully glide and spin as they move across the frame, contrasting beautifully against the natural backdrop of the forest.\n\nThe scene is peaceful and serene, with the gentle rustling of leaves and the occasional chirping sounds of birds providing a soothing soundtrack to the visual display. The overall atmosphere is one of tranquility and harmony between nature and human creativity in a beautiful jungle setting."
OUTPUT: "Colorful paper airplanes flying through a vibrant and lush jungle environment, with dense green foliage covering the ground and towering trees stretching towards the sky. The trees have a mix of thin and thick trunks, some with bark that appears weathered and rugged. The canopy overhead is thick with leaves, allowing only patches of sunlight to filter through and cast dappled shadows on the forest floor below.\n\nA variety of paper airplanes, including shades of white, pink, purple, yellow, and green, initially drift gently into the frame, then gradually accelerate as they glide and spin across the scene. The planes vary in size and design, some appearing more complex than others. They gracefully swoop and spiral as they move, some darting forward quickly while others flutter slowly downward, contrasting beautifully against the natural backdrop of the forest.\n\nThe scene is peaceful and serene, with the overall atmosphere one of tranquility and harmony between nature and human creativity, the camera panning left to right following the paper airplanes through the dappled jungle light."
WHY: Subject="paper airplanes" (they fly), not "jungle environment" (static setting). The entire first paragraph about trees/canopy is copied word-for-word. Only the airplanes' motion in paragraph 2 gets temporal markers. Paragraph 3's meta-text trimmed, ended with "dappled jungle light".

### Example 3 (sailboats on coffee — multi-paragraph):
INPUT: "The video captures a unique scene of two small sailboats floating on a cup of coffee. The first boat, positioned towards the left side of the frame, is larger and more detailed, with a white sail that has a black symbol on it. The second boat, slightly smaller and to the right, also features a white sail with a distinct black symbol. Both boats have dark brown hulls and appear to be intricately designed.\n\nThe coffee in the cup is dark, providing a stark contrast to the light-colored boats. The camera remains steady throughout the video, providing a clear and unobstructed view of the boats and the coffee. There are no other objects or distractions in the frame, keeping the focus solely on the boats and the coffee.\n\nOverall, the video is a creative and visually appealing representation of two sailboats on a cup of coffee, with the dark coffee serving as the 'sea' for the boats to sail on."
OUTPUT: "Two small sailboats floating on a cup of coffee. The first boat, positioned towards the left side of the frame, is larger and more detailed, with a white sail that has a black symbol on it. The second boat, slightly smaller and to the right, also features a white sail with a distinct black symbol. Both boats have dark brown hulls and appear to be intricately designed.\n\nAs the scene progresses, the two boats initially remain still, then begin to drift slowly around the cup of coffee. The larger boat moves clockwise while the smaller one moves counterclockwise, creating a sense of dynamic movement within the still setting. The contrast between the dark coffee and the light wooden boats creates a striking visual effect. The camera remains steady throughout, allowing viewers to fully absorb the intricate details of the boats as they navigate through the dark coffee surface in gentle circular motion."
WHY: Subject="Two small sailboats" (they float/drift). Deleted "The video captures a unique scene of" and "Overall..." meta-text. Added temporal chain for the boats' motion. Ended with "gentle circular motion". Visual details (dark brown hulls, white sail, black symbol) all preserved verbatim.

### Example 4 (SUV on mountain road — multi-paragraph):
INPUT: "The video depicts a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. A white SUV is seen driving on a dirt road that winds through the mountains. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.\n\nThe SUV's tire tracks are visible on the road, and its headlights illuminate the path ahead. The vehicle moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.\n\nThe combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience."
OUTPUT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.\n\nThe SUV initially appears from the left side of the frame, then accelerates steadily forward along the dirt road, kicking up a growing trail of dust as it moves. The vehicle's tire tracks are visible on the road, and its headlights illuminate the path ahead. The SUV moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.\n\nThe combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience with the dust trail billowing behind the vehicle."
WHY: Subject="White SUV" (it drives), not "scenic mountainous landscape" (static). Landscape/vegetation sentences copied verbatim. SUV motion expanded with temporal chain. Ended with "dust trail billowing behind the vehicle".

Output ONLY the restructured prompt. No explanations."""

REFINE_SYSTEM = """You fix video generation prompts based on VLM feedback. You make SURGICAL fixes — change only what the VLM says is wrong, leave everything else VERBATIM.

## Your constraints:
- You will receive: (1) the current prompt, (2) a VLM comparison between reference video and generated video.
- Fix ONLY the top 1-2 differences the VLM identified. Do NOT touch anything else.
- The current prompt already follows Subject-First Opening + Temporal Action Chain structure. PRESERVE this structure.

## What you must NOT do:
- Do NOT rewrite the entire prompt. Copy it and make targeted edits.
- Do NOT compress or shorten. Output word count must be within ±15% of input.
- Do NOT rephrase visual descriptions that the VLM did NOT flag. "dark brown hulls" stays "dark brown hulls".
- Do NOT remove temporal markers (initially/then/gradually) unless VLM says timing is wrong.
- Do NOT change the subject in position 0 unless VLM says the wrong subject is shown.

## How to fix common VLM feedback:

### "Motion direction is wrong" (e.g., moves left but should move right):
→ Find the motion sentence, change direction words only. Keep everything else.

### "Subject appearance differs" (e.g., wrong color, wrong size):
→ Find the subject description, adjust the specific attribute. Keep surrounding sentences.

### "Background/scene differs" (e.g., missing element, wrong lighting):
→ Find the relevant background sentence, add/modify the specific detail. Keep other descriptions.

### "Motion speed/intensity differs" (e.g., too fast, too slow):
→ Adjust temporal adverbs: "rapidly" → "slowly", "sudden burst" → "gentle emergence". Keep structure.

## Example:

CURRENT PROMPT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight. The SUV initially appears from the left side of the frame, then accelerates steadily forward along the dirt road, kicking up a growing trail of dust as it moves."

VLM FEEDBACK: "In the reference video, the SUV moves from right to left, but in the generated video it moves left to right. Also the dust trail is barely visible in the reference."

FIXED PROMPT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight. The SUV initially appears from the right side of the frame, then moves steadily leftward along the dirt road, with a faint trail of dust barely visible behind it."

WHY: Changed "left→right" to "right→left" (direction fix), changed "growing trail of dust" to "faint trail of dust barely visible" (intensity fix). Everything else copied verbatim.

## Output ONLY the fixed prompt. No explanations. English only."""


def _compute_edit_ratio(text_a: str, text_b: str) -> float:
    """计算两段文本的 token-level 编辑距离比率 (0~1)。
    使用 SequenceMatcher 的 ratio 取反：1 - similarity = edit_ratio。
    """
    from difflib import SequenceMatcher
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    similarity = SequenceMatcher(None, tokens_a, tokens_b).ratio()
    return 1.0 - similarity


def call_llm(prompt: str, system: str, model: str = "qwen-plus",
             temperature: float = 0.5) -> str:
    """调用 DashScope LLM"""
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    result = response.choices[0].message.content.strip()
    # 清理引号
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    if result.startswith("'") and result.endswith("'"):
        result = result[1:-1]
    return result


def llm_rewrite(caption: str, model: str = "qwen-plus",
                max_retries: int = 2) -> str:
    """LLM 融合策略初始改写（带 length 验证 + diff check）"""
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

    for attempt in range(max_retries + 1):
        # 首次用 0.5，重试时逐步降低 temperature 增加保守性
        temp = 0.5 if attempt == 0 else max(0.3, 0.5 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)

        # ── 验证 1: 长度检查（不能压缩超过 30%）──
        result_words = len(result.split())
        ratio = result_words / max(word_count, 1)
        if ratio < 0.70:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words}/{word_count} = {ratio:.0%}")
            continue

        # ── 验证 2: diff check（编辑距离不能超过 50%）──
        edit_ratio = _compute_edit_ratio(caption, result)
        if edit_ratio > 0.50:
            logger.warning(f"  [重试 {attempt+1}] 改动过大: edit_ratio={edit_ratio:.0%}")
            continue

        return result

    # 所有重试都失败，返回最后一次结果（总比没有好）
    logger.warning(f"  所有重试均未通过验证，使用最后一次结果")
    return result


def llm_refine(current_prompt: str, vlm_feedback: str, model: str = "qwen-plus",
               max_retries: int = 2) -> str:
    """LLM 根据 VLM 反馈修复（带 length 验证 + diff check）"""
    word_count = len(current_prompt.split())
    user_msg = f"## Current Prompt:\n{current_prompt}\n\n## VLM Feedback:\n{vlm_feedback}\n\nFix the prompt. Output ONLY the fixed prompt:"

    for attempt in range(max_retries + 1):
        temp = 0.4 if attempt == 0 else max(0.2, 0.4 - attempt * 0.1)
        result = call_llm(user_msg, REFINE_SYSTEM, model, temperature=temp)

        # ── 验证 1: 长度检查 ──
        result_words = len(result.split())
        ratio = result_words / max(word_count, 1)
        if ratio < 0.70:
            logger.warning(f"  [REFINE 重试 {attempt+1}] 输出过短: {result_words}/{word_count} = {ratio:.0%}")
            continue

        # ── 验证 2: diff check（修复阶段允许更小的改动，阈值 35%）──
        edit_ratio = _compute_edit_ratio(current_prompt, result)
        if edit_ratio > 0.35:
            logger.warning(f"  [REFINE 重试 {attempt+1}] 改动过大: edit_ratio={edit_ratio:.0%}")
            continue

        return result

    logger.warning(f"  REFINE 所有重试均未通过验证，使用最后一次结果")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 调用 run.py 生成视频
# ─────────────────────────────────────────────────────────────────────────────

def generate_videos(data_dir: str, caption_dir: str, output_dir: str,
                    sample_ids: list, args) -> None:
    """调用 run.py 生成视频"""
    cmd = [
        sys.executable, "run.py",
        "--data_dir", data_dir,
        "--caption_dir", caption_dir,
        "--output_dir", output_dir,
        "--sample_ids", *[str(s) for s in sample_ids],
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--seed", str(args.seed),
        "--vlm_provider", "mock",  # 不需要 VLM（我们自己管 prompt）
    ]
    if args.resume:
        cmd.append("--resume")

    logger.info(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent),
                           capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  run.py 失败:\n{result.stderr[-500:]}")
        raise RuntimeError("run.py failed")


# ─────────────────────────────────────────────────────────────────────────────
# 调用评测脚本
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(orig_dir: str, gen_dir: str, caption_dir: str,
             output_dir: str) -> dict:
    """调用 evaluation/run_clip_xclip_eval.py，返回评测结果"""
    cmd = [
        sys.executable, "evaluation/run_clip_xclip_eval.py",
        "--orig-dir", orig_dir,
        "--gen-dir", gen_dir,
        "--caption-dir", caption_dir,
        "--output-dir", output_dir,
    ]
    logger.info(f"  评测: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent),
                           capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  评测失败:\n{result.stderr[-500:]}")
        return {}

    # 读取结果
    json_path = Path(output_dir) / "eval_results.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# VLM 对比（结构化对比 prompt）
# ─────────────────────────────────────────────────────────────────────────────

VLM_COMPARE_PROMPT = """You are comparing two videos shown as key frames. The TOP half is the REFERENCE (ground truth), the BOTTOM half is the GENERATED video.

Analyze the differences in these 4 dimensions ONLY. Be specific and concise:

1. SUBJECT: What is the main moving entity? Is it the same in both? (species, color, size, count)
2. MOTION: Is the motion direction, speed, and trajectory the same? (left→right vs right→left, fast vs slow, straight vs curved)
3. BACKGROUND: Are the background elements, colors, and lighting consistent?
4. TIMING: Does the action sequence match? (what happens first/then/finally)

Format your response as:
SUBJECT: [differences or "matches"]
MOTION: [differences or "matches"]
BACKGROUND: [differences or "matches"]
TIMING: [differences or "matches"]

If a dimension matches perfectly, just write "matches". Focus on the 1-2 biggest differences that would matter most for prompt correction."""


def vlm_compare(ref_video: str, gen_video: str, vlm_client) -> str:
    """VLM 对比两个视频，返回结构化差异分析"""
    from src.video_utils import load_video, save_video_tensor, create_vertical_composite

    # 加载并拼接（CPU 上操作）
    ref = load_video(ref_video, num_frames=81, height=480, width=832, device="cpu")
    gen = load_video(gen_video, num_frames=81, height=480, width=832, device="cpu")
    composite = create_vertical_composite([ref, gen])
    composite_path = "/tmp/hybrid_iter_composite.mp4"
    save_video_tensor(composite, composite_path, fps=15)
    del ref, gen, composite

    try:
        # 直接用 VLM 的底层接口，传自定义结构化 prompt
        num_frames = 16 if getattr(vlm_client, 'use_video_mode', True) else 8
        frames_pil = vlm_client._extract_frames_pil(composite_path, num_frames=num_frames)
        if not frames_pil:
            return "Unable to extract frames for comparison."

        content_list = []
        for img in frames_pil:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": VLM_COMPARE_PROMPT})

        messages = [
            {"role": "user", "content": content_list},
        ]

        response_text = vlm_client._generate(messages)
        if response_text and len(response_text.strip()) > 10:
            return response_text.strip()

    except Exception as e:
        logger.warning(f"  VLM structured compare failed: {e}, falling back to analyze_and_refine")
        # Fallback: 用原来的通用接口
        try:
            result = vlm_client.analyze_and_refine(
                composite_video_path=composite_path,
                current_prompt="[Comparing reference vs generated]",
                iteration=1,
                i_max=1,
            )
            analysis = result.get("analysis", {})
            comparison = analysis.get("comparison", "")
            if comparison:
                return comparison
            parts = []
            if analysis.get("reference_description"):
                parts.append(f"Reference: {analysis['reference_description']}")
            if analysis.get("new_generated_description"):
                parts.append(f"Generated: {analysis['new_generated_description']}")
            if parts:
                return "\n".join(parts)
        except Exception as e2:
            logger.warning(f"  VLM fallback also failed: {e2}")

    return "Unable to analyze differences."


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def make_flat_dir(output_dir: str, sample_ids: list) -> str:
    """创建 flat 目录（软链接），供评测使用"""
    flat_dir = Path(output_dir) / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        sample_dir = Path(output_dir) / f"sample_{sid}"
        src = sample_dir / f"{sid}.mp4"
        dst = flat_dir / f"{sid}.mp4"
        if src.exists():
            dst.unlink(missing_ok=True)
            os.symlink(src, dst)
    return str(flat_dir)


def main():
    p = argparse.ArgumentParser(description="Hybrid Iterative Pipeline 一体化脚本")

    # I/O
    p.add_argument("--data_dir", type=str, required=True,
                   help="原始视频目录")
    p.add_argument("--baseline_dir", type=str, required=True,
                   help="baseline 输出目录（读取 VLM caption + 评测对比）")
    p.add_argument("--output_dir", type=str, required=True,
                   help="本次实验输出目录")
    p.add_argument("--sample_ids", type=int, nargs="+", required=True,
                   help="样本 ID 列表")

    # 迭代
    p.add_argument("--iter", type=int, default=3, help="迭代轮数")

    # 生成参数（与 baseline 保持一致）
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    # LLM
    p.add_argument("--llm_model", type=str, default="qwen-plus")

    # VLM（用于迭代对比）
    p.add_argument("--vlm_provider", type=str, default="local")
    p.add_argument("--vlm_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct")

    # 控制
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_vlm", action="store_true",
                   help="跳过 VLM 对比（仅用 LLM 自主迭代改写，用于快速测试）")

    args = p.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("需要设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取 baseline 评测结果（用于最终对比）──
    baseline_eval_path = Path(args.baseline_dir) / "eval_clip" / "eval_results.json"
    baseline_metrics = {}
    if baseline_eval_path.exists():
        baseline_metrics = json.loads(baseline_eval_path.read_text())
        logger.info(f"Baseline 指标: CLIP={baseline_metrics.get('orig_gen_clip_mean', 'N/A'):.4f}, "
                    f"XCLIP={baseline_metrics.get('orig_gen_xclip_mean', 'N/A'):.4f}")

    # ── Step 1: 读取 baseline VLM caption + LLM 初始改写 ──
    logger.info("=" * 60)
    logger.info("Step 1: 读取 baseline caption → LLM 融合策略改写")
    logger.info("=" * 60)

    caption_dir_iter0 = out_dir / "captions_iter0"
    caption_dir_iter0.mkdir(exist_ok=True)

    for sid in args.sample_ids:
        out_file = caption_dir_iter0 / f"{sid}.txt"
        if args.resume and out_file.exists():
            continue

        # 读 baseline 的 VLM caption
        cap_file = Path(args.baseline_dir) / f"sample_{sid}" / "vlm_caption.txt"
        if not cap_file.exists():
            logger.error(f"  找不到 baseline caption: {cap_file}")
            continue
        vlm_caption = cap_file.read_text(encoding="utf-8").strip()

        # LLM 改写
        hybrid_prompt = llm_rewrite(vlm_caption, args.llm_model)
        out_file.write_text(hybrid_prompt, encoding="utf-8")
        logger.info(f"  [{sid}] {vlm_caption[:40]}... → {hybrid_prompt[:40]}...")

    # ── Step 2-N: 迭代循环 ──
    all_iter_metrics = []

    for iteration in range(1, args.iter + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Iteration {iteration}/{args.iter}")
        logger.info(f"{'=' * 60}")

        # 当前轮的 caption 目录
        if iteration == 1:
            current_caption_dir = str(caption_dir_iter0)
        else:
            current_caption_dir = str(out_dir / f"captions_iter{iteration - 1}")

        # 当前轮的输出目录
        iter_output_dir = str(out_dir / f"gen_iter{iteration}")

        # ── 生成视频 ──
        logger.info(f"  [生成] 使用 caption: {current_caption_dir}")
        generate_videos(
            data_dir=args.data_dir,
            caption_dir=current_caption_dir,
            output_dir=iter_output_dir,
            sample_ids=args.sample_ids,
            args=args,
        )

        # ── 创建 flat 目录 ──
        flat_dir = make_flat_dir(iter_output_dir, args.sample_ids)

        # ── 评测 ──
        eval_output = str(out_dir / f"eval_iter{iteration}")
        metrics = run_eval(
            orig_dir=args.data_dir,
            gen_dir=flat_dir,
            caption_dir=current_caption_dir,
            output_dir=eval_output,
        )

        clip_score = metrics.get("orig_gen_clip_mean", 0)
        xclip_score = metrics.get("orig_gen_xclip_mean", 0)
        logger.info(f"  [评测] Iter {iteration}: CLIP={clip_score:.4f}, XCLIP={xclip_score:.4f}")

        all_iter_metrics.append({
            "iteration": iteration,
            "orig_gen_clip": clip_score,
            "orig_gen_xclip": xclip_score,
            "caption_dir": current_caption_dir,
            "gen_dir": iter_output_dir,
        })

        # ── VLM 对比 + LLM 修复（非最后一轮）──
        if iteration < args.iter:
            next_caption_dir = out_dir / f"captions_iter{iteration}"
            next_caption_dir.mkdir(exist_ok=True)

            if args.skip_vlm:
                # 跳过 VLM，直接用 LLM 自主改写（基于上一轮 prompt 微调）
                logger.info(f"  [LLM] 自主迭代改写（无 VLM 反馈）...")
                for sid in args.sample_ids:
                    cur_prompt = Path(current_caption_dir, f"{sid}.txt").read_text(encoding="utf-8").strip()
                    refined = llm_refine(cur_prompt, "Try to improve motion description and subject clarity.", args.llm_model)
                    (next_caption_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
            else:
                # VLM 对比 + LLM 修复
                logger.info(f"  [VLM+LLM] 对比分析 + 修复改写...")
                from src.vlm_client import create_vlm_client
                vlm_client = create_vlm_client({
                    "provider": args.vlm_provider,
                    "model_path": args.vlm_path,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "max_retries": 3,
                    "use_video_mode": True,
                    "lazy_load": True,
                })

                for sid in args.sample_ids:
                    ref_video = str(Path(args.data_dir) / f"{sid}.mp4")
                    gen_video = str(Path(iter_output_dir) / f"sample_{sid}" / f"{sid}.mp4")
                    cur_prompt = Path(current_caption_dir, f"{sid}.txt").read_text(encoding="utf-8").strip()

                    if not Path(gen_video).exists():
                        logger.warning(f"  [{sid}] 生成视频不存在，跳过")
                        (next_caption_dir / f"{sid}.txt").write_text(cur_prompt, encoding="utf-8")
                        continue

                    # VLM 对比
                    feedback = vlm_compare(ref_video, gen_video, vlm_client)
                    logger.info(f"  [{sid}] VLM: {feedback[:60]}...")

                    # LLM 修复
                    refined = llm_refine(cur_prompt, feedback, args.llm_model)
                    (next_caption_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
                    logger.info(f"  [{sid}] 修复: {refined[:50]}...")

    # ── 汇总输出 ──
    logger.info(f"\n{'=' * 60}")
    logger.info("汇总: 各轮次指标 vs Baseline")
    logger.info(f"{'=' * 60}")

    baseline_clip = baseline_metrics.get("orig_gen_clip_mean", 0)
    baseline_xclip = baseline_metrics.get("orig_gen_xclip_mean", 0)

    print(f"\n{'─' * 70}")
    print(f"{'Iter':<6} {'CLIP':>10} {'Δ CLIP':>10} {'XCLIP':>10} {'Δ XCLIP':>10}")
    print(f"{'─' * 70}")
    print(f"{'base':<6} {baseline_clip:>10.4f} {'—':>10} {baseline_xclip:>10.4f} {'—':>10}")

    for m in all_iter_metrics:
        d_clip = m["orig_gen_clip"] - baseline_clip
        d_xclip = m["orig_gen_xclip"] - baseline_xclip
        print(f"{'iter' + str(m['iteration']):<6} {m['orig_gen_clip']:>10.4f} {d_clip:>+10.4f} "
              f"{m['orig_gen_xclip']:>10.4f} {d_xclip:>+10.4f}")
    print(f"{'─' * 70}\n")

    # 保存汇总 JSON
    summary = {
        "baseline": {"orig_gen_clip": baseline_clip, "orig_gen_xclip": baseline_xclip},
        "iterations": all_iter_metrics,
        "config": {
            "iter": args.iter,
            "llm_model": args.llm_model,
            "vlm_provider": args.vlm_provider if not args.skip_vlm else "skipped",
            "sample_ids": args.sample_ids,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
