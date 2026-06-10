#!/usr/bin/env python3
"""
Hybrid Pipeline — 单轮改写+校验流程

流程：
  1. 读取 VLM caption（已有的 captions_qwen 等）
  2. LLM 约束式改写 → rewritten prompt
  3. VLM 校验：用原始视频 + 改写后文字对比，输出不一致之处
  4. LLM 修复：根据 VLM 反馈做定向修正 → final prompt
  5. 用 final prompt 生成视频
  6. 评测 CLIP/XCLIP

用法:
    cd /root/xixihaha/P-Flow

    export DASHSCOPE_API_KEY="sk-xxxxx"

    # 快速测试（仅 LLM 改写，不加载 VLM）
    python scripts/run_hybrid_iter.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/hybrid_v5 \
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \
        --skip_vlm

    # 完整流程（含 VLM 校验）
    python scripts/run_hybrid_iter.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/hybrid_v5 \
        --sample_ids 7 17 21 31 32 33 34 43 46 47
"""

import sys
import os
import json
import subprocess
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
# LLM 系统提示词
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You are a text-to-video prompt optimizer. You restructure VLM captions into effective video generation prompts for the Wan2.1 model.

## YOUR ROLE
The input caption was written by a VLM that watched the video. You have NOT seen the video. Your job is to restructure the caption into a tighter, more effective generation prompt — staying extremely faithful to the input content.

## STRUCTURE RULES

1. SUBJECT-FIRST OPENING: Start with the main subject + action + key visual detail. Never start with "The video shows/depicts/features..."

2. NATURAL TEMPORAL FLOW: Weave temporal structure into motion descriptions using phrases like "initially... then...", "at first... before...", "gradually...". This should read as natural narration.

3. ONE VISUAL INFERENCE (maximum): You may add exactly ONE material/texture/lighting detail that is physically implied by the input (e.g., "wooden hulls" → "dark brown hulls"; "buildings" → "glass facades"; "dim light" → "silhouette effect"). This must be a surface-level attribute — NEVER an entire new object, person, animal, action, or weather phenomenon.

4. CAMERA & CLOSING: End with a brief camera note + a short closing phrase.

## MOTION FIDELITY (CRITICAL — highest priority rule)

Motion descriptions MUST be copied VERBATIM from the input. You must NOT:
- Infer or invent motion directions (e.g., "leftward", "clockwise", "right-to-left") unless explicitly stated in the input
- Change motion speed descriptions (e.g., "slowly" → "rapidly")
- Add motion trajectories not described in the input (e.g., adding "drifts leftward" when input only says "floating")
- Speculate about motion patterns (e.g., "circular motion", "zigzag path") not in the input

You MAY only:
- Add temporal connectors (initially/then/finally) to EXISTING motion descriptions
- Restructure the ORDER of existing motion sentences for better flow
- Keep motion verbs as-is: if input says "moves", keep "moves" — do NOT change to "drifts/glides/sweeps"

If the input describes vague motion (e.g., "the boat moves"), keep it vague. Do NOT specify a direction.

## CONSTRAINTS

- MAXIMUM 1 inferred detail (appearance ONLY — never motion).
- NEVER add objects, animals, people, sounds, smells, or phenomena not explicitly stated in the input.
- NEVER change stated colors (intensifying is OK: "blue" → "deep blue").
- PRESERVE all nouns, counts, and attributes from the input.
- OUTPUT LENGTH: 100-150 words, 1-2 paragraphs. Be concise.
- Output ONLY the final prompt. No explanations.

## EXAMPLES

### Example 1 (miniature scene):
INPUT: "The video depicts a close-up view of a cup filled with dark liquid, likely coffee or tea, with two small toy sailboats floating on its surface. The sailboats have white sails and wooden hulls, and they appear to be miniature models. The liquid in the cup is smooth, with some ripples around the boats, suggesting a gentle movement. The lighting highlights the reflective surface of the liquid, creating subtle reflections of the boats. The background is slightly blurred, focusing attention on the cup and the boats. The overall scene has a serene and whimsical feel, as if the boats are sailing on a miniature sea within the cup."

OUTPUT: "Two small sailboats with white sails and dark brown hulls floating on a cup of dark coffee. The miniature boats sit on the smooth liquid surface, with gentle ripples forming around them suggesting subtle movement. The lighting highlights the reflective surface, creating delicate reflections of the boats in the dark liquid. Initially the boats remain relatively still, then begin to drift gently as the ripples spread outward. The background is softly blurred, keeping attention focused on the cup and boats. The camera remains steady in close-up throughout, capturing the serene, whimsical scene of miniature boats sailing on their tiny coffee sea."

### Example 2 (vehicle + landscape):
INPUT: "The video depicts a white SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear, suggesting it might be on a journey or adventure. As the SUV moves forward, it kicks up a cloud of dust behind it, indicating the dryness of the terrain and the speed at which it is traveling. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky. The overall atmosphere conveys a sense of exploration and outdoor adventure."

OUTPUT: "White SUV with roof racks driving forward on a dusty unpaved road through a forested mountainous area. The vehicle kicks up a cloud of dust behind it as it moves, indicating the dryness of the terrain. Tall pine trees and dense vegetation line both sides of the road. The SUV initially moves forward at a steady pace, then continues along the road as the dust trail grows behind it. Distant mountains are visible under a clear blue sky, creating a scenic backdrop. The overall atmosphere conveys exploration and outdoor adventure. The camera follows the vehicle smoothly as it progresses through the rugged terrain."

### Example 3 (animals):
INPUT: "The video features two adorable golden retriever puppies playing joyfully in a snowy landscape. The scene is set during what appears to be late afternoon, as indicated by the warm, soft light casting long shadows on the snow. The puppies are covered in fluffy, golden fur and are energetically moving through the snow, their paws kicking up small clouds of snow as they play. Their tails are wagging, and their expressions convey a sense of excitement and happiness. The background shows a serene winter setting with snow-covered ground and bare trees, adding to the picturesque and cozy atmosphere of the video."

OUTPUT: "Two adorable golden retriever puppies with fluffy golden fur playing joyfully in a serene snowy landscape. The warm, soft light of late afternoon casts long shadows on the snow as the puppies energetically move through the drifts. They initially trot together with excited expressions and wagging tails, then continue playing as their paws kick up small clouds of snow with each step. The background shows a serene winter setting with snow-covered ground and bare trees. The camera remains steady, capturing the puppies' playful energy as they explore the picturesque snowy scene in the warm golden light."

### Example 4 (dramatic natural event):
INPUT: "The video depicts a dramatic volcanic eruption set against a backdrop of lush green mountains and a body of water. The sequence begins with a large plume of dark smoke and ash rising into the sky, accompanied by bright blue lightning bolts that strike through the cloud. As the frames progress, the cloud of smoke becomes denser and more voluminous, expanding upwards and outwards. The surrounding landscape is bathed in a dim light, suggesting either early morning or late evening, adding to the ominous atmosphere of the scene. The ocean in the background remains calm, contrasting sharply with the intense activity of the volcano. The overall mood of the video is one of natural power and awe-inspiring force."

OUTPUT: "Massive volcanic eruption set against lush green mountains and a body of water. The sequence begins with a large plume of dark smoke and ash rising steeply into the sky, accompanied by bright blue lightning bolts striking through the cloud. Initially the plume emerges from the crater, then as the frames progress it becomes denser and more voluminous, expanding upwards and outwards in a towering column. The surrounding landscape is bathed in dim light of early evening, adding an ominous atmosphere. The ocean in the background remains calm, contrasting sharply with the intense volcanic activity. The camera captures the full scale of the eruption, conveying the raw dramatic power of nature."

### Example 5 (human motion):
INPUT: "The video features a person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run. The overall atmosphere of the video is focused on the physical activity and the simplicity of the setting."

OUTPUT: "Athletic person wearing a white tank top and black shorts running against a plain, light-colored background. The soft, even lighting casts minimal shadows, emphasizing the runner's movement and highlighting their athletic build. The person initially jogs at a steady pace with arms swinging naturally, then continues running with consistent rhythmic motion. The overall atmosphere is focused on the physical activity and simplicity of the setting. The camera remains stationary, capturing the runner's movement from a steady perspective against the clean minimal backdrop."

## Output ONLY the restructured prompt. No explanations."""

REFINE_SYSTEM = """You fix video generation prompts based on VLM feedback. You make SURGICAL fixes — change only what the VLM says is wrong, leave everything else VERBATIM.

## PRIORITY ORDER (fix these first):
1. MOTION errors (HIGHEST) — direction, trajectory, speed, pattern
2. SUBJECT errors — wrong object, color, count
3. BACKGROUND errors — missing/wrong elements
4. TIMING errors — wrong sequence

## Your constraints:
- You will receive: (1) the current prompt, (2) a VLM analysis of how the prompt differs from the actual video content.
- Fix ALL motion errors the VLM identified — these are critical for video quality.
- For other dimensions, fix only the issues the VLM flagged.
- The current prompt already follows Subject-First Opening + Temporal Action Chain structure. PRESERVE this structure.

## What you must NOT do:
- Do NOT rewrite the entire prompt. Copy it and make targeted edits.
- Do NOT compress or shorten. Output word count must be within ±15% of input.
- Do NOT rephrase visual descriptions that the VLM did NOT flag.
- Do NOT remove temporal markers (initially/then/gradually) unless VLM says timing is wrong.
- Do NOT change the subject in position 0 unless VLM says the wrong subject is shown.
- Do NOT add new motion directions that the VLM didn't mention. If VLM says "direction is wrong" but doesn't specify the correct direction, use vague motion ("moves gently", "drifts") instead.

## How to fix MOTION feedback:

### "Direction is wrong" (e.g., moves left but should move right):
→ Find the motion sentence, replace the direction with what VLM says is correct.
→ If VLM doesn't specify direction, remove the specific direction and use vague motion.

### "Motion pattern is wrong" (e.g., circular but should be linear):
→ Replace the motion pattern with what VLM describes.

### "Invented motion not in video" (e.g., text says accelerates but video shows steady pace):
→ Remove the invented motion. Replace with what VLM describes, or use simple "moves" if unclear.

### "Motion speed/intensity differs" (e.g., too fast, too slow):
→ Adjust temporal adverbs: "rapidly" → "slowly", "sudden burst" → "gentle emergence".

## How to fix other feedback:

### "Subject appearance differs" (e.g., wrong color, wrong size):
→ Find the subject description, adjust the specific attribute. Keep surrounding sentences.

### "Background/scene differs" (e.g., missing element, wrong lighting):
→ Find the relevant background sentence, add/modify the specific detail.

## Example:

CURRENT PROMPT: "White SUV driving on a dirt road through a mountainous landscape. The SUV initially appears from the left side of the frame, then accelerates steadily forward, kicking up a growing trail of dust as it moves."

VLM FEEDBACK: "MOTION: The SUV moves from right to left in the video, not left to right. The speed is steady, not accelerating. SUBJECT: accurate. BACKGROUND: accurate. TIMING: accurate."

FIXED PROMPT: "White SUV driving on a dirt road through a mountainous landscape. The SUV initially appears from the right side of the frame, then moves steadily leftward along the dirt road, kicking up a trail of dust as it moves."

## Output ONLY the fixed prompt. No explanations. English only."""


# ─────────────────────────────────────────────────────────────────────────────
# VLM 校验提示词（视频 vs 文字对比）
# ─────────────────────────────────────────────────────────────────────────────

VLM_VERIFY_PROMPT = """You are watching a video (shown as key frames) and reading a text prompt that is INTENDED to describe this video for regeneration.

Your task: Compare the VIDEO CONTENT with the TEXT PROMPT and identify FACTUAL ERRORS — especially MOTION errors.

## PRIORITY: MOTION is the most critical dimension. Be extra vigilant about:
- Direction errors: text says "moves left" but video shows rightward motion
- Trajectory errors: text says "circular" but video shows linear motion
- Speed errors: text says "rapidly" but video shows slow movement
- Invented motion: text describes specific motion patterns not visible in the video
- Missing motion: video shows clear movement that text doesn't mention

Analyze these 4 dimensions. For each, state whether the text accurately describes the video:

1. SUBJECT: Does the text correctly identify the main subject? (species, color, size, count, appearance)
2. MOTION (HIGHEST PRIORITY): Does the text correctly describe motion direction, speed, and trajectory? Be STRICT here — if the text specifies a direction (left/right/up/down/clockwise/etc.) that doesn't match what you see in the video, flag it immediately. Also flag if the text invents motion patterns not visible in the video.
3. BACKGROUND: Does the text accurately describe the background elements, colors, and lighting?
4. TIMING: Does the text correctly capture the sequence of events? (what happens first/then/finally)

Format your response as:
SUBJECT: [what's wrong, or "accurate"]
MOTION: [what's wrong, or "accurate"]
BACKGROUND: [what's wrong, or "accurate"]
TIMING: [what's wrong, or "accurate"]

IMPORTANT — What is NOT an error (do NOT flag these):
- Temporal connectors: "initially... then... gradually..." is valid if the actions themselves are correct
- Camera descriptions: "wide shot", "camera remains steady", "tracking shot"
- Spatial positioning that matches the video: "positioned left of center"
- Vague motion: "moves gently", "drifts" without specifying direction — this is acceptable

IMPORTANT — What IS an error (DO flag these):
- Wrong motion direction: text says "left to right" but video shows "right to left" — FLAG THIS
- Wrong motion pattern: text says "clockwise" but video shows random drifting — FLAG THIS
- Invented specific motion: text says "accelerates into a sprint" but video shows constant speed — FLAG THIS
- Wrong colors: text says "white" but video shows "black"
- Wrong count: text says "two dogs" but video shows "three dogs"
- Wrong species/object: text says "cat" but video shows "dog"
- Non-existent elements: text mentions "rain" but there is no rain
- Wrong action type: text says "running" but video shows "standing still"
- Invented materials not visible: text says "dark brown hulls" but boats are a different color

Flag ALL motion-related errors you find — motion accuracy is critical for video regeneration quality. For other dimensions, flag only the biggest 1-2 errors. If everything is accurate, write "accurate" for that dimension."""


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
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
        max_tokens=2048,
    )
    result = response.choices[0].message.content.strip()
    # 清理引号
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    if result.startswith("'") and result.endswith("'"):
        result = result[1:-1]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: LLM 丰富型改写
# ─────────────────────────────────────────────────────────────────────────────

def llm_rewrite(caption: str, model: str = "qwen-plus",
                max_retries: int = 2) -> str:
    """LLM v7d 改写：自然时序 + 最多1处推断(材质/光线) + 100-150词"""
    word_count = len(caption.split())
    user_msg = (
        f"Rewrite this VLM caption ({word_count} words) into a video generation prompt.\n\n"
        f"RULES:\n"
        f"- Start with subject + action (no \"The video shows...\")\n"
        f"- Weave natural temporal flow (initially/then/gradually)\n"
        f"- Maximum 1 inferred detail (material/texture/lighting only)\n"
        f"- NEVER add objects, animals, actions not in the input\n"
        f"- PRESERVE all stated colors, counts, attributes\n"
        f"- Target: 100-150 words, 1-2 paragraphs\n\n"
        f"INPUT:\n{caption}\n\n"
        f"OUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.7 if attempt == 0 else max(0.5, 0.7 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)

        # ── 验证 1: 长度检查（≥80词下限，≤160词上限）──
        result_words = len(result.split())
        if result_words < 80:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words} 词 (最低 80)")
            continue
        if result_words > 160:
            logger.warning(f"  [重试 {attempt+1}] 输出过长: {result_words} 词 (最高 160)")
            continue

        # ── 验证 2: 不能以 preamble 开头 ──
        if result.lower().startswith(("the video", "this video", "in this video")):
            logger.warning(f"  [重试 {attempt+1}] 仍以 preamble 开头")
            continue

        # ── 验证 3: 已移除（v6 策略：放开写，由后续 VLM verify 兜底纠错）──

        return result

    # 所有重试都失败，返回最后一次结果（总比没有好）
    logger.warning(f"  所有重试均未通过验证，使用最后一次结果")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: VLM 校验（原始视频 vs 改写后文字）
# ─────────────────────────────────────────────────────────────────────────────

def vlm_verify_prompt(video_path: str, rewritten_prompt: str, vlm_client) -> str:
    """VLM 看原始视频，对比改写后的 prompt 文字，输出不一致之处。
    
    这是核心创新：不需要生成视频就能发现 prompt 偏差。
    VLM 读视频内容 + 读文字 prompt，判断文字是否准确描述了视频。
    """
    try:
        num_frames = 16 if getattr(vlm_client, 'use_video_mode', True) else 8
        frames_pil = vlm_client._extract_frames_pil(video_path, num_frames=num_frames)
        if not frames_pil:
            return "accurate"  # 无法提取帧，跳过校验

        # 构建 message：视频帧 + 文字 prompt + 验证指令
        content_list = []
        for img in frames_pil:
            content_list.append({"type": "image", "image": img})

        verify_instruction = (
            f"{VLM_VERIFY_PROMPT}\n\n"
            f"## The text prompt to verify:\n"
            f'"{rewritten_prompt}"'
        )
        content_list.append({"type": "text", "text": verify_instruction})

        messages = [
            {"role": "user", "content": content_list},
        ]

        response_text = vlm_client._generate(messages)
        if response_text and len(response_text.strip()) > 10:
            return response_text.strip()

    except Exception as e:
        logger.warning(f"  VLM 校验失败: {e}")

    return "accurate"


def has_real_issues(vlm_feedback: str) -> bool:
    """判断 VLM 反馈是否包含实质性问题（非全部 accurate）"""
    if not vlm_feedback or vlm_feedback == "accurate":
        return False
    lines = vlm_feedback.lower().split("\n")
    for line in lines:
        # 跳过空行和只有标签的行
        line = line.strip()
        if not line:
            continue
        # 如果某一行包含冒号，检查冒号后面的内容
        if ":" in line:
            value = line.split(":", 1)[1].strip()
            # "inaccurate" 包含 "accurate" 子串，需要精确判断
            # 如果值不是纯 "accurate" / "[accurate]"，就认为有问题
            clean_value = value.strip("[] ")
            if clean_value and clean_value != "accurate":
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: LLM 修复
# ─────────────────────────────────────────────────────────────────────────────

def llm_refine(current_prompt: str, vlm_feedback: str, model: str = "qwen-plus",
               max_retries: int = 2) -> str:
    """LLM 根据 VLM 反馈修复（带 length 验证 + diff check）"""
    word_count = len(current_prompt.split())
    user_msg = (
        f"## Current Prompt:\n{current_prompt}\n\n"
        f"## VLM Feedback (what's wrong with the prompt vs actual video):\n"
        f"{vlm_feedback}\n\n"
        f"Fix the prompt based on the VLM feedback. Output ONLY the fixed prompt:"
    )

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
# Step 5: 调用 run.py 生成视频
# ─────────────────────────────────────────────────────────────────────────────

def generate_videos(data_dir: str, caption_dir: str, output_dir: str,
                    sample_ids: list, args) -> None:
    """调用 run.py 生成视频（支持 noise_prior 参数透传）"""
    cmd = [
        sys.executable, "run.py",
        "--data_dir", data_dir,
        "--caption_dir", caption_dir,
        "--output_dir", output_dir,
        "--sample_ids", *[str(s) for s in sample_ids],
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--seed", str(args.seed),
        "--resume",  # 跳过已生成的视频
    ]

    # Layer 2: SVD Noise Prior
    if getattr(args, 'noise_prior', False):
        cmd.extend(["--noise_prior", "--alpha", str(args.alpha)])
        if getattr(args, 'svd_mode', None):
            cmd.extend(["--svd_mode", args.svd_mode])

    logger.info(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.resolve()))
    if result.returncode != 0:
        raise RuntimeError("run.py failed")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: 调用评测脚本
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
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.resolve()))
    if result.returncode != 0:
        logger.error(f"  评测失败")
        return {}

    # 读取结果
    json_path = Path(output_dir) / "eval_results.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────────────────────

def make_flat_dir(output_dir: str, sample_ids: list) -> str:
    """创建 flat 目录（软链接用绝对路径），供评测使用"""
    flat_dir = Path(output_dir).resolve() / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        src = (Path(output_dir).resolve() / f"sample_{sid}" / f"{sid}.mp4")
        dst = flat_dir / f"{sid}.mp4"
        if src.exists():
            dst.unlink(missing_ok=True)
            os.symlink(src, dst)
    return str(flat_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Hybrid Pipeline — 单轮改写+校验流程")

    # I/O
    p.add_argument("--data_dir", type=str, required=True,
                   help="原始视频目录 (如 data/videos)")
    p.add_argument("--caption_dir", type=str, required=True,
                   help="VLM caption 目录 (如 data/captions_qwen)，每个文件命名为 {id}.txt")
    p.add_argument("--output_dir", type=str, required=True,
                   help="本次实验输出目录")
    p.add_argument("--sample_ids", type=int, nargs="+", required=True,
                   help="样本 ID 列表")
    p.add_argument("--baseline_eval", type=str, default=None,
                   help="(可选) baseline 评测结果 JSON 路径，用于最终对比")

    # 生成参数（与 baseline 保持一致）
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    # LLM
    p.add_argument("--llm_model", type=str, default="qwen-plus")

    # VLM（用于 prompt 校验）
    p.add_argument("--vlm_provider", type=str, default="local")
    p.add_argument("--vlm_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct")

    # Noise Prior (Layer 2)
    p.add_argument("--noise_prior", action="store_true",
                   help="启用 SVD Noise Prior (--inversion --svd --blend)")
    p.add_argument("--alpha", type=float, default=0.004,
                   help="噪声混合权重 (推荐 0.001~0.01, 默认 0.004)")
    p.add_argument("--svd_mode", type=str, default=None,
                   choices=["v1", "renorm", "highfreq", "adaptive"],
                   help="SVD 滤波模式 (默认: None, 由 run.py 决定)")

    # 控制
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_vlm", action="store_true",
                   help="跳过 VLM 校验（仅 LLM 改写后直接生成，用于快速测试）")

    args = p.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("需要设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取 baseline 评测结果（可选，用于最终对比）──
    baseline_metrics = {}
    if args.baseline_eval and Path(args.baseline_eval).exists():
        baseline_metrics = json.loads(Path(args.baseline_eval).read_text())
        logger.info(f"Baseline 指标: CLIP={baseline_metrics.get('orig_gen_clip_mean', 'N/A'):.4f}, "
                    f"XCLIP={baseline_metrics.get('orig_gen_xclip_mean', 'N/A'):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: 读取 VLM caption
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: 读取 VLM caption")
    logger.info("=" * 60)

    caption_input_dir = Path(args.caption_dir)
    captions = {}  # sid -> vlm_caption
    for sid in args.sample_ids:
        cap_file = caption_input_dir / f"{sid}.txt"
        if not cap_file.exists():
            logger.error(f"  找不到 caption: {cap_file}")
            continue
        captions[sid] = cap_file.read_text(encoding="utf-8").strip()
        logger.info(f"  [{sid}] 读取 caption ({len(captions[sid].split())} 词): {captions[sid][:50]}...")

    if not captions:
        logger.error("没有找到任何 caption，退出")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: LLM 约束式改写
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: LLM 丰富型改写（受控扩写 + VLM 兜底）")
    logger.info("=" * 60)

    rewrite_dir = out_dir / "captions_rewritten"
    rewrite_dir.mkdir(exist_ok=True)

    rewritten = {}  # sid -> rewritten_prompt
    for sid, caption in captions.items():
        out_file = rewrite_dir / f"{sid}.txt"
        if args.resume and out_file.exists():
            rewritten[sid] = out_file.read_text(encoding="utf-8").strip()
            logger.info(f"  [{sid}] (resume) {rewritten[sid][:50]}...")
            continue

        result = llm_rewrite(caption, args.llm_model)
        rewritten[sid] = result
        out_file.write_text(result, encoding="utf-8")
        logger.info(f"  [{sid}] 改写完成: {result[:50]}...")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: VLM 校验（原始视频 vs 改写文字）
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: VLM 校验（原始视频 vs 改写后文字）")
    logger.info("=" * 60)

    final_dir = out_dir / "captions_final"
    final_dir.mkdir(exist_ok=True)

    if args.skip_vlm:
        logger.info("  --skip_vlm: 跳过 VLM 校验，直接使用改写结果")
        for sid, prompt in rewritten.items():
            (final_dir / f"{sid}.txt").write_text(prompt, encoding="utf-8")
    else:
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

        vlm_feedback_dir = out_dir / "vlm_feedback"
        vlm_feedback_dir.mkdir(exist_ok=True)

        for sid, prompt in rewritten.items():
            video_path = str(Path(args.data_dir) / f"{sid}.mp4")
            if not Path(video_path).exists():
                logger.warning(f"  [{sid}] 原始视频不存在: {video_path}，跳过校验")
                (final_dir / f"{sid}.txt").write_text(prompt, encoding="utf-8")
                continue

            # VLM 校验：看原始视频 + 读改写后文字
            logger.info(f"  [{sid}] VLM 校验中...")
            feedback = vlm_verify_prompt(video_path, prompt, vlm_client)
            (vlm_feedback_dir / f"{sid}.txt").write_text(feedback, encoding="utf-8")
            logger.info(f"  [{sid}] VLM 反馈: {feedback[:80]}...")

            # ── Step 4: 如果有实质问题，LLM 修复 ──
            if has_real_issues(feedback):
                logger.info(f"  [{sid}] 发现偏差，LLM 修复中...")
                refined = llm_refine(prompt, feedback, args.llm_model)
                (final_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
                logger.info(f"  [{sid}] 修复后: {refined[:50]}...")
            else:
                logger.info(f"  [{sid}] VLM 校验通过，无需修复")
                (final_dir / f"{sid}.txt").write_text(prompt, encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: 生成视频
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: 用 final prompt 生成视频")
    logger.info("=" * 60)

    gen_output_dir = str((out_dir / "generated").resolve())
    generate_videos(
        data_dir=str(Path(args.data_dir).resolve()),
        caption_dir=str(final_dir.resolve()),
        output_dir=gen_output_dir,
        sample_ids=list(rewritten.keys()),
        args=args,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: 评测
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: 评测 CLIP/XCLIP")
    logger.info("=" * 60)

    flat_dir = make_flat_dir(gen_output_dir, list(rewritten.keys()))
    eval_output = str((out_dir / "eval_results").resolve())
    metrics = run_eval(
        orig_dir=str(Path(args.data_dir).resolve()),
        gen_dir=flat_dir,
        caption_dir=str(final_dir.resolve()),
        output_dir=eval_output,
    )

    clip_score = metrics.get("orig_gen_clip_mean", 0)
    xclip_score = metrics.get("orig_gen_xclip_mean", 0)

    # ── 汇总输出 ──
    logger.info("\n" + "=" * 60)
    logger.info("汇总")
    logger.info("=" * 60)

    baseline_clip = baseline_metrics.get("orig_gen_clip_mean", 0)
    baseline_xclip = baseline_metrics.get("orig_gen_xclip_mean", 0)

    print(f"\n{'─' * 60}")
    print(f"{'Method':<15} {'CLIP':>10} {'Δ CLIP':>10} {'XCLIP':>10} {'Δ XCLIP':>10}")
    print(f"{'─' * 60}")
    if baseline_metrics:
        print(f"{'Baseline':<15} {baseline_clip:>10.4f} {'—':>10} {baseline_xclip:>10.4f} {'—':>10}")

    d_clip = clip_score - baseline_clip if baseline_metrics else 0
    d_xclip = xclip_score - baseline_xclip if baseline_metrics else 0
    method_name = "Hybrid-v6" if not args.skip_vlm else "Hybrid-noVLM"
    if baseline_metrics:
        print(f"{method_name:<15} {clip_score:>10.4f} {d_clip:>+10.4f} {xclip_score:>10.4f} {d_xclip:>+10.4f}")
    else:
        print(f"{method_name:<15} {clip_score:>10.4f} {'—':>10} {xclip_score:>10.4f} {'—':>10}")
    print(f"{'─' * 60}\n")

    # 保存汇总 JSON
    summary = {
        "hybrid_v5": {"orig_gen_clip": clip_score, "orig_gen_xclip": xclip_score},
        "config": {
            "strategy": "v6_enrichment_single_pass",
            "vlm_verify": not args.skip_vlm,
            "llm_model": args.llm_model,
            "vlm_provider": args.vlm_provider if not args.skip_vlm else "skipped",
            "caption_dir": args.caption_dir,
            "sample_ids": list(rewritten.keys()),
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
    }
    if baseline_metrics:
        summary["baseline"] = {"orig_gen_clip": baseline_clip, "orig_gen_xclip": baseline_xclip}
        summary["delta"] = {"clip": d_clip, "xclip": d_xclip}
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
