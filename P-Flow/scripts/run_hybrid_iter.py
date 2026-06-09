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

REWRITE_SYSTEM = """You are a text-to-video prompt optimizer for the Wan2.1 model (UMT5 text encoder with U-shaped attention: first and last tokens receive ~15x weight).

## CRITICAL RULE: You have NOT seen the video. The input caption was written by a VLM that DID see the video.

Therefore:
- Every color, material, texture, and attribute in the input is GROUND TRUTH — preserve them exactly.
- You MUST NOT invent new colors, materials, textures, or specific visual attributes.
  (Don't add "dark brown hulls" if input just says "boats"; don't add "glass facades" if input just says "buildings")
- You CAN expand: temporal structure, spatial positioning, motion physics, camera descriptions, and atmosphere — these are structural, not visual inventions.

## STRUCTURAL REQUIREMENTS

### 1. SUBJECT-FIRST OPENING (UMT5 first-token = ~15x weight)
Start directly with [Subject noun phrase] + [primary action/state] + [key visual detail from input].
NEVER start with "The video shows/depicts/features/captures..."

### 2. DEEP TEMPORAL CHAIN (critical for motion coherence — this is where you ADD LENGTH)
Every prompt with motion MUST include a DETAILED temporal progression for EACH moving subject:
- Starting position/state in the frame
- Motion onset: how the movement begins (suddenly/gradually/smoothly)
- Motion direction + speed/manner + any acceleration/deceleration
- Mid-motion development: what changes as the action continues
- Physical consequence or evolution visible in the scene

Use varied temporal phrasing:
- "Initially [start state], then [main motion], as the scene progresses [development], gradually [evolution]"
- "At first... then... as momentum builds... eventually..."
- "The [subject] begins by [onset], transitions into [main action], and continues until [end state]"

This is the PRIMARY way to expand word count — through richer motion description, NOT through inventing visual details.

### 3. CAMERA & ATMOSPHERE CLOSING (UMT5 last-token = ~15x weight)
End with camera behavior + a vivid phrase reusing words from the original input:
- "...the camera remains steady throughout, capturing the gentle circular motion on the dark coffee surface."
- "...a wide tracking shot with the dust trail billowing behind the vehicle against the sunlit mountain backdrop."

## WHAT YOU CAN FREELY ADD (safe structural expansions):
- Spatial layout: "positioned left of center", "in the foreground", "receding into the background"
- Motion physics implied by stated actions: "kicking up dust" (if dust IS mentioned), "creating ripples" (if water/liquid IS mentioned)
- Temporal decomposition: "initially... then... as the scene progresses... gradually..."
- Speed/pace descriptors: "at a steady pace", "accelerating smoothly", "drifting slowly"
- Camera language: "medium shot", "camera remains stationary", "tracking shot", "wide angle"
- Relative spatial descriptions: "the larger one on the left" (if multiple subjects exist)
- Lighting direction (generic): "lit from above", "backlit", "silhouetted against"

## WHAT YOU MUST NOT ADD (visual inventions):
- Materials not stated: NO "dark brown hulls", "glass facades", "stone path", "wooden texture"
- Colors not stated: NO "golden light", "pristine white", "deep blue" unless input says so
- Objects/elements not mentioned: NO extra animals, people, furniture, weather
- Specific textures: NO "fluffy coats glisten", "matte ceramic", "brushed metal"
- Emotional interpretation: NO "feeling free", "sense of adventure"
- Sound descriptions: NO "roaring", "splashing sounds"

## OUTPUT SPECIFICATIONS
- 2-4 paragraphs of fluent English prose
- Length: 180-250 words (expand via temporal depth, NOT via inventing details)
- No bullet points, numbered lists, or section headers
- Translate Chinese inputs to English, preserving all factual content
- PRESERVE every noun, adjective, and attribute from the input verbatim
- Output ONLY the final prompt — no explanations, no "Here is..." prefix

## EXAMPLES

### Example 1 (miniature scene — simple motion):
INPUT: "这段视频展示了一杯咖啡中的微型帆船模型。一个白色的杯子里装满了深色的咖啡。在咖啡表面漂浮着两艘微型帆船模型，帆上印有黑色图案。视频开始时两艘帆船静止不动，随后逐渐向彼此靠近。帆船的颜色主要是棕色和白色。帆船上的小人偶清晰可见。整个场景被柔和的光线照亮，背景略微模糊。"

OUTPUT: "Two miniature sailboat models floating on a cup of dark coffee in a white cup, positioned centrally in the frame. The sails feature black patterns, and the boats are primarily brown and white in color. Small figurines on the boats are clearly visible.

The two boats initially remain completely still on the coffee surface, then begin to drift slowly toward each other, their movement barely perceptible at first before becoming a steady convergence. As the scene progresses, the gap between them narrows gradually, each boat maintaining its course toward the other in a smooth and unhurried manner.

The entire scene is illuminated by soft lighting with the background slightly blurred, and the camera holds a steady overhead shot throughout, capturing the delicate movement of the boats as they drift closer together on the dark coffee surface."

### Example 2 (vehicle motion — landscape):
INPUT: "这段视频展示了一辆越野车在山间道路上行驶的场景。车辆是一辆白色的SUV，车顶上装有行李箱。车辆沿着一条蜿蜒的土路行驶，周围是茂密的松树林，远处可以看到连绵起伏的山脉。视频开始时，车辆从画面左侧进入，逐渐驶向镜头方向。随着车辆的移动，尘土飞扬，形成了一片尘雾。阳光透过树叶洒在地面上，形成了斑驳的光影效果。镜头保持相对稳定，跟随车辆的移动而轻微晃动。"

OUTPUT: "White SUV with a roof-mounted cargo box driving on a winding dirt road through dense pine forests with rolling mountains visible in the distance. The vehicle initially enters from the left side of the frame, then moves steadily forward toward the camera along the curving road.

As the SUV progresses, it kicks up dust that rises and spreads into a growing haze behind it, the dust cloud expanding as the vehicle maintains its pace along the unpaved surface. The SUV follows the road's curves, momentarily disappearing partially behind tree cover before re-emerging into clearer view.

Sunlight filters through the tree leaves creating dappled light and shadow patterns on the road surface. The camera remains relatively stable with slight movements tracking the vehicle's forward motion, maintaining the SUV and its billowing dust trail in frame against the mountainous landscape."

### Example 3 (fantasy/underwater — single primary subject):
INPUT: "这段视频展示了一个充满幻想色彩的城市景象，背景设定在水下。画面中有一座现代化的城市，高楼大厦林立，街道上停满了车辆。一只巨大的鲸鱼从画面左侧游入，逐渐向右侧移动。鲸鱼的身体呈现出深灰色，尾巴和鳍部清晰可见。它优雅地游动着，周围有几条小鱼在游动。水面波光粼粼，整体光线较暗，氛围神秘宁静。"

OUTPUT: "Giant dark gray whale swimming gracefully through an underwater fantasy cityscape with modern tall buildings and vehicles parked on the streets below. The whale initially enters from the left side of the frame, then glides steadily rightward through the center of the scene, its movement smooth and unhurried.

As the whale progresses, its tail and fins move in slow rhythmic undulation, propelling it forward at an even pace. Several small fish swim around the whale, adjusting their paths as it passes through the submerged urban landscape. The whale's massive form gradually traverses the full width of the scene.

The water surface shimmers with rippled light while the overall lighting remains dim, creating a mysterious and serene atmosphere. The camera holds a wide stationary shot capturing the whale's complete journey from left to right through the darkened underwater city."

### Example 4 (animals — multiple subjects with interaction):
INPUT: "这段视频展示了两只金毛寻回犬幼犬在雪地中行走。它们的皮毛蓬松，在白雪中显得格外可爱。两只幼犬并排行走，偶尔一只会跑到前面，另一只紧随其后。它们偶尔停下来嗅嗅空气。雪很深，幼犬每走一步爪子都会陷进去。镜头保持稳定。"

OUTPUT: "Two adorable golden retriever puppies with fluffy fur walking through deep snow, appearing especially cute against the white backdrop. The puppies initially move side by side at a matched pace, their paws sinking into the deep snow with each step.

As the scene progresses, one puppy breaks ahead slightly, picking up speed before the other follows closely behind, matching the leader's trajectory. They occasionally pause to sniff the air, momentarily still before resuming their walk. The deep snow requires effort with each step, their paws pressing down and lifting out in a rhythmic trudging motion.

The camera remains steady throughout, maintaining a consistent framing as the two puppies navigate the snowy terrain together, alternating between walking side by side and one leading the other through the deep white snow."

### Example 5 (natural disaster — dramatic scale):
INPUT: "这段视频展示了一次壮观的火山爆发场景。主体是一股巨大的火山灰柱，从火山口喷发而出，迅速向上攀升。火山灰的颜色从深灰色逐渐过渡到浅灰色。背景是起伏的山脉和一片深蓝色的海洋。阳光照射在火山灰上。镜头从下往上移动，逐渐拉远。"

OUTPUT: "Massive volcanic ash column erupting from the crater and rapidly ascending into the sky, the plume's color transitioning from dark gray at the base to lighter gray as it rises higher. The eruption initially begins as a concentrated burst from the crater opening, then expands rapidly upward in a towering column that grows taller with each passing moment.

As the plume ascends, it widens and disperses at its upper reaches while fresh material continues feeding the column from below, maintaining its intense vertical momentum. Rolling mountains form the backdrop with a deep blue ocean visible beyond them. Sunlight illuminates the ash column, highlighting the contrast between the darker lower portions and the lighter upper sections.

The camera begins from a low angle looking upward, then gradually tilts up and pulls back to reveal the full scale of the eruption against the mountain and ocean backdrop."

### Example 6 (human motion — fitness):
INPUT: "这段视频展示了一名肌肉发达的男子穿着白色背心在黑色背景前原地跑步。灯光较暗，画面以柔焦呈现。男子的肌肉在跑步时清晰可见。他最初保持匀速，之后逐渐加速。镜头保持不动，正面拍摄。"

OUTPUT: "Muscular man wearing a white tank top running in place against a black background, his muscles clearly visible during the running motion. The scene is presented with dim lighting and soft focus.

The man initially maintains a steady even pace, his arms and legs moving in controlled rhythmic coordination. Then he gradually accelerates, the tempo of his strides increasing progressively as his movements become more intense and powerful. The transition from steady jogging to faster running is smooth and continuous, with his form remaining centered in the frame throughout.

The camera remains completely stationary, capturing the man from a front-facing perspective. The dim lighting creates a soft silhouette effect that emphasizes his muscular physique against the dark backdrop as he runs with increasing intensity."

## FINAL REMINDER
Your goal: FAITHFUL to source (no visual inventions) + DEEP temporal structure (expand length through motion detail) + STRONG first/last tokens. The first and last words carry 15x weight — make them count."""

REFINE_SYSTEM = """You fix video generation prompts based on VLM feedback. You make SURGICAL fixes — change only what the VLM says is wrong, leave everything else VERBATIM.

## Your constraints:
- You will receive: (1) the current prompt, (2) a VLM analysis of how the prompt differs from the actual video content.
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

VLM FEEDBACK: "MOTION: The SUV actually moves from right to left in the video, not left to right. SUBJECT: The dust trail is barely visible, much less prominent than described."

FIXED PROMPT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight. The SUV initially appears from the right side of the frame, then moves steadily leftward along the dirt road, with a faint trail of dust barely visible behind it."

## Output ONLY the fixed prompt. No explanations. English only."""


# ─────────────────────────────────────────────────────────────────────────────
# VLM 校验提示词（视频 vs 文字对比）
# ─────────────────────────────────────────────────────────────────────────────

VLM_VERIFY_PROMPT = """You are watching a video (shown as key frames) and reading a text prompt that is INTENDED to describe this video for regeneration.

Your task: Compare the VIDEO CONTENT with the TEXT PROMPT and identify any CLEAR FACTUAL ERRORS in the text.

Analyze these 4 dimensions. For each, state whether the text accurately describes the video or not:

1. SUBJECT: Does the text correctly identify the main subject? (species, color, size, count, appearance)
2. MOTION: Does the text correctly describe motion direction, speed, and trajectory? (e.g., left-to-right vs right-to-left, fast vs slow)
3. BACKGROUND: Does the text accurately describe the background elements, colors, and lighting?
4. TIMING: Does the text correctly capture the sequence of events? (what happens first/then/finally)

Format your response as:
SUBJECT: [what's wrong, or "accurate"]
MOTION: [what's wrong, or "accurate"]
BACKGROUND: [what's wrong, or "accurate"]
TIMING: [what's wrong, or "accurate"]

IMPORTANT — What is NOT an error (do NOT flag these):
- Temporal decomposition of actions: "initially appears... then moves... gradually..." is valid structural expansion
- Camera descriptions: "wide shot", "camera remains steady", "tracking shot"
- Spatial positioning: "positioned left of center", "in the foreground"
- Speed/pace descriptions: "at a steady pace", "accelerating smoothly"
- Motion physics that are directly implied by stated actions (e.g., "paws sinking" when walking in deep snow is stated)

IMPORTANT — What IS an error (DO flag these):
- Wrong colors: text says "white" but video shows "black"
- Wrong direction: text says "left to right" but video shows "right to left"
- Wrong count: text says "two dogs" but video shows "three dogs"
- Wrong species/object: text says "cat" but video shows "dog"
- Non-existent major elements: text mentions "rain" but there is no rain
- Wrong action type: text says "running" but video shows "standing still"
- Invented materials/textures not visible: text says "dark brown hulls" but boats are a different color
- Invented colors not in the video: text adds specific colors not visible in the footage

Only flag the 1-2 biggest factual errors that would cause a video model to generate something visually DIFFERENT from the original. If everything is reasonably accurate, just write "accurate" for all dimensions."""


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
    """LLM v7b 改写：严格忠实 + 深度时序链展开（不编造外观细节）"""
    word_count = len(caption.split())
    user_msg = (
        f"Rewrite this VLM caption into a well-structured video generation prompt for Wan2.1 T2V.\n\n"
        f"REMEMBER:\n"
        f"- PRESERVE every visual detail exactly as stated (colors, materials, counts, species)\n"
        f"- DO NOT invent any visual details not in the input (no new colors, materials, textures)\n"
        f"- EXPAND length through DEEP temporal chains: onset → direction → speed → development → consequence\n"
        f"- First sentence: subject noun + action (no \"The video shows...\")\n"
        f"- Last sentence: camera behavior + closing phrase using words from the input\n"
        f"- Target: 180-250 words, 2-4 paragraphs\n\n"
        f"INPUT:\n{caption}\n\n"
        f"OUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.7 if attempt == 0 else max(0.5, 0.7 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)

        # ── 验证 1: 长度检查（≥80词下限，≤250词上限）──
        result_words = len(result.split())
        if result_words < 80:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words} 词 (最低 80)")
            continue
        if result_words > 250:
            logger.warning(f"  [重试 {attempt+1}] 输出过长: {result_words} 词 (最高 250)")
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
    """调用 run.py 生成视频（纯 caption→T2V，不启用迭代/VLM）"""
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
