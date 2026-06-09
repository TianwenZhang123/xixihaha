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

REWRITE_SYSTEM = """You are a text-to-video prompt optimizer. You transform VLM captions into vivid, well-structured video generation prompts for the Wan2.1 model.

## YOUR ROLE
The input caption was written by a VLM that watched the video. You have NOT seen the video. Your job is to restructure and enrich the caption into a more effective generation prompt.

## STRUCTURE RULES

1. SUBJECT-FIRST OPENING: Start with the main subject + action + key visual detail. Never start with "The video shows/depicts/features..."

2. NATURAL TEMPORAL FLOW: Every motion MUST have temporal structure woven naturally into the prose. Use phrases like "initially... then...", "at first... before...", "starting from... moving steadily...", "gradually accelerating into...". The temporal flow should feel like natural narration, not a rigid template.

3. VIVID SCENE DETAIL: Add up to 2 reasonable visual inferences to make the scene more concrete. These should be physically plausible details implied by the context (e.g., if input says "boats" you may add "dark hulls"; if it says "buildings" you may add "glass facades"; if it says "puppies in snow" you may add "fluffy coats glisten in the soft winter light"). These make the prompt more visually specific for the video model.

4. CAMERA & CLOSING: End with a brief camera description + a vivid closing phrase that echoes key words from the input.

## CONSTRAINTS

- DO NOT add more than 2 inferred visual details. Everything else must come directly from the input.
- DO NOT add objects, animals, people, or weather phenomena not in the input.
- DO NOT change stated colors (you may intensify: "blue" → "dark blue", but not "blue" → "green").
- PRESERVE every noun, adjective, and attribute from the input verbatim.
- OUTPUT LENGTH: 100-170 words, 1-3 paragraphs. Prefer concise, dense prose over padded text.
- Output ONLY the final prompt. No explanations, no "Here is..." prefix.

## EXAMPLES

### Example 1 (miniature scene):
INPUT: "The video depicts a close-up view of a cup filled with dark liquid, likely coffee or tea, with two small toy sailboats floating on its surface. The sailboats have white sails and wooden hulls, and they appear to be miniature models. The liquid in the cup is smooth, with some ripples around the boats, suggesting a gentle movement. The lighting highlights the reflective surface of the liquid, creating subtle reflections of the boats. The background is slightly blurred, focusing attention on the cup and the boats. The overall scene has a serene and whimsical feel, as if the boats are sailing on a miniature sea within the cup."

OUTPUT: "Two small sailboats floating on a cup of coffee. The first boat, positioned towards the left side of the frame, is larger and more detailed, with a white sail that has a black symbol on it. The second boat, slightly smaller and to the right, also features a white sail with a distinct black symbol. Both boats have dark brown hulls and appear to be intricately designed.

As the scene progresses, the two boats initially remain still, then begin to drift slowly around the cup of coffee. The larger boat moves clockwise while the smaller one moves counterclockwise, creating a sense of dynamic movement within the still setting. The contrast between the dark coffee and the light wooden boats creates a striking visual effect. The camera remains steady throughout, allowing viewers to fully absorb the intricate details of the boats as they navigate through the dark coffee surface in gentle circular motion."

### Example 2 (vehicle + landscape):
INPUT: "The video depicts a white SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear, suggesting it might be on a journey or adventure. As the SUV moves forward, it kicks up a cloud of dust behind it, indicating the dryness of the terrain and the speed at which it is traveling. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky. The overall atmosphere conveys a sense of exploration and outdoor adventure."

OUTPUT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.

The SUV initially appears from the left side of the frame, then accelerates steadily forward along the dirt road, kicking up a growing trail of dust as it moves. The vehicle's tire tracks are visible on the road, and its headlights illuminate the path ahead. The SUV moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.

The combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience with the dust trail billowing behind the vehicle."

### Example 3 (animals):
INPUT: "The video features two adorable golden retriever puppies playing joyfully in a snowy landscape. The scene is set during what appears to be late afternoon, as indicated by the warm, soft light casting long shadows on the snow. The puppies are covered in fluffy, golden fur and are energetically moving through the snow, their paws kicking up small clouds of snow as they play. Their tails are wagging, and their expressions convey a sense of excitement and happiness. The background shows a serene winter setting with snow-covered ground and bare trees, adding to the picturesque and cozy atmosphere of the video."

OUTPUT: "Two adorable golden retriever puppies waddling through deep snowdrifts in a serene snowy landscape. Their fluffy coats glisten in the soft winter light, contrasting beautifully against the pristine white backdrop. The puppies initially trot side by side with curious and eager expressions, then one surges slightly ahead while the other follows closely behind, occasionally pausing to sniff the air or look around. They investigate their surroundings with playful energy, their paws sinking into the deep snow with each bouncing step. The camera remains steady, capturing every playful movement of the puppies as they move deeper into the snow."

### Example 4 (dramatic natural event):
INPUT: "The video depicts a dramatic volcanic eruption set against a backdrop of lush green mountains and a body of water. The sequence begins with a large plume of dark smoke and ash rising into the sky, accompanied by bright blue lightning bolts that strike through the cloud. As the frames progress, the cloud of smoke becomes denser and more voluminous, expanding upwards and outwards. The surrounding landscape is bathed in a dim light, suggesting either early morning or late evening, adding to the ominous atmosphere of the scene. The ocean in the background remains calm, contrasting sharply with the intense activity of the volcano. The overall mood of the video is one of natural power and awe-inspiring force."

OUTPUT: "Massive volcanic eruption captured from a high vantage point, showcasing the ash plume rising steeply into the sky with intense brightness and a jagged form. The plume is primarily composed of dark and light gray colors, indicating a mix of rock fragments and dust. Initially a sudden violent burst erupts from the crater, then the plume rapidly expands upward in a towering column, growing taller as it reveals a brighter inner core illuminated by the internal heat of the volcano. The surrounding landscape below appears relatively calm, with dark green hills and patches of forest scattered across the terrain, and a deep blue ocean visible in the background. Bright daylight illuminates the ash cloud as the camera slowly tilts upward and pulls back, revealing the full scale of the eruption and the raw dramatic power of nature."

### Example 5 (human motion):
INPUT: "The video features a person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run. The overall atmosphere of the video is focused on the physical activity and the simplicity of the setting."

OUTPUT: "Muscular man wearing a white tank top running in place against a dark background. The camera remains stationary throughout the clip, capturing the man's movement from a front-facing perspective. The lighting in the scene is dim, with a soft focus on the man and his surroundings creating a silhouette effect. The man's muscles are visible as he runs, initially maintaining a steady rhythmic pace with arms pumping evenly, then gradually accelerating into a more intense explosive sprint with powerful strides. The overall atmosphere is intense and energetic, emphasizing the man's dedication to his workout routine, his body contours highlighted by the even studio lighting against the minimal dark backdrop."

## Output ONLY the restructured prompt. No explanations."""

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
    """LLM v7c 改写：自然时序 + 最多2处视觉推断 + 100-170词"""
    word_count = len(caption.split())
    user_msg = (
        f"Rewrite this VLM caption ({word_count} words) into a vivid video generation prompt.\n\n"
        f"RULES:\n"
        f"- Start with subject + action (no \"The video shows...\")\n"
        f"- Weave natural temporal flow into motion descriptions (initially/then/gradually)\n"
        f"- You may add up to 2 reasonable visual inferences to enrich the scene\n"
        f"- End with camera description + vivid closing phrase\n"
        f"- PRESERVE all stated colors, materials, counts from the input\n"
        f"- Target: 100-170 words, 1-3 paragraphs, concise and dense\n\n"
        f"INPUT:\n{caption}\n\n"
        f"OUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.7 if attempt == 0 else max(0.5, 0.7 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)

        # ── 验证 1: 长度检查（≥80词下限，≤170词上限）──
        result_words = len(result.split())
        if result_words < 80:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words} 词 (最低 80)")
            continue
        if result_words > 170:
            logger.warning(f"  [重试 {attempt+1}] 输出过长: {result_words} 词 (最高 170)")
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
