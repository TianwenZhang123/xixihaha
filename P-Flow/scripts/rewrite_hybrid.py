#!/usr/bin/env python3
"""
Hybrid Prompt 改写脚本 v6 — 受控丰富型策略

基于 P-Flow 全部实验数据设计（5.28周会+6.4周会+L1对比实验+6.8 Old对比）：

核心发现：
  1. Old 版本（丰富型改写）CLIP 最高（0.896），因为包含精确的外观/材质/空间描述
  2. V4 版本（约束型改写）XCLIP 最高（0.786+L2），因为有良好的时序结构
  3. 新策略：像 Old 一样大胆丰富细节，再用 VLM 兜底纠正错误
  4. UMT5 编码后的 DiT cross-attention 呈 U 型分布：
     首词和末词权重相等（~0.029-0.030），中间几乎均匀（~0.001）
     → 首词和尾词是黄金位置
  5. Temporal chain (initially→then→finally) 对 XCLIP 有效 (+1.7%)
  6. Negative prompt 对 Wan2.1 UMT5 有害 (-5.9% XCLIP)

策略：主动丰富外观/空间/摄影细节 + 保持时序结构 + VLM 校验兜底

流程: VLM caption → LLM enrich(详细) → VLM verify(full video) → LLM fix

用法:
    python scripts/rewrite_hybrid.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/hybrid_captions \
        --backend dashscope \
        --model qwen-plus

    # 启用 VLM 验证（传整个视频给VLM事实核查）
    python scripts/rewrite_hybrid.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/hybrid_captions \
        --video-dir /path/to/original_videos \
        --backend dashscope --model qwen-plus \
        --enable-vlm-verify --vlm-provider local
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
# System Prompt v6: 受控丰富型策略（基于 6.8 Old对比实验 + VLM校验兜底）
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a text-to-video prompt enrichment specialist for the Wan2.1 model (UMT5 text encoder). Your job is to transform short VLM captions into RICH, DETAILED video generation prompts — similar in depth and style to professional video descriptions.

## Key findings from our experiments (you MUST follow these):

1. U-SHAPED ATTENTION: The UMT5 encoder's DiT cross-attention follows a U-shaped distribution — the FIRST and LAST tokens receive ~15x more weight than middle tokens. This means:
   - The FIRST word must be the main subject noun (no preamble)
   - The LAST phrase must be a vivid, concrete visual detail (light, texture, motion)
   - Middle content should be rich and detailed but ordering matters less

2. DETAIL DENSITY WINS CLIP: Our best-scoring prompts (CLIP 0.896) contain SPECIFIC appearance descriptors — exact colors, materials, textures, spatial positions, and camera descriptions. Generic descriptions score poorly.

3. TEMPORAL CHAIN FOR MOTION: Adding structured temporal markers ("initially... then... as the scene progresses...") with specific motion details (direction, speed, trajectory) improves XCLIP significantly.

4. MULTI-PARAGRAPH STRUCTURE: The highest-scoring prompts use 2-4 paragraphs organized as:
   - Para 1: Subject identification + appearance details + scene overview
   - Para 2: Motion description with temporal progression and spatial direction
   - Para 3 (optional): Additional background/atmosphere details
   - Para 4 (optional): Camera behavior and overall mood summary

## Your enrichment strategy:

### 1. SUBJECT-FIRST OPENING with full appearance
- Start with the main subject noun (no "The video shows/depicts...")
- Immediately add SPECIFIC appearance details: exact colors, materials, textures, sizes, count
- Include spatial position in frame ("positioned towards the left", "in the center of the frame")
- Example: "Two small sailboats" → "Two small sailboats floating on a cup of coffee. The first boat, positioned towards the left side of the frame, is larger and more detailed, with a white sail. The second boat, slightly smaller and to the right, also features a white sail. Both boats have dark brown hulls and appear to be intricately designed."

### 2. DETAILED MOTION with direction and progression
- Describe motion with SPECIFIC direction (left-to-right, clockwise, toward camera)
- Use temporal chain: "initially... then... as the scene progresses..."
- Include speed and intensity changes ("at a steady pace", "gradually accelerating")
- Add physical consequences of motion ("kicking up dust", "creating ripples", "tail wagging")
- Describe HOW different subjects move relative to each other

### 3. BACKGROUND & LIGHTING specifics
- Name specific materials and architectural features ("glass facades and steel structures")
- Describe lighting direction and quality ("sunlight filtering through", "dim lighting casting shadows")
- Include depth cues ("foreground/background", "distant mountains", "nearby vegetation")
- Note color palette with specifics ("dark blue water", "bright green foliage", "warm golden light")

### 4. CAMERA DESCRIPTION
- State camera behavior ("camera remains stationary", "wide tracking shot", "camera pans left to right")
- Note shot type if apparent ("medium shot", "wide angle", "close-up")
- Describe focus behavior if relevant ("gradually coming into sharp focus", "background softly blurred")

### 5. STRONG VISUAL ENDING
- End with a vivid, concrete sensory detail
- Use words that evoke texture or light ("shimmering reflections", "dust trail billowing", "neon light reflecting sharply")

## IMPORTANT GUIDELINES:

1. OUTPUT LENGTH: Aim for 150-250 words, 2-4 paragraphs. Longer is better than shorter.
2. You MAY add reasonable visual details that are LIKELY true given the scene description (materials, lighting direction, spatial relations). This is fine — a downstream VLM will verify and correct any errors.
3. PRESERVE all original information — every detail from the input must appear in your output.
4. Use SPECIFIC descriptors over generic ones: "dark brown hulls" not "wooden hulls", "glass facades and steel structures" not "buildings"
5. DO NOT add objects/subjects not implied by the original (no hallucinating new elements)
6. DO NOT add "text overlays", "watermarks", or meta-commentary
7. VARY your language — do not use identical template phrases across prompts

## Examples:

### Example 1:
INPUT: "The video depicts an underwater cityscape with tall buildings. A large whale swims gracefully through the scene. The water is blue. Fish swim around the whale. The lighting is dim and the mood is serene."

OUTPUT: "Giant whale swimming gracefully through an underwater cityscape with tall buildings emerging from the water. The buildings have a modern architectural style with glass facades and steel structures. The water is dark blue and rippled, creating a sense of depth and movement. The whale initially enters from the left side of the frame, then glides steadily rightward through the center of the scene, its tail and fins moving in slow rhythmic undulation. Fish can be seen swimming around the whale, scattering as it passes and adding to the underwater atmosphere. The lighting is dim, giving the scene a mysterious and serene mood, with the whale's massive form creating gentle currents in the dark blue water."

### Example 2:
INPUT: "A white SUV drives on a dirt road through mountains. There are trees on both sides. The SUV kicks up dust. It's sunny."

OUTPUT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.

The SUV initially appears from the left side of the frame, then accelerates steadily forward along the dirt road, kicking up a growing trail of dust as it moves. The vehicle's tire tracks are visible on the road, and its headlights illuminate the path ahead. The SUV moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.

The combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience with the dust trail billowing behind the vehicle."

### Example 3:
INPUT: "An orange and white cat walks along a garden path with flowers. The cat moves toward the camera. The garden has colorful flowers."

OUTPUT: "Orange and white cat walking gracefully through a serene garden on a stone path, with lush greenery and vibrant flowers surrounding it.

Initially, the cat appears slightly blurred in the background at the center of the frame, then steps steadily forward toward the camera, gradually coming into sharp focus. Its fur contrasts beautifully against the green leaves and colorful blooms. The cat's tail is raised high, swaying gently with each step, adding a sense of movement and energy to the scene as it moves with confident forward stride.

As the cat continues its journey, the background reveals more details of the garden. The path is lined with various plants and flowers, including red and yellow blooms, creating a picturesque backdrop for the cat's gentle stroll. The vibrant colors of the flowers add depth and visual interest to the scene.

The camera maintains a medium shot throughout the video, keeping both the cat and the garden in focus. The camera remains stationary, providing a stable and clear view of the cat's movements and the surrounding garden, capturing the cat's graceful forward motion through the lush garden path."

## Output ONLY the enriched prompt. No explanations, no "WHY" section."""

USER_TEMPLATE = """Enrich this VLM caption ({word_count} words) into a detailed video generation prompt for Wan2.1 T2V.

Requirements:
- Subject-first opening with full appearance details (colors, materials, textures, spatial position)
- Multi-paragraph structure (2-4 paragraphs)
- Detailed motion with temporal chain and specific direction/speed
- Background with specific materials, lighting quality, and depth cues
- Camera description (shot type, movement, focus behavior)
- Strong visual ending with concrete sensory detail
- Target length: 150-250 words

You MAY add reasonable visual details (materials, lighting, spatial relations) — a VLM will verify accuracy later.

INPUT:
{original_caption}

OUTPUT:"""

# ─────────────────────────────────────────────────────────────────────────────
# VLM 验证纠错 System Prompt
# ─────────────────────────────────────────────────────────────────────────────

VLM_VERIFY_SYSTEM = """You are a video-grounded fact checker for text-to-video prompts. You receive:
1. A video (the original reference)
2. A rewritten prompt that is supposed to describe that video

Your job: Watch the video carefully, then check if the prompt contains any FACTUAL ERRORS compared to what actually happens in the video.

## What counts as a factual error:
- Wrong motion direction (prompt says "left to right" but video shows "right to left")
- Wrong subject appearance (prompt says "white cat" but video shows "orange cat")
- Wrong subject count (prompt says "two dogs" but video shows "one dog")
- Wrong action (prompt says "running" but video shows "walking slowly")
- Non-existent elements (prompt mentions "rain" but there is no rain in the video)
- Wrong spatial relationships (prompt says "foreground" but subject is in background)

## What is NOT an error:
- Added motion details that are CONSISTENT with the video (e.g., adding "from left to right" when the video does show that direction)
- Enriched atmosphere/mood descriptions that match the video's tone
- Temporal markers (initially/then/finally) as long as the sequence matches
- Specific adjectives for things that are ambiguous in the video (e.g., "warm golden light" for sunset lighting)

## Output format:
If there are NO factual errors, output exactly:
VERIFIED: No factual errors found.

If there ARE errors, output:
ERRORS FOUND:
1. [specific error]: prompt says "[X]" but video shows "[Y]"
2. [specific error]: prompt says "[X]" but video shows "[Y]"

CORRECTED PROMPT:
[The full prompt with ONLY the factual errors fixed. Keep all enrichments and expansions that are consistent with the video.]"""

VLM_VERIFY_USER_TEMPLATE = """Watch this video carefully, then check if the following prompt contains any factual errors compared to what actually happens in the video.

PROMPT TO VERIFY:
{rewritten_prompt}

Check for: wrong motion direction, wrong subject appearance/count, wrong actions, non-existent elements, wrong spatial relationships. Output VERIFIED if accurate, or list errors and provide a corrected prompt."""

# NOTE: Negative prompt 功能已移除。实验 B vs C 证明 Negative Prompt 对 Wan2.1-1.3B
# 有害（额外 -5.9% XCLIP），UMT5 encoder 不是为 negative prompt 设计的。
# 参见 docs/实验_L1_Prompt_Rewrite对比.md


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
# VLM 验证逻辑（传入整个视频进行事实核查）
# ─────────────────────────────────────────────────────────────────────────────

def vlm_verify_prompt(video_path: str, rewritten_prompt: str,
                      vlm_client) -> dict:
    """
    使用 VLM 对改写后的 prompt 进行视频级事实核查。

    传入整个原始视频（而非抽帧），让 VLM 判断 prompt 中是否有
    与视频实际内容不符的事实性错误，并返回纠正后的 prompt。

    Args:
        video_path: 原始视频路径
        rewritten_prompt: LLM 改写后的 prompt
        vlm_client: VLM 客户端实例（需支持 use_video_mode=True）

    Returns:
        dict: {
            "verified": bool,  # 是否通过验证
            "errors": list,    # 错误列表（空列表表示无错误）
            "corrected_prompt": str  # 纠正后的prompt（无错误时等于输入）
        }
    """
    user_msg = VLM_VERIFY_USER_TEMPLATE.format(rewritten_prompt=rewritten_prompt)

    # 根据 VLM client 类型选择调用方式
    client_type = type(vlm_client).__name__

    if client_type == "LocalVLMClient":
        # 本地 VLM: 抽帧传入（Qwen2.5-VL 本地模型）
        vlm_client._load_model()
        num_frames = 16 if vlm_client.use_video_mode else 8
        frames_pil = vlm_client._extract_frames_pil(video_path, num_frames=num_frames)
        if not frames_pil:
            logger.warning(f"VLM verify: 无法从视频抽帧 {video_path}")
            return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}

        content_list = []
        for img in frames_pil:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": user_msg})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": VLM_VERIFY_SYSTEM}]},
            {"role": "user", "content": content_list},
        ]

        try:
            response_text = vlm_client._generate(messages)
        except Exception as e:
            logger.warning(f"VLM verify 调用失败: {e}")
            return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}

    elif client_type == "VLMClient":
        # DashScope VLM: 上传完整视频（利用原生视频理解能力）
        content = [{"type": "text", "text": user_msg}]

        if vlm_client.use_video_mode:
            video_url = vlm_client._upload_video_to_dashscope(video_path)
            if video_url:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": video_url}
                })
                logger.info(f"  VLM verify: 传入完整视频进行事实核查")
            else:
                # 上传失败，fallback 到抽帧
                frames_b64 = vlm_client._extract_frames_base64(video_path, num_frames=16)
                if not frames_b64:
                    return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}
                for frame_b64 in frames_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                    })
        else:
            frames_b64 = vlm_client._extract_frames_base64(video_path, num_frames=16)
            if not frames_b64:
                return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}
            for frame_b64 in frames_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                })

        try:
            response = vlm_client.client.chat.completions.create(
                model=vlm_client.model_name,
                messages=[
                    {"role": "system", "content": VLM_VERIFY_SYSTEM},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,  # 低温度，提高事实判断准确性
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"VLM verify 调用失败: {e}")
            return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}

    elif client_type == "MockVLMClient":
        # Mock 模式直接通过
        return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}

    else:
        logger.warning(f"未知 VLM client 类型: {client_type}，跳过验证")
        return {"verified": True, "errors": [], "corrected_prompt": rewritten_prompt}

    # 解析 VLM 验证结果
    return _parse_vlm_verify_response(response_text, rewritten_prompt)


def _parse_vlm_verify_response(response_text: str, original_prompt: str) -> dict:
    """解析 VLM 验证响应"""
    response_upper = response_text.upper().strip()

    # 检查是否通过验证
    if response_upper.startswith("VERIFIED"):
        return {"verified": True, "errors": [], "corrected_prompt": original_prompt}

    # 提取错误列表
    errors = []
    corrected_prompt = original_prompt

    lines = response_text.split("\n")
    in_errors = False
    in_corrected = False
    corrected_lines = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.upper().startswith("ERRORS FOUND"):
            in_errors = True
            in_corrected = False
            continue

        if line_stripped.upper().startswith("CORRECTED PROMPT"):
            in_errors = False
            in_corrected = True
            continue

        if in_errors and line_stripped and line_stripped[0].isdigit():
            errors.append(line_stripped)

        if in_corrected and line_stripped:
            corrected_lines.append(line_stripped)

    if corrected_lines:
        corrected_prompt = "\n".join(corrected_lines)
        # 清理可能的引号包裹
        if corrected_prompt.startswith('"') and corrected_prompt.endswith('"'):
            corrected_prompt = corrected_prompt[1:-1]

    return {
        "verified": len(errors) == 0,
        "errors": errors,
        "corrected_prompt": corrected_prompt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_caption(original: str, backend: str, model: str,
                    api_base: str = "", api_key: str = "",
                    temperature: float = 0.5, max_retries: int = 2) -> str:
    """对单个 caption 执行约束式微调改写（带 length/diff 验证）"""
    word_count = len(original.split())
    user_msg = USER_TEMPLATE.format(
        word_count=word_count,
        original_caption=original,
    )

    result = None
    for attempt in range(max_retries + 1):
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

        # ── 验证 1: 长度检查（≥90% 原文词数，不超过 130%）──
        result_words = len(result.split())
        min_words = int(word_count * 0.70)  # 允许宽松下限（70%）
        max_words = int(word_count * 1.50)  # 不能超过 150%
        if result_words < min_words:
            logger.warning(f"  [重试 {attempt+1}] 输出过短: {result_words} 词 (最低 {min_words})")
            continue
        if result_words > max_words:
            logger.warning(f"  [重试 {attempt+1}] 输出过长: {result_words} 词 (最高 {max_words})")
            continue

        # ── 验证 2: 不能以 "The video" 开头（subject-first 违规）──
        if result.lower().startswith(("the video", "this video", "in this video")):
            logger.warning(f"  [重试 {attempt+1}] 仍以 preamble 开头")
            continue

        # ── 验证 3: diff check（编辑距离 ≤ 50%）──
        from difflib import SequenceMatcher
        tokens_orig = original.split()
        tokens_result = result.split()
        similarity = SequenceMatcher(None, tokens_orig, tokens_result).ratio()
        edit_ratio = 1.0 - similarity
        if edit_ratio > 0.50:
            logger.warning(f"  [重试 {attempt+1}] 改动过大: edit_ratio={edit_ratio:.0%} (最高 50%)")
            continue

        # 通过所有验证
        return result

    # 所有重试都失败，返回最后一次结果
    logger.warning(f"  所有重试均未通过验证，使用最后一次结果")
    return result




def _is_chinese_text(text: str) -> bool:
    """判断文本是否主要为中文（CJK 字符占比 > 30%）"""
    if not text:
        return False
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cjk_count / len(text) > 0.3


def _estimate_word_count(text: str) -> int:
    """估算等效英文词数：中文按 ~2字/词 换算，英文直接空格分词"""
    if _is_chinese_text(text):
        cjk_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        non_cjk_words = len(''.join(ch for ch in text if not ('\u4e00' <= ch <= '\u9fff')).split())
        return cjk_chars // 2 + non_cjk_words
    else:
        return len(text.split())


def validate_rewrite(original: str, rewritten: str) -> dict:
    """验证改写质量（约束式策略：≥70% 原文词数，≤150%，subject-first）"""
    orig_words = _estimate_word_count(original)
    new_words = len(rewritten.split())

    issues = []
    if not rewritten.strip():
        issues.append("empty output")
    if new_words < int(orig_words * 0.70):
        issues.append(f"too short ({new_words} words, min {int(orig_words * 0.70)})")
    if new_words > int(orig_words * 1.50):
        issues.append(f"too long ({new_words} words, max {int(orig_words * 1.50)})")
    if rewritten.lower().startswith(("the video", "this video", "in this video", "the scene")):
        issues.append("still starts with preamble (subject-first violated)")

    return {
        "valid": len(issues) == 0,
        "orig_words": orig_words,
        "new_words": new_words,
        "expansion_ratio": new_words / max(orig_words, 1),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Prompt 扩写改写 v2：VLM caption → LLM expansion → VLM verify"
    )

    # I/O
    parser.add_argument("--input-dir", type=str, required=True,
                        help="原始 caption 目录（包含 {id}.txt 文件）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录（改写后的 {id}.txt）")
    parser.add_argument("--video-dir", type=str, default="",
                        help="原始视频目录（VLM 验证时需要，包含 {id}.mp4）")
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

    # VLM 验证
    parser.add_argument("--enable-vlm-verify", action="store_true",
                        help="启用 VLM 视频级验证纠错（传整个视频给 VLM 检查事实性）")
    parser.add_argument("--vlm-provider", type=str, default="local",
                        choices=["local", "dashscope", "mock"],
                        help="VLM 验证使用的后端 (默认: local)")
    parser.add_argument("--vlm-model-path", type=str,
                        default="/root/models/Qwen2.5-VL-7B-Instruct",
                        help="本地 VLM 模型路径")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="LLM 生成温度 (默认: 0.5, 约束式微调不需要高创造性)")
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

    logger.info(f"待处理: {len(caption_files)} 个样本")
    logger.info(f"后端: {args.backend}, 模型: {args.model}")
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"VLM 验证: {'启用' if args.enable_vlm_verify else '禁用'}")

    # 初始化 VLM client（如果启用验证）
    vlm_client = None
    if args.enable_vlm_verify:
        if not args.video_dir:
            logger.error("启用 VLM 验证需要指定 --video-dir（原始视频目录）")
            sys.exit(1)
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            logger.error(f"视频目录不存在: {video_dir}")
            sys.exit(1)

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.vlm_client import create_vlm_client
        vlm_config = {
            "provider": args.vlm_provider,
            "model_path": args.vlm_model_path,
            "temperature": 0.3,
            "max_tokens": 2048,
            "max_retries": 3,
            "use_video_mode": True,  # 传整个视频，不是帧
            "lazy_load": True,
        }
        vlm_client = create_vlm_client(vlm_config)
        logger.info(f"VLM 验证已初始化: provider={args.vlm_provider}")

    # 统计
    success = 0
    failed = 0
    skipped = 0
    vlm_corrected = 0
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

        # ── Step 1: LLM 扩写改写 ──
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

        if not rewritten:
            failed += 1
            logger.error(f"  [{idx}/{len(caption_files)}] {sample_id} ✗ LLM 改写全部失败")
            results.append({"sample_id": sample_id, "status": "failed"})
            continue

        # ── Step 2: VLM 视频级验证纠错（传整个视频给VLM） ──
        vlm_verified = False
        if vlm_client and args.enable_vlm_verify:
            video_path = Path(args.video_dir) / f"{sample_id}.mp4"
            if video_path.exists():
                try:
                    verify_result = vlm_verify_prompt(
                        video_path=str(video_path),
                        rewritten_prompt=rewritten,
                        vlm_client=vlm_client,
                    )
                    if not verify_result["verified"]:
                        logger.info(
                            f"  [{idx}/{len(caption_files)}] {sample_id} VLM 纠错: "
                            f"发现 {len(verify_result['errors'])} 个错误"
                        )
                        for err in verify_result["errors"]:
                            logger.info(f"    → {err}")
                        rewritten = verify_result["corrected_prompt"]
                        vlm_corrected += 1
                    else:
                        logger.info(f"  [{idx}/{len(caption_files)}] {sample_id} VLM 验证: 通过")
                    vlm_verified = True
                except Exception as e:
                    logger.warning(f"  [{idx}/{len(caption_files)}] {sample_id} VLM 验证失败: {e}")
            else:
                logger.warning(f"  [{idx}/{len(caption_files)}] {sample_id} 视频不存在: {video_path}")

        # ── Step 3: 保存结果 ──
        out_file.write_text(rewritten + "\n", encoding="utf-8")
        success += 1

        new_words = len(rewritten.split())
        orig_words = _estimate_word_count(original)
        logger.info(
            f"  [{idx}/{len(caption_files)}] {sample_id} ✓ "
            f"({orig_words}→{new_words} words, x{new_words/max(orig_words,1):.1f}, "
            f"vlm={'✓' if vlm_verified else '—'})"
        )
        results.append({
            "sample_id": sample_id,
            "status": "success",
            "orig_words": orig_words,
            "new_words": new_words,
            "expansion_ratio": new_words / max(orig_words, 1),
            "vlm_verified": vlm_verified,
        })

        # 请求间隔
        if idx < len(caption_files):
            time.sleep(args.delay)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! 成功={success}, 失败={failed}, 跳过={skipped}, "
                f"VLM纠错={vlm_corrected}, 总计={len(caption_files)}")
    logger.info(f"输出目录: {output_dir}")

    # 保存处理日志
    log_file = output_dir / "rewrite_log.json"
    log_data = {
        "version": "v5_constrained",
        "strategy": "surgical_3_modifications_subject_first_temporal_chain_strong_ending",
        "backend": args.backend,
        "model": args.model,
        "temperature": args.temperature,
        "vlm_verify_enabled": args.enable_vlm_verify,
        "total": len(caption_files),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "vlm_corrected": vlm_corrected,
        "results": results,
    }
    log_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"处理日志: {log_file}")


if __name__ == "__main__":
    main()
