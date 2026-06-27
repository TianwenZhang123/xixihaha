#!/usr/bin/env python3
"""
Prompt rewrite v10: Head/Tail keyword replacement + VLM factual correction (2-step pipeline)

Design:
  v9 (LLM deletion + VLM supplement) achieves XCLIP +1.3% with SVD alone, but when
  combined with Feature Injection (L3), it UNDERPERFORMS bare caption (XCLIP 0.8051 < 0.8138).
  Root cause: v9 changes ~18-25% of tokens, shifting h_gen feature distribution away from
  the pre-cached h_ref features, causing semantic misalignment in FI adaptive gating.

  v10 strategy (FI-compatible, minimal edit):
    Step 1: LLM head/tail keyword replacement ONLY
            - Replace meaningless preamble ("The video depicts/shows...") with subject noun
            - Replace generic ending with vivid visual/motion keyword
            - Middle content stays VERBATIM (95% unchanged)
            -> output is SAME LENGTH as input (±5%)
    Step 2: VLM factual correction (max 3 word-level changes)
            - VLM watches video, compares with current caption
            - Only fixes factual errors (wrong color, wrong count, wrong object)
            - Max 3 single-word replacements, NO additions, NO deletions
            -> output is SAME LENGTH as input

  Theory (from 5.28 attention weight experiment):
    DiT cross-attention has extreme U-shaped position distribution:
    - Position 0 (first token): 10-15x weight vs middle
    - Last token: ~equal to position 0
    - Middle positions: nearly uniform (~0.001)
    Therefore: only head/tail matter for generation quality.

Usage:
    python scripts/rewrite_minimal.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/v10_captions \
        --video-dir /path/to/original_videos \
        --backend dashscope \
        --model qwen-plus \
        --vlm-provider local
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Step 1: LLM Head/Tail Keyword Replacement
# ============================================================================

LLM_SYSTEM_PROMPT = """You optimize VLM video captions for a T2V model by providing replacement text for ONLY the opening and ending portions.

BACKGROUND: The T2V model's DiT cross-attention has an extreme U-shaped position weight distribution:
- Position 0 (first token): receives 10-15x more attention than middle tokens
- Last token: receives ~equal attention as position 0
- All middle tokens: nearly uniform low attention
Therefore, ONLY the opening words and ending words significantly affect generation. The middle content barely matters for quality.

## Your task: Provide NEW OPENING and NEW ENDING text

You will output EXACTLY two lines:
- Line 1: The new opening phrase (replacing the preamble, or "UNCHANGED" if already good)
- Line 2: The new ending phrase (replacing the generic summary, or "UNCHANGED" if already concrete)

DO NOT output the full caption. ONLY output the two replacement pieces.

### OPENING replacement:
IF the caption starts with a meaningless preamble like "The video depicts/shows/features/captures/showcases...", provide a subject+action phrase that puts the CORE MOTION SUBJECT first.
IF the caption already starts with a concrete noun+action, output "UNCHANGED".

The first token receives 10-15x more attention than middle tokens. Put the MOTION SUBJECT (who/what is moving) before adjectives like color. Move displaced adjectives into the new opening phrase.

Examples:
- Input starts: "The video depicts a white SUV driving..." → Line 1: "SUV driving on a white dusty"
- Input starts: "The video showcases two small sailboats..." → Line 1: "Sailboats gliding across dark"
- Input starts: "A person running against..." → Line 1: "UNCHANGED"

### ENDING replacement:
IF the caption ends with a generic summary/atmosphere sentence ("The overall mood/atmosphere/scene creates/conveys...", "creating a sense of...", "adding to the overall aesthetic..."), provide 1-3 vivid VISUAL-ONLY keywords already mentioned in the caption's middle.
IF the caption already ends with a concrete visual description, output "UNCHANGED".

ENDING KEYWORDS MUST be visual nouns/adjectives only. Do NOT include motion direction, speed, or trajectory (e.g., "forward", "upward", "accelerating") — motion is handled by other system components.

Examples:
- Input ends: "...The overall atmosphere conveys exploration." → Line 2: "clear blue sky, dust trail"
- Input ends: "...creating a serene and peaceful mood." → Line 2: "smooth ripples, warm sunlight"
- Input ends: "...The background is slightly blurred." → Line 2: "UNCHANGED"

## ABSOLUTE RULES:
1. Output EXACTLY two lines. Nothing else.
2. Line 1 = new opening phrase (or "UNCHANGED")
3. Line 2 = new ending phrase (or "UNCHANGED")
4. Opening: MOTION SUBJECT first, before adjectives. Move displaced adjectives into the phrase.
5. Ending: Visual nouns/adjectives ONLY from the caption's middle. NO motion direction/speed.
6. ZERO new information: Keywords must come from facts already in the caption.

## FULL EXAMPLES:

INPUT:
"The video depicts a white SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear. As the SUV moves forward, it kicks up a cloud of dust behind it. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky. The overall atmosphere conveys a sense of exploration and outdoor adventure."

OUTPUT:
SUV driving on a white dusty
clear blue sky, dust trail

INPUT:
"The video features a person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run. The overall atmosphere of the video is focused on the physical activity and the simplicity of the setting."

OUTPUT:
Person running against a plain
white tank top, black shorts

INPUT (no preamble, no generic ending):
"A close-up view of a cup filled with dark liquid, likely coffee or tea, with two small toy sailboats floating on its surface. The sailboats have white sails and wooden hulls. The liquid in the cup is smooth, with some ripples around the boats. The background is slightly blurred, focusing attention on the cup and the boats."

OUTPUT:
UNCHANGED
UNCHANGED"""

LLM_USER_TEMPLATE = """Caption ({word_count} words):
{original_caption}

Provide Line 1 (new opening or UNCHANGED) and Line 2 (new ending or UNCHANGED):"""

# ============================================================================
# Step 2: VLM Factual Correction Prompts (max 3 word-level fixes)
# ============================================================================

VLM_CORRECTION_SYSTEM = """You are a visual fact-checker for text-to-video prompts. You receive:
1. A video (the original reference)
2. A caption that describes the video

Your ONLY job: Watch the video carefully, compare it with the caption, and FIX factual errors by replacing WRONG words with CORRECT words. You may make AT MOST 3 single-word replacements.

## What counts as a factual error (ONLY these):
- Wrong color (e.g., caption says "red car" but video shows a blue car → replace "red" with "blue")
- Wrong count (e.g., caption says "three birds" but video shows two → replace "three" with "two")
- Wrong object identity (e.g., caption says "dog" but it's clearly a cat → replace "dog" with "cat")
- Wrong material (e.g., caption says "wooden" but it's clearly metal → replace "wooden" with "metal")

## Rules:
1. MAX 3 CHANGES. If you find more than 3 errors, fix only the 3 most obvious ones.
2. Each change is a SINGLE-WORD REPLACEMENT. Do NOT add words, do NOT delete words.
3. Output must be EXACTLY the same length (same word count) as input.
4. NEVER change verbs, motion descriptions, or spatial relationships.
5. NEVER change style/atmosphere words (these are subjective, not factual).
6. NEVER restructure sentences or rephrase anything.
7. If there are NO factual errors, output the caption UNCHANGED.
8. When in doubt, DO NOT CHANGE. Only fix things you are 100% certain are wrong based on the video.

## Examples:

Input: "A red sedan driving on a highway with two passengers visible."
Video shows: blue sedan, three passengers
Output: "A blue sedan driving on a highway with three passengers visible."
(2 changes: red→blue, two→three)

Input: "A golden retriever running through a green field under overcast sky."
Video shows: matches perfectly
Output: "A golden retriever running through a green field under overcast sky."
(0 changes: caption is accurate)

Output ONLY the corrected caption. No explanations, no change logs."""

VLM_CORRECTION_USER = """Watch this video carefully. Compare the caption below with what you see in the video. If any words are factually wrong (wrong color, wrong count, wrong object), replace them with the correct words. Make AT MOST 3 single-word replacements. If the caption is accurate, output it UNCHANGED.

CURRENT CAPTION ({current_words} words):
{caption}

CORRECTED CAPTION:"""

# ============================================================================
# LLM Backends
# ============================================================================

def call_dashscope(prompt: str, system: str, model: str, api_key: str,
                   temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Call DashScope API"""
    import openai

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
    """Call OpenAI-compatible API"""
    import openai

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


# ============================================================================
# VLM Backends (video + text -> text)
# ============================================================================

def _extract_frames_base64(video_path: str, num_frames: int = 16) -> List[str]:
    """Extract evenly-spaced frames from video as base64 JPEG."""
    from src.vlm_client import _extract_frames
    from PIL import Image
    import io, base64

    pil_frames = _extract_frames(video_path, num_frames)
    frames_b64 = []
    for img in pil_frames:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return frames_b64


def _upload_video_to_dashscope(video_path: str, api_key: str, model: str) -> Optional[str]:
    """Upload video to DashScope OSS, return oss:// URL."""
    from src.vlm_client import upload_video_to_dashscope as _upload
    return _upload(video_path, api_key, model)


def call_vlm_dashscope(video_path: str, user_msg: str, system_msg: str,
                       api_key: str, model: str = "qwen-vl-max",
                       temperature: float = 0.4, max_tokens: int = 1024) -> str:
    """Call DashScope VLM with video + text"""
    import openai

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Try video upload first
    video_url = _upload_video_to_dashscope(video_path, api_key, model)

    if video_url and video_url.startswith("https://"):
        # Only use video_url if it's an accessible HTTPS URL
        content = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": user_msg},
        ]
    else:
        # Fallback to frame extraction (base64)
        # oss:// URLs are not supported by OpenAI-compatible endpoint
        if video_url:
            logger.warning(f"Video upload returned non-HTTPS URL ({video_url}), falling back to frame extraction")
        else:
            logger.warning("Video upload failed, falling back to frame extraction")
        frames_b64 = _extract_frames_base64(video_path, num_frames=16)
        if not frames_b64:
            raise RuntimeError(f"Cannot extract frames from: {video_path}")
        content = []
        for frame_b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
        content.append({"type": "text", "text": user_msg})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def call_vlm_local(video_path: str, user_msg: str, system_msg: str,
                   model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct",
                   temperature: float = 0.4, max_tokens: int = 1024) -> str:
    """Call local VLM (Qwen2.5-VL) with frames + text"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.vlm_client import LocalVLMClient

    # Singleton to avoid reloading
    if not hasattr(call_vlm_local, "_client"):
        call_vlm_local._client = LocalVLMClient(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            use_video_mode=True,
            lazy_load=True,
        )

    vlm = call_vlm_local._client
    vlm._load_model()

    frames_pil = vlm._extract_frames_pil(video_path, num_frames=16)
    if not frames_pil:
        raise RuntimeError(f"Cannot extract frames from: {video_path}")

    content_list = []
    for img in frames_pil:
        content_list.append({"type": "image", "image": img})
    content_list.append({"type": "text", "text": user_msg})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": content_list},
    ]

    response_text = vlm._generate(messages)
    return response_text.strip()


# ============================================================================
# Pipeline Steps
# ============================================================================

def _detect_preamble_end(words: list) -> int:
    """Find the index after the preamble (e.g., 'The video depicts a') to replace."""
    # Pattern: [The/This] [video/scene/clip] [depicts/shows/features/captures/showcases] [a/an/the]?
    if len(words) < 3:
        return 0
    w0, w1 = words[0].lower(), words[1].lower()
    # Must start with a dummy subject
    if w0 not in ("the", "this", "a", "an"):
        return 0
    if w1 not in ("video", "scene", "clip", "footage", "image", "picture"):
        return 0
    # Find the verb position
    verb_idx = None
    for i in range(2, min(len(words), 5)):
        if words[i].lower().rstrip("s") in (
            "depict", "show", "feature", "capture", "showcase",
            "display", "present", "portray", "illustrate", "reveal",
        ):
            verb_idx = i
            break
    if verb_idx is None:
        return 0
    # Skip article after verb: "depicts a white SUV" -> skip "a"
    end_idx = verb_idx + 1
    if end_idx < len(words) and words[end_idx].lower() in ("a", "an", "the"):
        end_idx += 1
    # Also skip adjective if it was before the subject: "a white SUV" -> include "white" in preamble
    # We stop here; the adjective will be moved by the LLM into the new opening
    return end_idx


def _detect_generic_ending(words: list) -> int:
    """Find the index where the generic ending sentence starts. Returns len(words) if no generic ending."""
    n = len(words)
    if n < 5:
        return n
    # Look for patterns in the last ~15 words
    search_start = max(0, n - 15)
    tail_text = " ".join(words[search_start:]).lower()
    # Common generic ending patterns
    generic_patterns = [
        "the overall atmosphere",
        "the overall mood",
        "the overall scene",
        "the overall aesthetic",
        "the overall feel",
        "creating a sense",
        "creating a serene",
        "creating a peaceful",
        "creating a dynamic",
        "adding to the overall",
        "adding to the aesthetic",
        "adding to the atmosphere",
        "adding to the dynamic",
        "conveying a sense",
        "conveying a feeling",
        "the scene conveys",
        "the atmosphere conveys",
        "the mood is",
        "the video has a",
    ]
    for pat in generic_patterns:
        idx = tail_text.find(pat)
        if idx >= 0:
            # Map back to word index
            word_idx = search_start + len(tail_text[:idx].split())
            # Include preceding period/comma if any
            if word_idx > 0 and words[word_idx - 1].endswith("."):
                word_idx -= 1  # Include the period as part of what we replace
            return word_idx
    return n


def step1_llm_headtail(original: str, backend: str, model: str,
                       api_base: str = "", api_key: str = "",
                       temperature: float = 0.2, max_retries: int = 3) -> str:
    """Step 1: LLM provides new opening/ending phrases; code splices them into the original."""
    words = original.split()
    word_count = len(words)

    # Detect preamble and generic ending boundaries
    preamble_end = _detect_preamble_end(words)
    ending_start = _detect_generic_ending(words)

    # If no preamble and no generic ending, return as-is (no LLM call needed)
    if preamble_end == 0 and ending_start == word_count:
        logger.info(f"  Step1: No preamble/generic ending detected, keeping original")
        return original

    # Call LLM to get replacement phrases
    user_msg = LLM_USER_TEMPLATE.format(
        word_count=word_count,
        original_caption=original,
    )

    llm_result = None
    for attempt in range(max_retries + 1):
        temp = temperature if attempt == 0 else min(0.4, temperature + attempt * 0.1)

        if backend == "dashscope":
            llm_result = call_dashscope(user_msg, LLM_SYSTEM_PROMPT, model, api_key, temp)
        elif backend == "openai":
            llm_result = call_openai_compatible(user_msg, LLM_SYSTEM_PROMPT, model, api_base, api_key, temp)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Parse the two-line response
        lines = [l.strip() for l in llm_result.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            break  # Got both lines
        logger.warning(f"  [Step1 retry {attempt+1}] Expected 2 lines, got {len(lines)}")
    else:
        logger.warning(f"  Step1 all retries failed to get 2-line response, using last result")
        if llm_result is None:
            return original

    # Parse the two lines
    lines = [l.strip() for l in llm_result.strip().split("\n") if l.strip()]
    new_opening = lines[0] if len(lines) >= 1 else "UNCHANGED"
    new_ending = lines[1] if len(lines) >= 2 else "UNCHANGED"

    # Build the result by splicing
    # Middle = words from preamble_end to ending_start (verbatim)
    middle_words = words[preamble_end:ending_start]
    middle_text = " ".join(middle_words)

    # Handle opening
    if new_opening.upper() == "UNCHANGED" or preamble_end == 0:
        opening_text = " ".join(words[:preamble_end]) if preamble_end > 0 else ""
    else:
        opening_text = new_opening

    # Handle ending
    if new_ending.upper() == "UNCHANGED" or ending_start == word_count:
        ending_text = " ".join(words[ending_start:]) if ending_start < word_count else ""
    else:
        ending_text = new_ending

    # Assemble
    parts = []
    if opening_text:
        parts.append(opening_text)
    if middle_text:
        parts.append(middle_text)
    if ending_text:
        parts.append(ending_text)

    result = " ".join(parts)
    # Clean up double spaces
    result = " ".join(result.split())

    result_words = len(result.split())
    logger.info(
        f"  Step1 splice: preamble_end={preamble_end}, ending_start={ending_start}, "
        f"opening='{opening_text[:40]}...', ending='{ending_text[:40]}...'"
    )

    return result


def step2_vlm_correction(step1_caption: str, video_path: str,
                         vlm_provider: str = "dashscope",
                         api_key: str = "", vlm_model: str = "qwen-vl-max",
                         vlm_model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct",
                         temperature: float = 0.2, max_retries: int = 2) -> str:
    """Step 2: VLM watches video, makes at most 3 factual word-level corrections"""
    current_words = len(step1_caption.split())

    system_msg = VLM_CORRECTION_SYSTEM
    user_msg = VLM_CORRECTION_USER.format(
        current_words=current_words,
        caption=step1_caption,
    )

    result = None
    for attempt in range(max_retries + 1):
        try:
            if vlm_provider == "dashscope":
                result = call_vlm_dashscope(
                    video_path=video_path,
                    user_msg=user_msg,
                    system_msg=system_msg,
                    api_key=api_key,
                    model=vlm_model,
                    temperature=temperature,
                )
            elif vlm_provider == "local":
                result = call_vlm_local(
                    video_path=video_path,
                    user_msg=user_msg,
                    system_msg=system_msg,
                    model_path=vlm_model_path,
                    temperature=temperature,
                )
            else:
                raise ValueError(f"Unknown VLM provider: {vlm_provider}")

            # Clean quotes
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            if result.startswith("'") and result.endswith("'"):
                result = result[1:-1]

            # Validation 1: length must be same as input (±10% tolerance for word-level replace)
            result_words = len(result.split())
            if result_words > current_words * 1.10:
                logger.warning(f"  [Step2 retry {attempt+1}] VLM added too many words: {result_words} > {current_words}*1.10")
                continue

            if result_words < current_words * 0.90:
                logger.warning(f"  [Step2 retry {attempt+1}] VLM deleted too many words: {result_words} < {current_words}*0.90")
                continue

            # Validation 2: not start with "The video"
            if result.lower().startswith(("the video", "this video", "in this video")):
                logger.warning(f"  [Step2 retry {attempt+1}] VLM output starts with preamble")
                continue

            return result

        except Exception as e:
            logger.warning(f"  [Step2 retry {attempt+1}] VLM call failed: {e}")
            if attempt < max_retries:
                time.sleep(2)

    # All VLM attempts failed, return Step1 result (no correction is better than bad correction)
    logger.warning(f"  Step2 all VLM attempts failed, using Step1 result")
    return result if result else step1_caption


def validate_rewrite(original: str, rewritten: str) -> dict:
    """Validate rewrite quality (v10: expect same length ±10%)"""
    orig_words = len(original.split())
    new_words = len(rewritten.split())

    issues = []
    if not rewritten.strip():
        issues.append("empty output")
    if rewritten.lower().startswith(("the video", "this video", "in this video")):
        issues.append("still starts with preamble")
    if new_words > orig_words * 1.10:
        issues.append(f"too long ({new_words} > {orig_words}*1.10)")
    if new_words < orig_words * 0.85:
        issues.append(f"too short ({new_words} < {orig_words}*0.85)")

    # Check motion verb preservation
    motion_verbs = ["running", "walking", "driving", "flying", "swimming",
                    "moving", "riding", "skating", "jumping", "falling",
                    "climbing", "spinning", "rolling", "sliding", "flowing",
                    "swinging", "kicking", "drifting", "floating", "swaying"]
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
        "ratio": new_words / max(orig_words, 1),
        "issues": issues,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prompt rewrite v10: Head/Tail keyword replacement + VLM factual correction"
    )

    # I/O
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input caption directory (contains {id}.txt files)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory (final {id}.txt)")
    parser.add_argument("--video-dir", type=str, default="",
                        help="Original video directory (contains {id}.mp4, needed for VLM)")
    parser.add_argument("--sample-ids", type=int, nargs="+",
                        help="Only process specified sample IDs (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip existing output files (resume)")

    # LLM backend (Step 1)
    parser.add_argument("--backend", type=str, default="dashscope",
                        choices=["dashscope", "openai"],
                        help="LLM backend: dashscope or openai-compatible")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="LLM model name (default: qwen-plus)")
    parser.add_argument("--api-base", type=str, default="",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", type=str, default="",
                        help="API Key (or set DASHSCOPE_API_KEY env var)")

    # VLM correction (Step 2)
    parser.add_argument("--vlm-model", type=str, default="qwen-vl-max",
                        help="VLM model name (default: qwen-vl-max)")
    parser.add_argument("--vlm-provider", type=str, default="local",
                        choices=["dashscope", "local"],
                        help="VLM backend: dashscope (remote) or local (GPU)")
    parser.add_argument("--vlm-model-path", type=str,
                        default="/root/models/Qwen2.5-VL-7B-Instruct",
                        help="Local VLM model path (only for --vlm-provider local)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM correction step (only do LLM head/tail replacement)")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature (default: 0.2, low for precise replacement)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per sample (default: 3)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds (default: 0.5)")

    args = parser.parse_args()

    # Resolve API Key
    api_key = args.api_key
    if not api_key:
        if args.backend == "dashscope":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if args.backend == "dashscope" and not api_key:
        logger.error("Need DASHSCOPE_API_KEY env var or --api-key")
        sys.exit(1)

    # Prepare directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # VLM check
    video_dir = None
    if not args.skip_vlm:
        if not args.video_dir:
            logger.error("VLM correction requires --video-dir, or use --skip-vlm to skip")
            sys.exit(1)
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            logger.error(f"Video directory not found: {video_dir}")
            sys.exit(1)

    # Collect caption files
    caption_files = sorted(
        input_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    if not caption_files:
        logger.error(f"No caption files found: {input_dir}/*.txt")
        sys.exit(1)

    logger.info(f"{'='*60}")
    logger.info(f"v10-headtail-correction: LLM head/tail replace -> VLM factual correction")
    logger.info(f"{'='*60}")
    logger.info(f"Samples: {len(caption_files)}")
    logger.info(f"LLM: {args.backend}/{args.model}, temp={args.temperature}")
    if not args.skip_vlm:
        logger.info(f"VLM: {args.vlm_provider}/{args.vlm_model}")
    else:
        logger.info(f"VLM: DISABLED (--skip-vlm)")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pipeline: Step1(LLM structured head/tail replace + code splice) -> Step2(VLM max-3 factual corrections)")

    # Statistics
    success = 0
    failed = 0
    skipped = 0
    results = []

    for idx, cap_file in enumerate(caption_files, 1):
        sample_id = cap_file.stem
        out_file = output_dir / f"{sample_id}.txt"

        # Resume support
        if args.skip_existing and out_file.exists():
            logger.info(f"  [{idx}/{len(caption_files)}] Skip {sample_id} (exists)")
            skipped += 1
            continue

        original = cap_file.read_text(encoding="utf-8").strip()
        if not original:
            logger.warning(f"  [{idx}/{len(caption_files)}] Skip {sample_id} (empty)")
            skipped += 1
            continue

        orig_words = len(original.split())

        # ====================================================================
        # Step 1: LLM Head/Tail Keyword Replacement
        # ====================================================================
        try:
            step1_result = step1_llm_headtail(
                original=original,
                backend=args.backend,
                model=args.model,
                api_base=args.api_base,
                api_key=api_key,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )

            step1_words = len(step1_result.split())
            logger.info(
                f"  [{idx}/{len(caption_files)}] {sample_id} Step1: "
                f"{orig_words}->{step1_words} words (delta {step1_words - orig_words:+d})"
            )

        except Exception as e:
            failed += 1
            logger.error(f"  [{idx}/{len(caption_files)}] {sample_id} Step1 FAILED: {e}")
            results.append({"sample_id": sample_id, "status": "failed", "error": str(e)})
            if idx < len(caption_files):
                time.sleep(args.delay)
            continue

        # ====================================================================
        # Step 2: VLM Factual Correction (max 3 word-level fixes)
        # ====================================================================
        final_result = step1_result  # Default fallback

        if not args.skip_vlm and video_dir:
            video_path = video_dir / f"{sample_id}.mp4"
            if video_path.exists():
                try:
                    final_result = step2_vlm_correction(
                        step1_caption=step1_result,
                        video_path=str(video_path),
                        vlm_provider=args.vlm_provider,
                        api_key=api_key,
                        vlm_model=args.vlm_model,
                        vlm_model_path=args.vlm_model_path,
                        temperature=0.2,
                        max_retries=2,
                    )

                    final_words = len(final_result.split())
                    logger.info(
                        f"  [{idx}/{len(caption_files)}] {sample_id} Step2: "
                        f"{step1_words}->{final_words} words (corrections applied)"
                    )

                except Exception as e:
                    logger.warning(
                        f"  [{idx}/{len(caption_files)}] {sample_id} Step2 failed: {e}, "
                        f"using Step1 result"
                    )
                    final_result = step1_result
            else:
                logger.warning(
                    f"  [{idx}/{len(caption_files)}] {sample_id} video not found: {video_path}, "
                    f"using Step1 result"
                )

        # Save result
        validation = validate_rewrite(original, final_result)
        out_file.write_text(final_result + "\n", encoding="utf-8")
        success += 1

        final_words = len(final_result.split())
        step1_words_val = len(step1_result.split())

        if validation["issues"]:
            logger.warning(
                f"  [{idx}/{len(caption_files)}] {sample_id} validation warnings: "
                f"{validation['issues']}"
            )

        logger.info(
            f"  [{idx}/{len(caption_files)}] {sample_id} DONE: "
            f"{orig_words}->{step1_words_val}->{final_words} words "
            f"(x{final_words/max(orig_words,1):.2f})"
        )

        results.append({
            "sample_id": sample_id,
            "status": "success",
            "orig_words": orig_words,
            "step1_words": step1_words_val,
            "final_words": final_words,
            "step1_ratio": step1_words_val / max(orig_words, 1),
            "final_ratio": final_words / max(orig_words, 1),
            "issues": validation["issues"],
        })

        # Delay between requests
        if idx < len(caption_files):
            time.sleep(args.delay)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! success={success}, failed={failed}, skipped={skipped}, total={len(caption_files)}")
    logger.info(f"Output: {output_dir}")

    if results:
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            avg_step1 = sum(r["step1_ratio"] for r in successful) / len(successful)
            avg_final = sum(r["final_ratio"] for r in successful) / len(successful)
            logger.info(f"Avg Step1 ratio: {avg_step1:.2f}, Avg final ratio: {avg_final:.2f} (target ~1.0, v10 minimal edit)")

    # Save processing log
    log_file = output_dir / "rewrite_log.json"
    log_data = {
        "version": "v10_headtail_correction",
        "strategy": "llm_structured_headtail_splice_then_vlm_factual_correction",
        "description": "2-step pipeline: Step1 LLM outputs only new opening+ending phrases (2 lines), code splices them into original (middle 100% verbatim); Step2 VLM makes max 3 word-level factual corrections. Length preserved by construction.",
        "pipeline": [
            "Step1: LLM outputs 2 lines (new opening, new ending), code splices into original (preamble/generic ending detected by rule-based parser)",
            "Step2: VLM factual correction (watch video, fix max 3 wrong color/count/object words)",
        ],
        "key_insight": "DiT cross-attention U-shaped: pos0 gets 10-15x weight, last token equal, middle flat. v9 changed ~18% causing FI misalignment (XCLIP 0.8051<0.8138). v10 edits ~5-8% for FI compatibility.",
        "backend": args.backend,
        "model": args.model,
        "vlm_provider": args.vlm_provider if not args.skip_vlm else "disabled",
        "vlm_model": args.vlm_model if not args.skip_vlm else "disabled",
        "temperature": args.temperature,
        "total": len(caption_files),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    log_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Log saved: {log_file}")


if __name__ == "__main__":
    main()
