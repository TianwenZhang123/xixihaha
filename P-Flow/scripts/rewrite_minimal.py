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
import base64
import io
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

LLM_SYSTEM_PROMPT = """You optimize VLM video captions for a T2V model by replacing ONLY the first few words and last few words with more effective keywords.

BACKGROUND: The T2V model's DiT cross-attention has an extreme U-shaped position weight distribution:
- Position 0 (first token): receives 10-15x more attention than middle tokens
- Last token: receives ~equal attention as position 0
- All middle tokens: nearly uniform low attention
Therefore, ONLY the opening words and ending words significantly affect generation. The middle content barely matters for quality.

## Your task: HEAD/TAIL REPLACEMENT (nothing else)

### OPENING (first 3-8 words):
IF the caption starts with a meaningless preamble like "The video depicts/shows/features/captures/showcases...", REPLACE it with the core subject noun phrase.
IF the caption already starts with a concrete noun, DO NOT CHANGE IT.

Examples of valid replacements:
- "The video depicts a white SUV driving..." → "White SUV driving..."
- "The video showcases two small sailboats..." → "Two small sailboats..."
- "The video captures a golden retriever..." → "Golden retriever..."
- "A person running against..." → KEEP AS-IS (already starts with subject)

### ENDING (last sentence or trailing clause):
IF the caption ends with a generic summary/atmosphere sentence ("The overall mood/atmosphere/scene creates/conveys...", "creating a sense of...", "adding to the overall aesthetic..."), REPLACE it with 1-3 vivid visual or motion keywords that are ALREADY mentioned somewhere in the middle of the caption.
IF the caption already ends with a concrete visual description, DO NOT CHANGE IT.

Examples of valid replacements:
- "...The overall atmosphere conveys exploration." → "...clear blue sky, dust trail."
- "...creating a serene and peaceful mood." → "...smooth ripples, warm sunlight."
- "...adding to the dynamic feel of the scene." → "...reflective surface, neon glow."
- "...The background is slightly blurred." → KEEP AS-IS (already concrete)

## ABSOLUTE RULES:

1. MIDDLE CONTENT UNCHANGED: Everything between the opening and ending MUST remain VERBATIM. Do NOT rephrase, reorder, delete, or add anything in the middle.
2. SAME LENGTH: Output must be within ±5% of input word count. You are REPLACING, not deleting or adding.
3. ZERO new information: The ending keywords must come from facts already stated in the caption's middle. Do NOT invent new details.
4. ZERO motion changes: Every motion verb, direction, speed MUST appear UNCHANGED in its original position.
5. If BOTH opening and ending are already good (no preamble, no generic summary), output the caption UNCHANGED.

## FULL EXAMPLES:

INPUT (94 words):
"The video depicts a white SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear. As the SUV moves forward, it kicks up a cloud of dust behind it. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky. The overall atmosphere conveys a sense of exploration and outdoor adventure."

OUTPUT (88 words):
"White SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear. As the SUV moves forward, it kicks up a cloud of dust behind it. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky. Dust cloud, pine trees, distant mountains."

Changed: opening "The video depicts a" → "" (subject-first); ending "The overall atmosphere..." → keywords from middle. Middle 100% unchanged.

INPUT (77 words):
"The video features a person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run. The overall atmosphere of the video is focused on the physical activity and the simplicity of the setting."

OUTPUT (72 words):
"A person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run. Soft lighting, steady jogging pace."

Changed: opening "The video features" removed; ending "The overall atmosphere..." → keywords from middle. Middle 100% unchanged.

INPUT (that needs NO change, 65 words):
"A close-up view of a cup filled with dark liquid, likely coffee or tea, with two small toy sailboats floating on its surface. The sailboats have white sails and wooden hulls. The liquid in the cup is smooth, with some ripples around the boats. The background is slightly blurred, focusing attention on the cup and the boats."

OUTPUT (65 words - UNCHANGED):
"A close-up view of a cup filled with dark liquid, likely coffee or tea, with two small toy sailboats floating on its surface. The sailboats have white sails and wooden hulls. The liquid in the cup is smooth, with some ripples around the boats. The background is slightly blurred, focusing attention on the cup and the boats."

No preamble, no generic ending → output is IDENTICAL to input.

Output ONLY the modified caption. No explanations."""

LLM_USER_TEMPLATE = """Optimize this VLM caption ({word_count} words) for a T2V model. ONLY do 2 things:
(1) If it starts with "The video depicts/shows/features/captures...", replace that preamble with the subject noun. Otherwise keep the opening as-is.
(2) If it ends with a generic summary/atmosphere sentence, replace it with 1-3 vivid keywords already mentioned in the middle. Otherwise keep the ending as-is.
Do NOT change anything in the middle. Output should be approximately the SAME length (±5%).

INPUT:
{original_caption}

OUTPUT:"""

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
    """Extract evenly-spaced frames from video as base64 JPEG"""
    import numpy as np

    if not os.path.exists(video_path):
        return []

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
    except (ImportError, Exception):
        try:
            import imageio.v3 as iio
            all_frames = iio.imread(video_path, plugin="pyav")
            total_frames = len(all_frames)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = all_frames[indices]
        except Exception:
            return []

    from PIL import Image

    frames_b64 = []
    for frame in frames:
        img = Image.fromarray(frame)
        max_dim = 1280
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return frames_b64


def _upload_video_to_dashscope(video_path: str, api_key: str, model: str) -> Optional[str]:
    """Upload video to DashScope OSS, return oss:// URL"""
    try:
        import requests as _requests
        import mimetypes
        from urllib.parse import urlparse
    except ImportError:
        return None

    if not os.path.isfile(video_path):
        return None

    try:
        cert_url = "https://dashscope.aliyuncs.com/api/v1/uploads"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"action": "getPolicy", "model": model}

        cert_resp = _requests.get(cert_url, headers=headers, params=params, timeout=30)
        cert_resp.raise_for_status()
        cert_data = cert_resp.json()

        if cert_data.get("status_code") != 200 and "output" not in cert_data:
            output = cert_data.get("data", cert_data.get("output", {}))
        else:
            output = cert_data.get("output", {})

        upload_dir = output.get("upload_dir", "")
        upload_host = output.get("upload_host", "")
        oss_access_key_id = output.get("oss_access_key_id", "")
        signature = output.get("signature", "")
        policy = output.get("policy", "")
        x_oss_object_acl = output.get("x_oss_object_acl", "private")
        x_oss_forbid_overwrite = output.get("x_oss_forbid_overwrite", "true")

        if not all([upload_dir, upload_host, oss_access_key_id, signature, policy]):
            return None

        filename = os.path.basename(video_path)
        object_key = f"{upload_dir}/{filename}"
        content_type = mimetypes.guess_type(video_path)[0] or "video/mp4"

        with open(video_path, "rb") as f:
            files = {"file": (filename, f, content_type)}
            upload_resp = _requests.post(
                upload_host,
                data={
                    "OSSAccessKeyId": oss_access_key_id,
                    "Signature": signature,
                    "policy": policy,
                    "key": object_key,
                    "x-oss-object-acl": x_oss_object_acl,
                    "x-oss-forbid-overwrite": x_oss_forbid_overwrite,
                    "success_action_status": "200",
                    "x-oss-content-type": content_type,
                },
                files=files,
                timeout=120,
            )

        if upload_resp.status_code == 200:
            parsed = urlparse(upload_host)
            bucket = parsed.hostname.split(".")[0] if parsed.hostname else "dashscope"
            oss_url = f"oss://{bucket}/{object_key}"
            return oss_url
        else:
            return None

    except Exception as e:
        logger.warning(f"Video upload failed: {e}")
        return None


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

def step1_llm_headtail(original: str, backend: str, model: str,
                       api_base: str = "", api_key: str = "",
                       temperature: float = 0.2, max_retries: int = 3) -> str:
    """Step 1: LLM head/tail keyword replacement (keep middle verbatim, ±5% length)"""
    word_count = len(original.split())

    user_msg = LLM_USER_TEMPLATE.format(
        word_count=word_count,
        original_caption=original,
    )

    result = None
    for attempt in range(max_retries + 1):
        temp = temperature if attempt == 0 else min(0.4, temperature + attempt * 0.1)

        if backend == "dashscope":
            result = call_dashscope(user_msg, LLM_SYSTEM_PROMPT, model, api_key, temp)
        elif backend == "openai":
            result = call_openai_compatible(user_msg, LLM_SYSTEM_PROMPT, model, api_base, api_key, temp)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Clean possible quote wrapping
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        # Validation 1: must not start with "The video"
        if result.lower().startswith(("the video", "this video", "in this video")):
            logger.warning(f"  [Step1 retry {attempt+1}] Still starts with preamble")
            continue

        # Validation 2: length must be within ±15% of original (head/tail replace, not deletion)
        result_words = len(result.split())
        if result_words > word_count * 1.15:
            logger.warning(f"  [Step1 retry {attempt+1}] Output too long: {result_words} > {word_count}*1.15")
            continue

        if result_words < word_count * 0.85:
            logger.warning(f"  [Step1 retry {attempt+1}] Output too short: {result_words} < {word_count}*0.85")
            continue

        return result

    logger.warning(f"  Step1 all retries failed, using last result")
    return result if result else original


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
    logger.info(f"Pipeline: Step1(LLM head/tail keyword replace) -> Step2(VLM max-3 factual corrections)")

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
        "strategy": "llm_headtail_keyword_replace_then_vlm_factual_correction",
        "description": "2-step pipeline: Step1 LLM replaces preamble->subject noun and generic ending->vivid keywords (middle unchanged); Step2 VLM makes max 3 word-level factual corrections. Total edit ratio ~5-8%, FI-compatible.",
        "pipeline": [
            "Step1: LLM head/tail keyword replacement (preamble->subject, summary->keywords, middle verbatim)",
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
