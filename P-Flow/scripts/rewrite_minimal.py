#!/usr/bin/env python3
"""
Prompt rewrite v9-vlm: LLM pure subtraction + VLM visual supplement (2-step pipeline)

Design:
  v8-minimal showed CLIP 0.8915 < Pure L2's 0.8964.
  Root cause: LLM-fabricated camera sentences occupy UMT5 tokens but aren't grounded
  in visual facts, so they can't improve CLIP.

  v9-vlm strategy:
    Step 1: LLM pure subtraction (remove preamble + hedging + summary, add NOTHING)
            -> output becomes shorter (~70-85% of original)
    Step 2: VLM watches original video + reads shortened caption
            -> supplements missing visual details (colors, materials, lighting, spatial)
            -> output restored to ~original length
            -> every added detail is a real visual fact (VLM actually sees it)

Usage:
    python scripts/rewrite_minimal.py \
        --input-dir /path/to/baseline_captions \
        --output-dir /path/to/v9_captions \
        --video-dir /path/to/original_videos \
        --backend dashscope \
        --model qwen-plus \
        --vlm-provider dashscope
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
# Step 1: LLM Pure Subtraction System Prompt
# ============================================================================

LLM_SYSTEM_PROMPT = """You clean up VLM video captions for a T2V model. You ONLY DELETE noise. You NEVER add anything.

## Your 2 tasks (NOTHING ELSE):

### 1. SUBJECT-FIRST OPENING
Remove ONLY the "The video depicts/features/captures/shows" preamble phrase. Keep EVERYTHING else unchanged.

Rules:
- Delete ONLY the leading "The video [verb]" phrase (typically 3-5 words)
- Keep the rest of the sentence EXACTLY as-is
- Do NOT convert verb tenses (-ing stays -ing)
- Do NOT restructure or compress the sentence

### 2. REMOVE NOISE (delete ONLY these specific patterns):

DELETE these:
- Hedging trailing clauses: ", suggesting...", ", indicating that...", ", implying..."
- Overall summary sentences: any sentence starting with "The overall atmosphere/mood/scene/effect..."
- Redundant emotional trailing clauses: ", conveying a sense of...", ", adding to the...", ", creating a sense of..."
- Meta-commentary: "The video is focused on...", "The scene gives a feeling of..."

Do NOT delete:
- Any phrase containing a motion verb (walk, run, drive, fly, swim, move, ride, skate, jump, fall, climb, spin, roll, slide, flow, kick, swing, sway, drift, float)
- Any standalone sentence describing what happens in the scene
- Any visual detail (color, shape, size, position, texture, material)

IMPORTANT: When in doubt, KEEP the phrase. Over-deletion is worse than under-deletion.

## ABSOLUTE RULES:

1. ZERO additions: Do NOT add ANY new words. Your output is a STRICT SUBSET of the input words.
2. ZERO motion changes: Every verb, direction, speed description MUST appear UNCHANGED.
3. ZERO rephrasing: If you keep a phrase, it must be verbatim from the input.
4. Output will be shorter than input: This is expected. You are ONLY deleting. Typical: 70-90% of input.

## EXAMPLES:

INPUT (94 words):
"The video depicts a white SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear, suggesting it might be on a journey or adventure. As the SUV moves forward, it kicks up a cloud of dust behind it, indicating the dryness of the terrain and the speed at which it is traveling. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky. The overall atmosphere conveys a sense of exploration and outdoor adventure."

OUTPUT (63 words):
"A white SUV driving on a dusty, unpaved road through a forested area. The vehicle is equipped with roof racks carrying luggage or gear. As the SUV moves forward, it kicks up a cloud of dust behind it. The surrounding environment features tall pine trees and a scenic view of distant mountains under a clear blue sky."

Deleted: "The video depicts" (preamble); ", suggesting..." (hedging); ", indicating..." (hedging); "The overall atmosphere..." (summary). ALL motion preserved. NOTHING added.

INPUT (77 words):
"The video features a person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run. The overall atmosphere of the video is focused on the physical activity and the simplicity of the setting."

OUTPUT (56 words):
"A person running against a plain, light-colored background. The individual is wearing a white tank top and black shorts, which highlight their athletic build. The lighting is soft and even, casting minimal shadows and emphasizing the runner's movement. The person appears to be jogging at a steady pace, with their arms swinging naturally as they run."

Deleted: "The video features" (preamble); "The overall atmosphere..." (summary). ALL motion preserved. NOTHING added.

Output ONLY the cleaned caption. No explanations."""

LLM_USER_TEMPLATE = """Clean up this VLM caption ({word_count} words). ONLY do 2 things: (1) Remove "The video depicts/shows/features/captures" preamble, (2) Delete hedging trailing clauses and overall-summary sentences. Do NOT add anything. Output will be SHORTER than input.

INPUT:
{original_caption}

OUTPUT:"""

# ============================================================================
# Step 2: VLM Visual Supplement System Prompt
# ============================================================================

VLM_SUPPLEMENT_SYSTEM = """You are a visual detail specialist for text-to-video prompts. You receive:
1. A video (the original reference)
2. A shortened caption that describes the video but is missing some visual details

Your job: Watch the video carefully, then ADD specific visual details that you can SEE in the video but are NOT mentioned in the caption. Your goal is to make the caption more visually precise and restore it to approximately {target_words} words.

## What to ADD (only things you can clearly SEE in the video):
- Specific colors not mentioned (e.g., "the car is metallic silver", "golden sunlight")
- Materials/textures visible (e.g., "wooden fence", "concrete sidewalk", "leather jacket")
- Lighting conditions (e.g., "warm afternoon sunlight from the left", "overcast diffused light")
- Spatial relationships (e.g., "positioned in the lower-left of frame", "stretching into background")
- Surface details (e.g., "wet pavement reflecting lights", "dust particles in the air")
- Background elements clearly visible but not described

## Rules:
1. NEVER change or remove any existing text. Only INSERT new details.
2. NEVER add motion/action not already described (motion is handled separately by SVD).
3. NEVER add objects, people, or animals not visible in the video.
4. NEVER add temporal markers (initially, then, gradually).
5. NEVER add emotional/atmospheric interpretation. Only concrete visual facts.
6. Every detail you add MUST be something you can clearly see in the video frames.
7. Insert details naturally into existing sentences or add brief new descriptive phrases.
8. Target output length: approximately {target_words} words.

Output ONLY the enriched caption. No explanations, no labels."""

VLM_SUPPLEMENT_USER = """Watch this video carefully. Below is a caption describing this video, but it is missing some visual details. Add specific visual details you can clearly SEE in the video but are NOT in the caption. Keep ALL existing text unchanged. Only INSERT new visual details. Target: ~{target_words} words.

CURRENT CAPTION ({current_words} words):
{caption}

ENRICHED CAPTION:"""

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

def step1_llm_subtract(original: str, backend: str, model: str,
                       api_base: str = "", api_key: str = "",
                       temperature: float = 0.2, max_retries: int = 3) -> str:
    """Step 1: LLM pure subtraction (remove preamble + hedging + summary, add NOTHING)"""
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

        # Validation 2: must not be longer than original (pure subtraction)
        result_words = len(result.split())
        if result_words > word_count:
            logger.warning(f"  [Step1 retry {attempt+1}] Output longer than input: {result_words} > {word_count}")
            continue

        # Validation 3: not too short (at least 60% of original)
        if result_words < word_count * 0.60:
            logger.warning(f"  [Step1 retry {attempt+1}] Over-deleted: {result_words} < {word_count}*0.60")
            continue

        return result

    logger.warning(f"  Step1 all retries failed, using last result")
    return result if result else original


def step2_vlm_supplement(shortened_caption: str, video_path: str,
                         target_words: int, vlm_provider: str = "dashscope",
                         api_key: str = "", vlm_model: str = "qwen-vl-max",
                         vlm_model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct",
                         temperature: float = 0.4, max_retries: int = 2) -> str:
    """Step 2: VLM watches video + reads shortened caption, supplements real visual details"""
    current_words = len(shortened_caption.split())

    system_msg = VLM_SUPPLEMENT_SYSTEM.format(target_words=target_words)
    user_msg = VLM_SUPPLEMENT_USER.format(
        target_words=target_words,
        current_words=current_words,
        caption=shortened_caption,
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

            # Validation: not start with "The video"
            if result.lower().startswith(("the video", "this video", "in this video")):
                logger.warning(f"  [Step2 retry {attempt+1}] VLM output starts with preamble")
                continue

            # Validation: not too long (max 120% of target)
            result_words = len(result.split())
            if result_words > target_words * 1.20:
                logger.warning(f"  [Step2 retry {attempt+1}] VLM output too long: {result_words} > {target_words}*1.20")
                continue

            # Validation: should not be shorter than input
            if result_words < current_words:
                logger.warning(f"  [Step2 retry {attempt+1}] VLM output shorter than input: {result_words} < {current_words}")
                continue

            return result

        except Exception as e:
            logger.warning(f"  [Step2 retry {attempt+1}] VLM call failed: {e}")
            if attempt < max_retries:
                time.sleep(2)

    # All VLM attempts failed, return Step1 result (no supplement is better than bad supplement)
    logger.warning(f"  Step2 all VLM attempts failed, using Step1 result")
    return result if result else shortened_caption


def validate_rewrite(original: str, rewritten: str) -> dict:
    """Validate rewrite quality"""
    orig_words = len(original.split())
    new_words = len(rewritten.split())

    issues = []
    if not rewritten.strip():
        issues.append("empty output")
    if rewritten.lower().startswith(("the video", "this video", "in this video")):
        issues.append("still starts with preamble")
    if new_words > orig_words * 1.20:
        issues.append(f"too long ({new_words} > {orig_words}*1.20)")

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
        description="Prompt rewrite v9-vlm: LLM pure subtraction + VLM visual supplement"
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

    # VLM supplement (Step 2)
    parser.add_argument("--vlm-model", type=str, default="qwen-vl-max",
                        help="VLM model name (default: qwen-vl-max)")
    parser.add_argument("--vlm-provider", type=str, default="dashscope",
                        choices=["dashscope", "local"],
                        help="VLM backend: dashscope (remote) or local (GPU)")
    parser.add_argument("--vlm-model-path", type=str,
                        default="/root/models/Qwen2.5-VL-7B-Instruct",
                        help="Local VLM model path (only for --vlm-provider local)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM supplement step (only do LLM subtraction)")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature (default: 0.2, low for precise deletion)")
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
            logger.error("VLM supplement requires --video-dir, or use --skip-vlm to skip")
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
    logger.info(f"v9-subtract-supplement: LLM subtraction -> VLM visual supplement")
    logger.info(f"{'='*60}")
    logger.info(f"Samples: {len(caption_files)}")
    logger.info(f"LLM: {args.backend}/{args.model}, temp={args.temperature}")
    if not args.skip_vlm:
        logger.info(f"VLM: {args.vlm_provider}/{args.vlm_model}")
    else:
        logger.info(f"VLM: DISABLED (--skip-vlm)")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pipeline: Step1(LLM delete preamble/hedging/summary) -> Step2(VLM add visual details)")

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
        # Step 1: LLM Pure Subtraction
        # ====================================================================
        try:
            step1_result = step1_llm_subtract(
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
                f"{orig_words}->{step1_words} words (deleted {orig_words - step1_words})"
            )

        except Exception as e:
            failed += 1
            logger.error(f"  [{idx}/{len(caption_files)}] {sample_id} Step1 FAILED: {e}")
            results.append({"sample_id": sample_id, "status": "failed", "error": str(e)})
            if idx < len(caption_files):
                time.sleep(args.delay)
            continue

        # ====================================================================
        # Step 2: VLM Visual Supplement
        # ====================================================================
        final_result = step1_result  # Default fallback

        if not args.skip_vlm and video_dir:
            video_path = video_dir / f"{sample_id}.mp4"
            if video_path.exists():
                try:
                    final_result = step2_vlm_supplement(
                        shortened_caption=step1_result,
                        video_path=str(video_path),
                        target_words=orig_words,
                        vlm_provider=args.vlm_provider,
                        api_key=api_key,
                        vlm_model=args.vlm_model,
                        vlm_model_path=args.vlm_model_path,
                        temperature=0.4,
                        max_retries=2,
                    )

                    final_words = len(final_result.split())
                    logger.info(
                        f"  [{idx}/{len(caption_files)}] {sample_id} Step2: "
                        f"{step1_words}->{final_words} words (target {orig_words})"
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
            logger.info(f"Avg Step1 compression: {avg_step1:.2f}, Avg final ratio: {avg_final:.2f} (target ~1.0)")

    # Save processing log
    log_file = output_dir / "rewrite_log.json"
    log_data = {
        "version": "v9_subtract_supplement",
        "strategy": "llm_pure_subtraction_then_vlm_visual_supplement",
        "description": "2-step pipeline: Step1 LLM pure deletion (preamble+hedging+summary) -> Step2 VLM watches video and adds real visual details to restore length",
        "pipeline": [
            "Step1: LLM subtraction (remove preamble, hedging clauses, overall-summary sentences, add NOTHING)",
            "Step2: VLM supplement (watch original video + read shortened caption, add grounded visual details)",
        ],
        "key_insight": "LLM-fabricated content (camera/spatial sentences) hurts CLIP because not visually grounded. VLM sees actual video so every supplement is a real visual fact.",
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
