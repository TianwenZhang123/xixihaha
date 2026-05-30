#!/usr/bin/env python3
"""
VMAD Caption Rewriting — LLM Fusion Strategy for Layer 1 Enhancement.

Adapts P-Flow's V4 fusion strategy (Subject-First + Temporal Action Chain +
Visual Preservation) to VMAD's pipeline. This serves as a preprocessing step
before extraction, optimizing the caption for better T2V generation.

Relationship to VMAD's Three-Layer Architecture:
    Layer 1 (Text) in VMAD currently has two components:
        a) Token Decode: Δe → motion tokens (inverse mapping from embedding)
        b) Caption Rewriting: original caption → optimized caption (THIS SCRIPT)

    P-Flow demonstrated that LLM caption rewriting alone can boost:
        - CLIP +0.0139 (frame similarity)
        - XCLIP +0.0266 (temporal similarity)
    Combined with VMAD's Layer 2 (Δe) and Layer 3 (noise prior), the total
    gain should compound significantly.

Usage:
    # Rewrite a directory of captions
    python scripts/rewrite_caption.py \
        --input-dir /path/to/original_captions \
        --output-dir /path/to/rewritten_captions \
        --backend dashscope --model qwen-plus

    # Rewrite specific samples
    python scripts/rewrite_caption.py \
        --input-dir /path/to/original_captions \
        --output-dir /path/to/rewritten_captions \
        --sample-ids 7 17 21 31 32 33 34 43 46 47

    # Use local model (vLLM)
    python scripts/rewrite_caption.py \
        --input-dir /path/to/original_captions \
        --output-dir /path/to/rewritten_captions \
        --backend openai --api-base http://localhost:8000/v1 \
        --model Qwen2.5-72B-Instruct

    # Skip already rewritten files
    python scripts/rewrite_caption.py \
        --input-dir /path/to/original_captions \
        --output-dir /path/to/rewritten_captions \
        --skip-existing

V4 Rewrite Strategy (proven in P-Flow experiments):
    Three principles:
    1. Subject-First Opening — start with the action subject noun
    2. Temporal Action Chain — inject "initially/then/gradually" markers
    3. Visual Preservation — copy original visual descriptions verbatim
"""

import argparse
import json
import logging
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# V4 Rewrite Prompt (directly from P-Flow's proven run_hybrid_iter.py)
# ═══════════════════════════════════════════════════════════════════════════════

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

### Example 3 (SUV on mountain road):
INPUT: "The video depicts a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. A white SUV is seen driving on a dirt road that winds through the mountains. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.\n\nThe SUV's tire tracks are visible on the road, and its headlights illuminate the path ahead. The vehicle moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.\n\nThe combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience."
OUTPUT: "White SUV driving on a dirt road through a scenic mountainous landscape bathed in sunlight, highlighting the rugged terrain and the trees that line the path. The camera pans across the landscape, capturing the vastness of the mountains and the winding road that snakes through them. The dense vegetation adds depth to the image, with trees and bushes lining both sides of the road.\n\nThe SUV initially appears from the left side of the frame, then accelerates steadily forward along the dirt road, kicking up a growing trail of dust as it moves. The vehicle's tire tracks are visible on the road, and its headlights illuminate the path ahead. The SUV moves at a steady pace, creating a sense of progression within the stillness of the surrounding nature.\n\nThe combination of the rugged landscape, the winding dirt road, and the white SUV on the move creates a dynamic visual experience with the dust trail billowing behind the vehicle."
WHY: Subject="White SUV" (it drives), not "scenic mountainous landscape" (static). Landscape/vegetation sentences copied verbatim. SUV motion expanded with temporal chain. Ended with "dust trail billowing behind the vehicle".

Output ONLY the restructured prompt. No explanations."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Backend
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str, backend: str = "dashscope",
             model: str = "qwen-plus", api_base: Optional[str] = None,
             temperature: float = 0.5, max_retries: int = 3) -> str:
    """Call LLM API with retry logic."""
    for attempt in range(max_retries):
        try:
            if backend == "dashscope":
                return _call_dashscope(prompt, system, model, temperature)
            elif backend == "openai":
                return _call_openai(prompt, system, model, api_base, temperature)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"  LLM call failed (attempt {attempt+1}): {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _call_dashscope(prompt: str, system: str, model: str, temperature: float) -> str:
    """Call DashScope (Aliyun) API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set DASHSCOPE_API_KEY environment variable")

    client = OpenAI(
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
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _call_openai(prompt: str, system: str, model: str,
                 api_base: Optional[str], temperature: float) -> str:
    """Call OpenAI-compatible API (vLLM, etc.)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    client = OpenAI(api_key=api_key, base_url=api_base)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Quality Checks (from P-Flow V4 engineering optimizations)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_edit_ratio(original: str, rewritten: str) -> float:
    """Compute token-level edit distance ratio using SequenceMatcher."""
    orig_tokens = original.split()
    new_tokens = rewritten.split()
    matcher = SequenceMatcher(None, orig_tokens, new_tokens)
    return 1.0 - matcher.ratio()


def check_quality(original: str, rewritten: str,
                  min_length_ratio: float = 0.7,
                  max_edit_ratio: float = 0.5) -> tuple:
    """
    Check rewrite quality. Returns (passed: bool, reason: str).

    Criteria:
        1. Length ratio >= min_length_ratio (prevents over-compression)
        2. Edit ratio <= max_edit_ratio (prevents over-rewriting)
        3. Not empty
        4. Doesn't start with "The video shows/captures/depicts..."
    """
    if not rewritten.strip():
        return False, "empty output"

    orig_words = len(original.split())
    new_words = len(rewritten.split())

    if orig_words == 0:
        return True, "ok"

    length_ratio = new_words / orig_words
    if length_ratio < min_length_ratio:
        return False, f"too short (ratio={length_ratio:.2f} < {min_length_ratio})"

    edit_ratio = compute_edit_ratio(original, rewritten)
    if edit_ratio > max_edit_ratio:
        return False, f"too many changes (edit={edit_ratio:.2f} > {max_edit_ratio})"

    # Check Subject-First principle
    bad_starts = ["the video", "this video", "in this video", "the scene"]
    lower_start = rewritten.lower()[:30]
    for bad in bad_starts:
        if lower_start.startswith(bad):
            return False, f"starts with '{bad}' (violates Subject-First)"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Rewriting Logic
# ═══════════════════════════════════════════════════════════════════════════════

def rewrite_caption(original: str, backend: str, model: str,
                    api_base: Optional[str] = None,
                    max_retries: int = 2) -> str:
    """
    Rewrite a single caption using LLM fusion strategy.

    Includes quality check + automatic retry with lower temperature.
    """
    temperature = 0.5

    for attempt in range(max_retries + 1):
        rewritten = call_llm(
            prompt=original,
            system=REWRITE_SYSTEM,
            backend=backend,
            model=model,
            api_base=api_base,
            temperature=temperature,
        )

        passed, reason = check_quality(original, rewritten)
        if passed:
            return rewritten

        logger.warning(f"    Quality check failed: {reason}")
        if attempt < max_retries:
            temperature = max(0.2, temperature - 0.15)
            logger.info(f"    Retrying with temperature={temperature}")

    # Return last attempt even if quality check failed
    logger.warning(f"    All retries exhausted, using last result")
    return rewritten


def main():
    parser = argparse.ArgumentParser(
        description="VMAD Caption Rewriting — LLM Fusion Strategy",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing original captions ({id}.txt)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save rewritten captions")
    parser.add_argument("--sample-ids", type=str, nargs="*", default=None,
                        help="Only process specific sample IDs")
    parser.add_argument("--backend", type=str, default="dashscope",
                        choices=["dashscope", "openai"],
                        help="LLM backend")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="Model name")
    parser.add_argument("--api-base", type=str, default=None,
                        help="API base URL (for openai backend)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already rewritten files")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find caption files
    caption_files = sorted(args.input_dir.glob("*.txt"))
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if f.stem in id_set]

    if not caption_files:
        logger.error(f"No caption files found in {args.input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(caption_files)} captions to rewrite")
    logger.info(f"Backend: {args.backend}, Model: {args.model}")

    # Process each caption
    results = []
    success_count = 0
    skip_count = 0

    for idx, cap_path in enumerate(caption_files, 1):
        sample_id = cap_path.stem
        output_path = args.output_dir / f"{sample_id}.txt"

        if args.skip_existing and output_path.exists():
            skip_count += 1
            continue

        original = cap_path.read_text(encoding="utf-8").strip()
        if not original:
            logger.warning(f"  [{idx}] {sample_id}: empty caption, skipping")
            continue

        logger.info(f"  [{idx}/{len(caption_files)}] Rewriting {sample_id}...")

        try:
            rewritten = rewrite_caption(
                original=original,
                backend=args.backend,
                model=args.model,
                api_base=args.api_base,
            )
            output_path.write_text(rewritten, encoding="utf-8")

            edit_ratio = compute_edit_ratio(original, rewritten)
            length_ratio = len(rewritten.split()) / max(len(original.split()), 1)

            results.append({
                "id": sample_id,
                "status": "success",
                "edit_ratio": round(edit_ratio, 3),
                "length_ratio": round(length_ratio, 3),
            })
            success_count += 1

            if args.verbose:
                logger.info(f"    Edit ratio: {edit_ratio:.3f}, Length ratio: {length_ratio:.3f}")

        except Exception as e:
            logger.error(f"    FAILED: {e}")
            results.append({"id": sample_id, "status": "failed", "error": str(e)})

    # Save log
    log_path = args.output_dir / "rewrite_log.json"
    log_data = {
        "backend": args.backend,
        "model": args.model,
        "total": len(caption_files),
        "success": success_count,
        "skipped": skip_count,
        "failed": len(caption_files) - success_count - skip_count,
        "avg_edit_ratio": (
            sum(r["edit_ratio"] for r in results if r["status"] == "success") /
            max(success_count, 1)
        ),
        "results": results,
    }
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    logger.info(f"\nDone! Success={success_count}, Skipped={skip_count}, "
                f"Failed={len(caption_files) - success_count - skip_count}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
