#!/usr/bin/env python3
"""
Compute M_d (Motion Definiteness) scores for video captions using LLM.

Uses DashScope Qwen API (same as VLM rewrite) to classify each caption into:
  1 = No motion (static scene)
  2 = Ambient motion (wind, water, camera movement)
  3 = Object motion (people/animals/vehicles actively moving)

Output: data/md_scores.csv with columns: sample_id, md_raw, md_score

Usage:
    # Score all captions in a directory
    python scripts/compute_md.py --caption_dir data/captions_qwen --output data/md_scores.csv

    # Score specific samples
    python scripts/compute_md.py --caption_dir data/captions_qwen --output data/md_scores.csv --sample_ids 7 21 32

    # Resume (skip already scored)
    python scripts/compute_md.py --caption_dir data/captions_qwen --output data/md_scores.csv --resume
"""

import os
import csv
import time
import json
import argparse
import logging
from pathlib import Path

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── M_d 评分 Prompt ──

MD_SYSTEM_PROMPT = (
    "You are a video motion classification expert. "
    "Your task is to classify the type of motion described in a video caption."
)

MD_USER_TEMPLATE = """Classify the PRIMARY source of motion in this video caption:

1 = No motion: static scene, still life, landscape, architecture — nothing moves.
2 = Camera/scene motion: the VIEWPOINT moves (FPV, drone, tracking shot, zoom, flythrough, hyperlapse), or the SCENE transforms (plants growing, objects morphing, smoke/steam rising, water flowing, objects deforming). The key signal is: no character/subject is locomoting through space under its own power. Even fast camera movement counts here if no subject is independently moving.
3 = Object/subject motion: a person, animal, vehicle, or physical object is actively moving through space under its own power (walking, running, flying, swimming, chasing, colliding, accelerating). The camera may ALSO move, but the PRIMARY motion source is the subject's own physical displacement.

IMPORTANT: Choose based on the DOMINANT motion SOURCE, not intensity:
- "FPV drone flying fast" → 2 (camera motion, no subject locomoting)
- "A cat running through a garden" → 3 (subject locomoting)
- "Zoom into a dandelion" → 2 (camera zoom, no subject moving)
- "Sailboats sailing on coffee" → 3 (subjects moving through water)
- "Warehouse plants exploding from ground" → 2 (scene transformation, not locomotion)
- "Water splashing in slow motion" → 2 (fluid dynamics, no subject locomoting)

Caption: {caption}

Answer with ONLY a single number: 1, 2, or 3."""

# ── M_d 映射表 ──

MD_RAW_TO_SCORE = {
    1: 0.0,   # 无运动
    2: 0.3,   # 环境动态
    3: 1.0,   # 物体运动
}


def create_dashscope_client(api_key=None, base_url=None):
    """Create DashScope OpenAI-compatible client (same as vlm_client.py VLMClient)."""
    if not HAS_OPENAI:
        raise ImportError("openai package required. Install: pip install openai")

    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "DashScope API key required. Set DASHSCOPE_API_KEY env var "
            "or pass --api_key parameter."
        )

    base_url = base_url or os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    return openai.OpenAI(api_key=api_key, base_url=base_url)


def score_caption(client, caption: str, model_name: str = "qwen-plus",
                  max_retries: int = 3) -> int:
    """
    Score a single caption using LLM.

    Returns:
        int: 1, 2, or 3
    """
    user_msg = MD_USER_TEMPLATE.format(caption=caption)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": MD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,  # Low temperature for deterministic classification
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()

            # Parse response: extract first digit
            for char in raw:
                if char in "123":
                    return int(char)

            # If no digit found, try parsing as int
            logger.warning(f"Unexpected LLM response: '{raw}', defaulting to 2")
            return 2

        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error("All retries failed, defaulting to 2")
    return 2


def load_existing_scores(output_path: str) -> dict:
    """Load existing scores from CSV for resume support."""
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[int(row["sample_id"])] = {
                    "md_raw": int(row["md_raw"]),
                    "md_score": float(row["md_score"]),
                }
    return existing


def main():
    parser = argparse.ArgumentParser(description="Compute M_d scores for video captions")
    parser.add_argument("--caption_dir", type=str, required=True,
                        help="Directory containing caption files ({id}.txt)")
    parser.add_argument("--output", type=str, default="data/md_scores.csv",
                        help="Output CSV path (default: data/md_scores.csv)")
    parser.add_argument("--sample_ids", type=int, nargs="+",
                        help="Only score specific sample IDs")
    parser.add_argument("--model_name", type=str, default="qwen-plus",
                        help="DashScope model name (default: qwen-plus)")
    parser.add_argument("--api_key", type=str, default="",
                        help="DashScope API key (or set DASHSCOPE_API_KEY)")
    parser.add_argument("--base_url", type=str, default="",
                        help="DashScope base URL (or set DASHSCOPE_BASE_URL)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already scored samples")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create client
    client = create_dashscope_client(api_key=args.api_key, base_url=args.base_url)
    logger.info(f"DashScope client created, model={args.model_name}")

    # Discover caption files
    caption_dir = Path(args.caption_dir)
    if not caption_dir.exists():
        logger.error(f"Caption directory not found: {args.caption_dir}")
        return

    caption_files = sorted(caption_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    logger.info(f"Found {len(caption_files)} caption files")

    # Load existing scores (for resume)
    existing = {}
    if args.resume:
        existing = load_existing_scores(args.output)
        logger.info(f"Resuming: {len(existing)} samples already scored")

    # Score each caption
    results = []
    if existing and args.resume:
        # Start with existing scores
        for sid, scores in existing.items():
            results.append({
                "sample_id": sid,
                "md_raw": scores["md_raw"],
                "md_score": scores["md_score"],
            })

    scored_count = len(existing) if args.resume else 0
    skipped_count = 0

    for idx, cap_file in enumerate(caption_files, 1):
        sample_id = int(cap_file.stem)

        # Skip if already scored
        if args.resume and sample_id in existing:
            skipped_count += 1
            continue

        # Read caption
        caption = cap_file.read_text(encoding="utf-8").strip()
        if not caption:
            logger.warning(f"Empty caption for sample {sample_id}, defaulting to M_d=2")
            md_raw = 2
        else:
            md_raw = score_caption(client, caption, model_name=args.model_name)

        md_score = MD_RAW_TO_SCORE[md_raw]
        results.append({
            "sample_id": sample_id,
            "md_raw": md_raw,
            "md_score": md_score,
        })
        scored_count += 1

        label = {1: "no-motion", 2: "ambient", 3: "object-motion"}
        logger.info(
            f"  [{idx}/{len(caption_files)}] ID={sample_id}: "
            f"md_raw={md_raw} ({label[md_raw]}), M_d={md_score:.1f}"
        )

        # Rate limit: be gentle with API
        time.sleep(0.5)

    # Sort by sample_id
    results.sort(key=lambda x: x["sample_id"])

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "md_raw", "md_score"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    raw_counts = {1: 0, 2: 0, 3: 0}
    for r in results:
        raw_counts[r["md_raw"]] += 1

    logger.info(f"\n{'='*50}")
    logger.info(f"M_d Scoring Complete!")
    logger.info(f"  Total scored: {scored_count}")
    logger.info(f"  Skipped (resume): {skipped_count}")
    logger.info(f"  Distribution:")
    logger.info(f"    1 (no-motion):    {raw_counts[1]}")
    logger.info(f"    2 (ambient):      {raw_counts[2]}")
    logger.info(f"    3 (object-motion): {raw_counts[3]}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
