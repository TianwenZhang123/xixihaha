#!/usr/bin/env python3
"""
VMAD Iterative Refinement Pipeline — VLM Feedback Loop for Layer 1 Optimization.

This is the VMAD equivalent of P-Flow's run_hybrid_iter.py, adapted for
VMAD's three-layer architecture. The key difference: P-Flow iterates ONLY
on text (Layer 1), while VMAD iterates on text that is INFORMED by Layer 2
(Δe) metrics, making the feedback loop more targeted.

Pipeline:
    1. Extract motion asset (Δe + η_motion) from reference video
    2. LLM rewrite original caption → iter0 prompt (V4 fusion strategy)
    3. Loop N iterations:
       a. Apply asset with current prompt → generate video
       b. Compute reproduction metrics (CLIP/XCLIP)
       c. VLM compare (reference vs generated) → structured diff
       d. LLM refine prompt based on feedback
    4. Select best iteration based on metrics
    5. Output final optimized prompt + video

Key Advantages over P-Flow's Approach:
    - Layer 2 (Δe) provides a continuous motion floor — even if text is
      imperfect, Δe ensures motion fidelity
    - VLM feedback can be more targeted: focus on CONTENT/STYLE differences
      since motion is already handled by Δe
    - Fewer iterations needed (typically 1-2 vs P-Flow's 3+)

Usage:
    # Full iterative refinement on selected samples
    python scripts/run_iterative_refinement.py \
        --video-dir /path/to/videos \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/iterative_refine \
        --sample-ids 7 17 21 31 32 33 34 43 46 47 \
        --iter 2

    # Skip extraction (use pre-extracted assets)
    python scripts/run_iterative_refinement.py \
        --video-dir /path/to/videos \
        --caption-dir /path/to/captions \
        --asset-dir ./outputs/assets \
        --output-dir ./outputs/iterative_refine \
        --sample-ids 7 17 21 31 \
        --iter 2

    # Compare against P-Flow baseline
    python scripts/run_iterative_refinement.py \
        --video-dir /path/to/videos \
        --caption-dir /path/to/captions \
        --output-dir ./outputs/iterative_refine \
        --baseline-json /path/to/pflow/eval_results.json \
        --iter 3
"""

import argparse
import json
import logging
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Integration (reuses P-Flow's proven V4 strategy)
# ═══════════════════════════════════════════════════════════════════════════════

REWRITE_SYSTEM = """You restructure VLM video captions into better T2V generation prompts. Your output must be nearly identical to the input — you only make 3 surgical changes.

## CRITICAL: How to identify the ACTION SUBJECT
The action subject is THE THING THAT MOVES OR ACTS in the video. Ask: "What is performing the main motion?"
The subject is NEVER the background/environment/setting. It is always the moving entity.

## The 3 changes you make (NOTHING ELSE):

1. OPENING — Move the action subject to the very first words. Delete "The video shows/captures/depicts/showcases..." and start with the subject noun phrase.

2. ACTION — Find the 1-2 sentences where the subject's motion is described. Convert static verbs to a temporal chain using "initially/then/gradually". Add motion direction if implied but not stated. Do NOT touch any other sentences.

3. ENDING — Make the final phrase end with a vivid motion or visual keyword.

## What you must NOT do:
- Do NOT compress or summarize. If the input is 150 words, output ~150 words.
- Do NOT rephrase visual descriptions. Copy them VERBATIM.
- Do NOT merge paragraphs.
- Do NOT add information the original doesn't mention.

Output ONLY the restructured prompt. No explanations."""


REFINE_SYSTEM = """You fix video generation prompts based on VLM feedback. You make SURGICAL fixes — change only what the VLM says is wrong, leave everything else VERBATIM.

## Your constraints:
- Fix ONLY the top 1-2 differences the VLM identified. Do NOT touch anything else.
- The current prompt follows Subject-First Opening + Temporal Action Chain structure. PRESERVE this.

## What you must NOT do:
- Do NOT rewrite the entire prompt. Copy it and make targeted edits.
- Do NOT compress. Output word count within ±15% of input.
- Do NOT rephrase visual descriptions that the VLM did NOT flag.

## How to fix common VLM feedback:
- "Motion direction wrong" → change direction words only
- "Subject appearance differs" → adjust specific attribute only
- "Background differs" → add/modify specific detail only
- "Speed/intensity differs" → adjust temporal adverbs only

Output ONLY the fixed prompt. No explanations."""


VLM_COMPARE_PROMPT = """You are comparing two videos. The TOP half is the REFERENCE (ground truth), the BOTTOM half is the GENERATED video.

Analyze differences in these 4 dimensions ONLY:
1. SUBJECT: Main entity identity (species, color, size, count)
2. MOTION: Direction, speed, trajectory
3. BACKGROUND: Scene elements, colors, lighting
4. TIMING: Action sequence order

Format:
SUBJECT: [differences or "matches"]
MOTION: [differences or "matches"]
BACKGROUND: [differences or "matches"]
TIMING: [differences or "matches"]

Focus on the 1-2 biggest differences for prompt correction."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Call Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str, model: str = "qwen-plus",
             temperature: float = 0.5, max_retries: int = 3) -> str:
    """Call DashScope LLM with retry logic."""
    import openai

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set DASHSCOPE_API_KEY environment variable")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    for attempt in range(max_retries):
        try:
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
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"  LLM call failed (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def compute_edit_ratio(text_a: str, text_b: str) -> float:
    """Token-level edit distance ratio (0=identical, 1=completely different)."""
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    return 1.0 - SequenceMatcher(None, tokens_a, tokens_b).ratio()


def llm_rewrite(caption: str, model: str = "qwen-plus",
                max_retries: int = 2) -> str:
    """LLM V4 fusion strategy initial rewrite with quality validation."""
    word_count = len(caption.split())
    user_msg = (
        f"Restructure this VLM caption ({word_count} words). "
        f"First, identify what MOVES — that is your action subject. "
        f"Then make ONLY 3 changes: "
        f"(1) move subject to first word, "
        f"(2) add temporal chain to motion sentences, "
        f"(3) end with motion/visual keyword. "
        f"Output ~{word_count} words (±15%). Do NOT compress.\n\n"
        f"INPUT:\n{caption}\n\nOUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = max(0.3, 0.5 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)

        # Quality check: length
        result_words = len(result.split())
        ratio = result_words / max(word_count, 1)
        if ratio < 0.70:
            logger.warning(f"  [retry {attempt+1}] too short: {result_words}/{word_count}")
            continue

        # Quality check: edit distance
        edit_ratio = compute_edit_ratio(caption, result)
        if edit_ratio > 0.50:
            logger.warning(f"  [retry {attempt+1}] too much change: {edit_ratio:.0%}")
            continue

        return result

    return result


def llm_refine(current_prompt: str, vlm_feedback: str,
               model: str = "qwen-plus", max_retries: int = 2) -> str:
    """LLM refine prompt based on VLM feedback."""
    user_msg = (
        f"## Current Prompt:\n{current_prompt}\n\n"
        f"## VLM Feedback:\n{vlm_feedback}\n\n"
        f"Fix the prompt. Output ONLY the fixed prompt:"
    )

    for attempt in range(max_retries + 1):
        temp = max(0.2, 0.4 - attempt * 0.1)
        result = call_llm(user_msg, REFINE_SYSTEM, model, temperature=temp)

        # Quality check
        result_words = len(result.split())
        ratio = result_words / max(len(current_prompt.split()), 1)
        if ratio < 0.70:
            continue

        edit_ratio = compute_edit_ratio(current_prompt, result)
        if edit_ratio > 0.35:
            continue

        return result

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# VLM Compare
# ═══════════════════════════════════════════════════════════════════════════════

def vlm_compare(ref_video: str, gen_video: str, vlm_client) -> str:
    """VLM structured comparison between reference and generated video."""
    from src.video_utils import load_video, save_video_tensor

    try:
        # Load and vertically concatenate
        ref = load_video(ref_video, num_frames=81, height=480, width=832, device="cpu")
        gen = load_video(gen_video, num_frames=81, height=480, width=832, device="cpu")

        # Stack vertically: (C, F, 2*H, W)
        import torch
        composite = torch.cat([ref, gen], dim=2)  # concat along H
        composite_path = "/tmp/vmad_iter_composite.mp4"
        save_video_tensor(composite, composite_path, fps=15)
        del ref, gen, composite

        # Call VLM
        frames_pil = vlm_client._extract_frames_pil(composite_path, num_frames=16)
        if not frames_pil:
            return "Unable to extract frames."

        content_list = []
        for img in frames_pil:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": VLM_COMPARE_PROMPT})

        messages = [{"role": "user", "content": content_list}]
        response = vlm_client._generate(messages)
        if response and len(response.strip()) > 10:
            return response.strip()

    except Exception as e:
        logger.warning(f"  VLM compare failed: {e}")

    return "Unable to analyze differences."


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_quick_eval(orig_dir: Path, gen_dir: Path, caption_dir: Path,
                   output_dir: Path) -> dict:
    """Run reproduction evaluation and return metrics dict."""
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "evaluation" / "run_reproduction_eval.py"),
        "--orig-dir", str(orig_dir),
        "--gen-dir", str(gen_dir),
        "--caption-dir", str(caption_dir),
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        logger.error(f"  Eval failed:\n{result.stderr[-500:]}")
        return {}

    json_path = output_dir / "eval_results.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="VMAD Iterative Refinement — VLM Feedback Loop"
    )
    # I/O
    p.add_argument("--video-dir", type=Path, required=True,
                   help="Original video directory")
    p.add_argument("--caption-dir", type=Path, required=True,
                   help="Original caption directory")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--asset-dir", type=Path, default=None,
                   help="Pre-extracted assets (skip extraction)")

    # Samples
    p.add_argument("--sample-ids", type=str, nargs="+", default=None,
                   help="Specific sample IDs to process")
    p.add_argument("--limit", type=int, default=0)

    # Iteration
    p.add_argument("--iter", type=int, default=2,
                   help="Number of refinement iterations")

    # LLM/VLM
    p.add_argument("--llm-model", type=str, default="qwen-plus")
    p.add_argument("--vlm-provider", type=str, default="local")
    p.add_argument("--vlm-path", type=str,
                   default="/root/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--skip-vlm", action="store_true",
                   help="Skip VLM comparison (LLM-only iteration)")

    # VMAD config
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Comparison
    p.add_argument("--baseline-json", type=Path, default=None,
                   help="P-Flow baseline eval_results.json for comparison")

    # Control
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("DASHSCOPE_API_KEY is required")
        sys.exit(1)

    # ── Discover samples ──
    caption_files = sorted(args.caption_dir.glob("*.txt"))
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if f.stem in id_set]
    if args.limit > 0:
        caption_files = caption_files[:args.limit]

    sample_ids = [f.stem for f in caption_files]
    if not sample_ids:
        logger.error(f"No samples found in {args.caption_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(sample_ids)} samples: {sample_ids[:10]}...")

    # ── Step 0: Extract motion assets (if not pre-extracted) ──
    assets_dir = args.asset_dir or (out_dir / "assets")
    if not args.asset_dir:
        logger.info("=" * 60)
        logger.info("Step 0: Extracting motion assets")
        logger.info("=" * 60)

        from src.pipeline import VMADPipeline, VMADConfig
        config = VMADConfig()
        if args.model_path:
            config.t2v_path = args.model_path
        config.seed = args.seed

        pipeline = VMADPipeline(config)

        for idx, sid in enumerate(sample_ids, 1):
            asset_path = assets_dir / sid / "asset"
            if args.resume and (asset_path / "asset.json").exists():
                logger.info(f"  [{idx}/{len(sample_ids)}] {sid} - SKIP (exists)")
                continue

            video_path = args.video_dir / f"{sid}.mp4"
            caption = (args.caption_dir / f"{sid}.txt").read_text(encoding="utf-8").strip()

            if not video_path.exists():
                logger.warning(f"  [{idx}] Video not found: {video_path}")
                continue

            logger.info(f"  [{idx}/{len(sample_ids)}] Extracting {sid}...")
            try:
                pipeline.extract(
                    video_path=str(video_path),
                    output_dir=str(assets_dir / sid),
                    caption=caption,
                )
            except Exception as e:
                logger.error(f"  FAILED: {e}")

    # ── Step 1: LLM rewrite → iter0 prompts ──
    logger.info("=" * 60)
    logger.info("Step 1: LLM V4 Rewrite → iter0 prompts")
    logger.info("=" * 60)

    iter0_dir = out_dir / "captions_iter0"
    iter0_dir.mkdir(exist_ok=True)

    for sid in sample_ids:
        out_file = iter0_dir / f"{sid}.txt"
        if args.resume and out_file.exists():
            continue

        original = (args.caption_dir / f"{sid}.txt").read_text(encoding="utf-8").strip()
        rewritten = llm_rewrite(original, args.llm_model)
        out_file.write_text(rewritten, encoding="utf-8")

        edit_r = compute_edit_ratio(original, rewritten)
        logger.info(f"  [{sid}] edit_ratio={edit_r:.2f}: {rewritten[:50]}...")

    # ── Step 2-N: Iterative loop ──
    from src.pipeline import VMADPipeline, VMADConfig

    config = VMADConfig()
    if args.model_path:
        config.t2v_path = args.model_path
    config.seed = args.seed
    pipeline = VMADPipeline(config)

    all_iter_metrics = []

    for iteration in range(1, args.iter + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Iteration {iteration}/{args.iter}")
        logger.info(f"{'=' * 60}")

        # Current caption dir
        if iteration == 1:
            current_caption_dir = iter0_dir
        else:
            current_caption_dir = out_dir / f"captions_iter{iteration - 1}"

        # Generate videos using assets + current prompt
        iter_gen_dir = out_dir / f"generated_iter{iteration}"
        iter_gen_dir.mkdir(exist_ok=True)

        logger.info(f"  [Generate] Applying assets with prompts from {current_caption_dir.name}")

        for sid in sample_ids:
            gen_path = iter_gen_dir / f"{sid}.mp4"
            if args.resume and gen_path.exists():
                continue

            asset_path = str(assets_dir / sid / "asset")
            if not Path(asset_path).exists():
                logger.warning(f"  [{sid}] No asset, skipping")
                continue

            prompt = (current_caption_dir / f"{sid}.txt").read_text(encoding="utf-8").strip()

            try:
                result = pipeline.apply(
                    content_prompt=prompt,
                    asset_dir=asset_path,
                    output_dir=str(out_dir / "tmp_apply" / sid),
                    strength=1.0,
                )
                # Copy to flat structure
                import shutil
                src = Path(result["video_path"])
                if src.exists():
                    shutil.copy2(str(src), str(gen_path))
            except Exception as e:
                logger.error(f"  [{sid}] Apply failed: {e}")

        # Evaluate
        eval_dir = out_dir / f"eval_iter{iteration}"
        metrics = run_quick_eval(
            args.video_dir, iter_gen_dir, current_caption_dir, eval_dir
        )

        clip_score = metrics.get("orig_gen_clip_mean", 0)
        xclip_score = metrics.get("orig_gen_xclip_mean", 0)
        logger.info(f"  [Eval] Iter {iteration}: CLIP={clip_score:.4f}, XCLIP={xclip_score:.4f}")

        all_iter_metrics.append({
            "iteration": iteration,
            "orig_gen_clip": clip_score,
            "orig_gen_xclip": xclip_score,
        })

        # VLM compare + LLM refine (if not last iteration)
        if iteration < args.iter:
            next_caption_dir = out_dir / f"captions_iter{iteration}"
            next_caption_dir.mkdir(exist_ok=True)

            if args.skip_vlm:
                logger.info("  [LLM] Self-iteration (no VLM)...")
                for sid in sample_ids:
                    cur_prompt = (current_caption_dir / f"{sid}.txt").read_text(encoding="utf-8").strip()
                    refined = llm_refine(
                        cur_prompt,
                        "Improve motion clarity and temporal precision.",
                        args.llm_model,
                    )
                    (next_caption_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
            else:
                logger.info("  [VLM+LLM] Compare + Refine...")
                from src.vlm_client import create_vlm_client
                vlm_client = create_vlm_client({
                    "provider": args.vlm_provider,
                    "model_path": args.vlm_path,
                    "lazy_load": True,
                })

                for sid in sample_ids:
                    ref_video = str(args.video_dir / f"{sid}.mp4")
                    gen_video = str(iter_gen_dir / f"{sid}.mp4")
                    cur_prompt = (current_caption_dir / f"{sid}.txt").read_text(encoding="utf-8").strip()

                    if not Path(gen_video).exists():
                        (next_caption_dir / f"{sid}.txt").write_text(cur_prompt, encoding="utf-8")
                        continue

                    feedback = vlm_compare(ref_video, gen_video, vlm_client)
                    logger.info(f"  [{sid}] VLM: {feedback[:60]}...")

                    refined = llm_refine(cur_prompt, feedback, args.llm_model)
                    (next_caption_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")

    # ── Summary ──
    logger.info(f"\n{'=' * 60}")
    logger.info("Summary: All Iterations vs Baseline")
    logger.info(f"{'=' * 60}")

    baseline_clip = 0
    baseline_xclip = 0
    if args.baseline_json and args.baseline_json.exists():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_clip = baseline.get("orig_gen_clip_mean", 0)
        baseline_xclip = baseline.get("orig_gen_xclip_mean", 0)

    print(f"\n{'─' * 70}")
    print(f"{'Iter':<8} {'CLIP':>10} {'Δ CLIP':>10} {'XCLIP':>10} {'Δ XCLIP':>10}")
    print(f"{'─' * 70}")
    if baseline_clip > 0:
        print(f"{'baseline':<8} {baseline_clip:>10.4f} {'—':>10} {baseline_xclip:>10.4f} {'—':>10}")

    for m in all_iter_metrics:
        d_clip = m["orig_gen_clip"] - baseline_clip if baseline_clip else 0
        d_xclip = m["orig_gen_xclip"] - baseline_xclip if baseline_xclip else 0
        print(f"{'iter'+str(m['iteration']):<8} {m['orig_gen_clip']:>10.4f} "
              f"{d_clip:>+10.4f} {m['orig_gen_xclip']:>10.4f} {d_xclip:>+10.4f}")
    print(f"{'─' * 70}\n")

    # Save summary
    summary = {
        "baseline": {"orig_gen_clip": baseline_clip, "orig_gen_xclip": baseline_xclip},
        "iterations": all_iter_metrics,
        "config": {
            "iter": args.iter,
            "llm_model": args.llm_model,
            "vlm_provider": args.vlm_provider if not args.skip_vlm else "skipped",
            "sample_ids": sample_ids,
            "seed": args.seed,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(f"Summary saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
