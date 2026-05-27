#!/usr/bin/env python3
"""
Hybrid Iterative Pipeline — 一体化脚本

流程：
  1. 读取 baseline 的 VLM caption
  2. LLM 融合策略改写 → iter0 prompt
  3. 循环 N 轮：
     a. 用当前 prompt 调 run.py 生成视频
     b. 调评测脚本算 CLIP/XCLIP
     c. VLM 对比（原始 vs 生成）→ 差异分析
     d. LLM 根据反馈修复 prompt
  4. 汇总所有轮次指标 + baseline 对比

用法:
    cd /root/autodl-tmp/videofake/P-Flow

    export DASHSCOPE_API_KEY="sk-xxxxx"

    python scripts/run_hybrid_iter.py \
        --data_dir /root/autodl-tmp/data/video-200/water_mark_out \
        --baseline_dir /root/autodl-tmp/outputs/baseline \
        --output_dir /root/autodl-tmp/outputs/hybrid_iter \
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \
        --iter 3
"""

import sys
import os
import json
import subprocess
import time
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
# LLM 改写
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You are a video prompt optimization expert. Rewrite the given text-to-video prompt following the "Hybrid Strategy":

## Three Principles:
1. Subject-First Opening — first word(s) MUST be the concrete main subject noun(s). Remove "The video shows..."
2. Inject Temporal Action Chain — add "initially...", "then...", "gradually..." to describe motion over time.
3. Preserve Original Visual Vocabulary — DO NOT replace specific visual descriptors (colors, materials, textures).

## Constraints:
- Output ONLY the rewritten prompt. No explanations.
- Keep word count within ±20% of the original.
- Write in English."""

REFINE_SYSTEM = """You are a video prompt optimization expert. You will receive:
1. The current prompt
2. A VLM's analysis comparing the generated video with the reference (differences)

Fix the prompt to address the differences, while STILL following:
1. Subject-First Opening — keep first word(s) as main subject noun(s).
2. Temporal Action Chain — maintain/improve temporal markers.
3. Preserve Visual Vocabulary — only modify parts the VLM identified as different.

Rules:
- Make TARGETED fixes (top 1-2 differences). Do NOT rewrite everything.
- Output ONLY the fixed prompt. No explanations.
- Keep word count similar (±20%). English only."""


def call_llm(prompt: str, system: str, model: str = "qwen-plus") -> str:
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
        temperature=0.7,
        max_tokens=1024,
    )
    result = response.choices[0].message.content.strip()
    # 清理引号
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    if result.startswith("'") and result.endswith("'"):
        result = result[1:-1]
    return result


def llm_rewrite(caption: str, model: str = "qwen-plus") -> str:
    """LLM 融合策略初始改写"""
    user_msg = f"Original prompt ({len(caption.split())} words):\n\n{caption}\n\nRewrite following the Hybrid Strategy. Output ONLY the rewritten prompt:"
    return call_llm(user_msg, REWRITE_SYSTEM, model)


def llm_refine(current_prompt: str, vlm_feedback: str, model: str = "qwen-plus") -> str:
    """LLM 根据 VLM 反馈修复"""
    user_msg = f"## Current Prompt:\n{current_prompt}\n\n## VLM Feedback:\n{vlm_feedback}\n\nFix the prompt. Output ONLY the fixed prompt:"
    return call_llm(user_msg, REFINE_SYSTEM, model)


# ─────────────────────────────────────────────────────────────────────────────
# 调用 run.py 生成视频
# ─────────────────────────────────────────────────────────────────────────────

def generate_videos(data_dir: str, caption_dir: str, output_dir: str,
                    sample_ids: list, args) -> None:
    """调用 run.py 生成视频"""
    cmd = [
        sys.executable, "run.py",
        "--data_dir", data_dir,
        "--caption_dir", caption_dir,
        "--output_dir", output_dir,
        "--sample_ids", *[str(s) for s in sample_ids],
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--seed", str(args.seed),
        "--vlm_provider", "mock",  # 不需要 VLM（我们自己管 prompt）
    ]
    if args.resume:
        cmd.append("--resume")

    logger.info(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent),
                           capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  run.py 失败:\n{result.stderr[-500:]}")
        raise RuntimeError("run.py failed")


# ─────────────────────────────────────────────────────────────────────────────
# 调用评测脚本
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
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent),
                           capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  评测失败:\n{result.stderr[-500:]}")
        return {}

    # 读取结果
    json_path = Path(output_dir) / "eval_results.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# VLM 对比（调用现有 VLM 接口）
# ─────────────────────────────────────────────────────────────────────────────

def vlm_compare(ref_video: str, gen_video: str, vlm_client) -> str:
    """VLM 对比两个视频，返回差异分析"""
    from src.video_utils import load_video, save_video_tensor, create_vertical_composite

    # 加载并拼接（CPU 上操作）
    ref = load_video(ref_video, num_frames=81, height=480, width=832, device="cpu")
    gen = load_video(gen_video, num_frames=81, height=480, width=832, device="cpu")
    composite = create_vertical_composite([ref, gen])
    composite_path = "/tmp/hybrid_iter_composite.mp4"
    save_video_tensor(composite, composite_path, fps=15)
    del ref, gen, composite

    try:
        result = vlm_client.analyze_and_refine(
            composite_video_path=composite_path,
            current_prompt="[Comparing reference vs generated]",
            iteration=1,
            i_max=1,
        )
        analysis = result.get("analysis", {})
        comparison = analysis.get("comparison", "")
        if comparison:
            return comparison
        # fallback
        parts = []
        if analysis.get("reference_description"):
            parts.append(f"Reference: {analysis['reference_description']}")
        if analysis.get("new_generated_description"):
            parts.append(f"Generated: {analysis['new_generated_description']}")
        if parts:
            return "\n".join(parts)
    except Exception as e:
        logger.warning(f"  VLM compare failed: {e}")

    return "Unable to analyze differences."


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def make_flat_dir(output_dir: str, sample_ids: list) -> str:
    """创建 flat 目录（软链接），供评测使用"""
    flat_dir = Path(output_dir) / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        sample_dir = Path(output_dir) / f"sample_{sid}"
        src = sample_dir / f"{sid}.mp4"
        dst = flat_dir / f"{sid}.mp4"
        if src.exists():
            dst.unlink(missing_ok=True)
            os.symlink(src, dst)
    return str(flat_dir)


def main():
    p = argparse.ArgumentParser(description="Hybrid Iterative Pipeline 一体化脚本")

    # I/O
    p.add_argument("--data_dir", type=str, required=True,
                   help="原始视频目录")
    p.add_argument("--baseline_dir", type=str, required=True,
                   help="baseline 输出目录（读取 VLM caption + 评测对比）")
    p.add_argument("--output_dir", type=str, required=True,
                   help="本次实验输出目录")
    p.add_argument("--sample_ids", type=int, nargs="+", required=True,
                   help="样本 ID 列表")

    # 迭代
    p.add_argument("--iter", type=int, default=3, help="迭代轮数")

    # 生成参数（与 baseline 保持一致）
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    # LLM
    p.add_argument("--llm_model", type=str, default="qwen-plus")

    # VLM（用于迭代对比）
    p.add_argument("--vlm_provider", type=str, default="local")
    p.add_argument("--vlm_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct")

    # 控制
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_vlm", action="store_true",
                   help="跳过 VLM 对比（仅用 LLM 自主迭代改写，用于快速测试）")

    args = p.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("需要设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取 baseline 评测结果（用于最终对比）──
    baseline_eval_path = Path(args.baseline_dir) / "eval_clip" / "eval_results.json"
    baseline_metrics = {}
    if baseline_eval_path.exists():
        baseline_metrics = json.loads(baseline_eval_path.read_text())
        logger.info(f"Baseline 指标: CLIP={baseline_metrics.get('orig_gen_clip_mean', 'N/A'):.4f}, "
                    f"XCLIP={baseline_metrics.get('orig_gen_xclip_mean', 'N/A'):.4f}")

    # ── Step 1: 读取 baseline VLM caption + LLM 初始改写 ──
    logger.info("=" * 60)
    logger.info("Step 1: 读取 baseline caption → LLM 融合策略改写")
    logger.info("=" * 60)

    caption_dir_iter0 = out_dir / "captions_iter0"
    caption_dir_iter0.mkdir(exist_ok=True)

    for sid in args.sample_ids:
        out_file = caption_dir_iter0 / f"{sid}.txt"
        if args.resume and out_file.exists():
            continue

        # 读 baseline 的 VLM caption
        cap_file = Path(args.baseline_dir) / f"sample_{sid}" / "vlm_caption.txt"
        if not cap_file.exists():
            logger.error(f"  找不到 baseline caption: {cap_file}")
            continue
        vlm_caption = cap_file.read_text(encoding="utf-8").strip()

        # LLM 改写
        hybrid_prompt = llm_rewrite(vlm_caption, args.llm_model)
        out_file.write_text(hybrid_prompt, encoding="utf-8")
        logger.info(f"  [{sid}] {vlm_caption[:40]}... → {hybrid_prompt[:40]}...")

    # ── Step 2-N: 迭代循环 ──
    all_iter_metrics = []

    for iteration in range(1, args.iter + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Iteration {iteration}/{args.iter}")
        logger.info(f"{'=' * 60}")

        # 当前轮的 caption 目录
        if iteration == 1:
            current_caption_dir = str(caption_dir_iter0)
        else:
            current_caption_dir = str(out_dir / f"captions_iter{iteration - 1}")

        # 当前轮的输出目录
        iter_output_dir = str(out_dir / f"gen_iter{iteration}")

        # ── 生成视频 ──
        logger.info(f"  [生成] 使用 caption: {current_caption_dir}")
        generate_videos(
            data_dir=args.data_dir,
            caption_dir=current_caption_dir,
            output_dir=iter_output_dir,
            sample_ids=args.sample_ids,
            args=args,
        )

        # ── 创建 flat 目录 ──
        flat_dir = make_flat_dir(iter_output_dir, args.sample_ids)

        # ── 评测 ──
        eval_output = str(out_dir / f"eval_iter{iteration}")
        metrics = run_eval(
            orig_dir=args.data_dir,
            gen_dir=flat_dir,
            caption_dir=current_caption_dir,
            output_dir=eval_output,
        )

        clip_score = metrics.get("orig_gen_clip_mean", 0)
        xclip_score = metrics.get("orig_gen_xclip_mean", 0)
        logger.info(f"  [评测] Iter {iteration}: CLIP={clip_score:.4f}, XCLIP={xclip_score:.4f}")

        all_iter_metrics.append({
            "iteration": iteration,
            "orig_gen_clip": clip_score,
            "orig_gen_xclip": xclip_score,
            "caption_dir": current_caption_dir,
            "gen_dir": iter_output_dir,
        })

        # ── VLM 对比 + LLM 修复（非最后一轮）──
        if iteration < args.iter:
            next_caption_dir = out_dir / f"captions_iter{iteration}"
            next_caption_dir.mkdir(exist_ok=True)

            if args.skip_vlm:
                # 跳过 VLM，直接用 LLM 自主改写（基于上一轮 prompt 微调）
                logger.info(f"  [LLM] 自主迭代改写（无 VLM 反馈）...")
                for sid in args.sample_ids:
                    cur_prompt = Path(current_caption_dir, f"{sid}.txt").read_text(encoding="utf-8").strip()
                    refined = llm_refine(cur_prompt, "Try to improve motion description and subject clarity.", args.llm_model)
                    (next_caption_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
            else:
                # VLM 对比 + LLM 修复
                logger.info(f"  [VLM+LLM] 对比分析 + 修复改写...")
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

                for sid in args.sample_ids:
                    ref_video = str(Path(args.data_dir) / f"{sid}.mp4")
                    gen_video = str(Path(iter_output_dir) / f"sample_{sid}" / f"{sid}.mp4")
                    cur_prompt = Path(current_caption_dir, f"{sid}.txt").read_text(encoding="utf-8").strip()

                    if not Path(gen_video).exists():
                        logger.warning(f"  [{sid}] 生成视频不存在，跳过")
                        (next_caption_dir / f"{sid}.txt").write_text(cur_prompt, encoding="utf-8")
                        continue

                    # VLM 对比
                    feedback = vlm_compare(ref_video, gen_video, vlm_client)
                    logger.info(f"  [{sid}] VLM: {feedback[:60]}...")

                    # LLM 修复
                    refined = llm_refine(cur_prompt, feedback, args.llm_model)
                    (next_caption_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
                    logger.info(f"  [{sid}] 修复: {refined[:50]}...")

    # ── 汇总输出 ──
    logger.info(f"\n{'=' * 60}")
    logger.info("汇总: 各轮次指标 vs Baseline")
    logger.info(f"{'=' * 60}")

    baseline_clip = baseline_metrics.get("orig_gen_clip_mean", 0)
    baseline_xclip = baseline_metrics.get("orig_gen_xclip_mean", 0)

    print(f"\n{'─' * 70}")
    print(f"{'Iter':<6} {'CLIP':>10} {'Δ CLIP':>10} {'XCLIP':>10} {'Δ XCLIP':>10}")
    print(f"{'─' * 70}")
    print(f"{'base':<6} {baseline_clip:>10.4f} {'—':>10} {baseline_xclip:>10.4f} {'—':>10}")

    for m in all_iter_metrics:
        d_clip = m["orig_gen_clip"] - baseline_clip
        d_xclip = m["orig_gen_xclip"] - baseline_xclip
        print(f"{'iter' + str(m['iteration']):<6} {m['orig_gen_clip']:>10.4f} {d_clip:>+10.4f} "
              f"{m['orig_gen_xclip']:>10.4f} {d_xclip:>+10.4f}")
    print(f"{'─' * 70}\n")

    # 保存汇总 JSON
    summary = {
        "baseline": {"orig_gen_clip": baseline_clip, "orig_gen_xclip": baseline_xclip},
        "iterations": all_iter_metrics,
        "config": {
            "iter": args.iter,
            "llm_model": args.llm_model,
            "vlm_provider": args.vlm_provider if not args.skip_vlm else "skipped",
            "sample_ids": args.sample_ids,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
