#!/usr/bin/env python3
"""
Hybrid Pipeline v8 — 双LLM改写+监督 + 自适应SVD跳过

修复三个问题：
  1. LLM 虚构运动方向 → 两次LLM：改写LLM + 监督LLM
  2. VLM 无法检测方向错误 → 监督LLM 替代 VLM 做运动校验
  3. 静态/弱运动样本不适合 SVD → 自适应跳过（temporal能量阈值）

流程（改写LLM → 监督LLM → VLM反馈）：
  1. 读取 VLM caption
  2. 改写LLM：约束式改写 → rewritten prompt
  3. 监督LLM：对比原始 caption vs 改写结果，检查运动是否被虚构/篡改
  4. 如果监督发现问题 → 改写LLM 根据监督反馈重写
  5. (可选) VLM 校验：仅检查非运动维度（subject/background/timing）
  6. LLM 修复：根据 VLM 反馈做非运动修正 → final prompt
  7. 自适应 SVD：计算 temporal 能量占比，低于阈值则跳过 SVD
  8. 生成视频 + 评测

用法:
    cd /root/xixihaha/P-Flow

    export DASHSCOPE_API_KEY="sk-xxxxx"

    # 完整流程（双LLM + VLM + 自适应SVD）
    python scripts/run_v8_dual_llm.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/v8 \
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \
        --alpha 0.001

    # 仅双LLM，跳过VLM
    python scripts/run_v8_dual_llm.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/v8_no_vlm \
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \
        --skip_vlm --alpha 0.001
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

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ═══════════════════════════════════════════════════════════════════════════════
# 系统提示词
# ═══════════════════════════════════════════════════════════════════════════════

# ── 改写LLM：约束式改写 ──
REWRITE_SYSTEM = """You are a text-to-video prompt optimizer. You restructure VLM captions into effective video generation prompts for the Wan2.1 model.

## YOUR ROLE
The input caption was written by a VLM that watched the video. You have NOT seen the video. Your job is to restructure the caption into a tighter, more effective generation prompt — staying extremely faithful to the input content.

## STRUCTURE RULES

1. SUBJECT-FIRST OPENING: Start with the main subject + action + key visual detail. Never start with "The video shows/depicts/features..."

2. NATURAL TEMPORAL FLOW: Weave temporal structure into motion descriptions using phrases like "initially... then...", "at first... before...", "gradually...". This should read as natural narration.

3. ONE VISUAL INFERENCE (maximum): You may add exactly ONE material/texture/lighting detail that is physically implied by the input (e.g., "wooden hulls" → "dark brown hulls"). This must be a surface-level attribute — NEVER an entire new object, person, animal, action, or weather phenomenon.

4. CAMERA & CLOSING: End with a brief camera note + a short closing phrase.

## MOTION FIDELITY (CRITICAL — highest priority rule)

Motion descriptions MUST be copied VERBATIM from the input. You must NOT:
- Infer or invent motion directions (e.g., "leftward", "clockwise", "right-to-left") unless explicitly stated in the input
- Change motion speed descriptions (e.g., "slowly" → "rapidly")
- Add motion trajectories not described in the input (e.g., adding "drifts leftward" when input only says "floating")
- Speculate about motion patterns (e.g., "circular motion", "zigzag path") not in the input
- Replace generic motion verbs with specific directional ones (e.g., "moves" → "drifts leftward")

You MAY only:
- Add temporal connectors (initially/then/finally) to EXISTING motion descriptions
- Restructure the ORDER of existing motion sentences for better flow
- Keep motion verbs as-is: if input says "moves", keep "moves" — do NOT change to "drifts/glides/sweeps"

If the input describes vague motion (e.g., "the boat moves"), keep it vague. Do NOT specify a direction.

## CONSTRAINTS

- MAXIMUM 1 inferred detail (appearance ONLY — never motion).
- NEVER add objects, animals, people, sounds, smells, or phenomena not explicitly stated in the input.
- NEVER change stated colors (intensifying is OK: "blue" → "deep blue").
- PRESERVE all nouns, counts, and attributes from the input.
- OUTPUT LENGTH: 100-150 words, 1-2 paragraphs. Be concise.
- Output ONLY the final prompt. No explanations."""


# ── 【新增】监督LLM：运动保真审核 ──
SUPERVISOR_SYSTEM = """You are a strict motion fidelity auditor. Your job is to compare an ORIGINAL caption with a REWRITTEN prompt, and detect any motion-related fabrications or alterations in the rewrite.

## YOUR TASK
You receive:
1. ORIGINAL: The VLM caption that describes real video content (ground truth)
2. REWRITTEN: A prompt optimizer's output that should faithfully preserve the original's motion

## WHAT TO CHECK (in priority order)

### 1. INVENTED DIRECTIONS (most critical)
Does the rewrite add specific directions NOT in the original?
- Original says "moves" → Rewrite says "drifts leftward" ❌ FABRICATED
- Original says "approaches each other" → Rewrite says "drifts slowly leftward" ❌ FABRICATED
- Original says "moves from left to right" → Rewrite says "moves from left to right" ✅ OK

### 2. INVENTED SPEED/ACCELERATION
Does the rewrite add speed changes NOT in the original?
- Original says "drives forward" → Rewrite says "accelerates slightly" ❌ FABRICATED
- Original says "runs steadily" → Rewrite says "sprints then accelerates" ❌ FABRICATED

### 3. INVENTED MOTION PATTERNS
Does the rewrite add trajectory patterns NOT in the original?
- Original says "floating" → Rewrite says "tilts minutely, catching wind" ❌ FABRICATED
- Original says "swimming" → Rewrite adds "undulates with gentle tail movements" ❌ if not in original

### 4. REVERSED DIRECTIONS
Does the rewrite flip stated directions?
- Original says "from left enters, moves to right" → Rewrite says "veers left" ❌ REVERSED

### 5. MOTION VERB UPGRADES
Does the rewrite replace neutral verbs with specific ones?
- "moves" → "drifts" ❌ (adds directional connotation)
- "moves" → "glides" ❌ (adds smoothness not stated)

## WHAT IS ACCEPTABLE (DO NOT flag)
- Adding temporal connectors: "initially... then..." ✅
- Reordering sentences ✅
- ONE material/texture inference (appearance only) ✅
- Keeping the same motion verb ✅

## OUTPUT FORMAT

If you find problems:
VERDICT: FAIL
ISSUES:
- [issue]: original says "[X]", rewrite says "[Y]" — [fabricated/reversed/upgraded]
CORRECTION_HINTS:
- [what should be used instead]

If no problems:
VERDICT: PASS

Be STRICT. False positives are less harmful than false negatives."""


# ── 【新增】改写LLM 重写（根据监督反馈修正）──
REWRITE_WITH_FEEDBACK_SYSTEM = """You are a text-to-video prompt optimizer doing a CORRECTIVE REWRITE. A supervisor found motion fabrication errors in your previous output. Fix them.

## CONTEXT
You will receive:
1. ORIGINAL CAPTION: The ground truth from a VLM
2. YOUR PREVIOUS OUTPUT: The prompt you wrote that has errors
3. SUPERVISOR FEEDBACK: Specific motion errors identified

## YOUR TASK
Rewrite the prompt, fixing ALL issues the supervisor identified:
- Replace fabricated directions with the original's wording (or omit direction if original is vague)
- Replace fabricated speed/acceleration with the original's wording
- Replace upgraded motion verbs back to the original's verbs
- Fix any reversed directions

## RULES
- Subject-first opening, natural temporal flow
- 100-150 words, 1-2 paragraphs
- Maximum 1 visual inference (appearance only, never motion)
- PRESERVE all nouns, counts, attributes
- End with camera note

## CRITICAL
- For each flagged issue, use EXACTLY the motion language from the ORIGINAL CAPTION
- If the original says "moves", write "moves" — not "drifts", "glides", or "sweeps"
- If the original doesn't specify direction, do NOT add one
- Output ONLY the fixed prompt. No explanations."""


# ── VLM 校验（v8 简化版：仅非运动维度）──
VLM_VERIFY_PROMPT_V8 = """You are watching a video (shown as key frames) and reading a text prompt.

Your task: Check visual accuracy ONLY. Do NOT evaluate motion direction/speed/trajectory (already verified).

Check these 3 dimensions:

1. SUBJECT: Correct species, color, size, count, appearance?
2. BACKGROUND: Correct background elements, colors, lighting?
3. TIMING: Correct sequence of events?

Format:
SUBJECT: [what's wrong, or "accurate"]
BACKGROUND: [what's wrong, or "accurate"]
TIMING: [what's wrong, or "accurate"]

Do NOT flag: motion direction, speed, camera descriptions, temporal connectors.
DO flag: wrong colors/count/species, non-existent elements, wrong action TYPE (not direction)."""


# ── REFINE（仅处理 VLM 非运动反馈）──
REFINE_SYSTEM = """You fix video generation prompts based on VLM feedback. Make SURGICAL fixes only.

Fix: subject appearance, background, timing issues.
Do NOT touch: motion descriptions (already verified and correct).

Rules:
- Copy prompt and make targeted edits only
- Output word count within ±15% of input
- Do NOT rephrase things VLM didn't flag
- Do NOT change motion verbs, directions, or speed
- Output ONLY the fixed prompt. No explanations."""


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_edit_ratio(text_a: str, text_b: str) -> float:
    """Token-level 编辑距离比率"""
    from difflib import SequenceMatcher
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    return 1.0 - SequenceMatcher(None, tokens_a, tokens_b).ratio()


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
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    if result.startswith("'") and result.endswith("'"):
        result = result[1:-1]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 核心步骤
# ═══════════════════════════════════════════════════════════════════════════════

def step_rewrite(caption: str, model: str = "qwen-plus",
                 max_retries: int = 2) -> str:
    """Step 2: 改写LLM 第一次改写"""
    word_count = len(caption.split())
    user_msg = (
        f"Rewrite this VLM caption ({word_count} words) into a video generation prompt.\n\n"
        f"RULES:\n"
        f"- Start with subject + action (no \"The video shows...\")\n"
        f"- Natural temporal flow (initially/then/gradually)\n"
        f"- Maximum 1 inferred detail (material/texture/lighting only)\n"
        f"- NEVER invent motion directions, speeds, or patterns\n"
        f"- PRESERVE all motion verbs exactly as stated\n"
        f"- Target: 100-150 words\n\n"
        f"INPUT:\n{caption}\n\nOUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.7 if attempt == 0 else max(0.5, 0.7 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_SYSTEM, model, temperature=temp)

        result_words = len(result.split())
        if result_words < 80 or result_words > 160:
            logger.warning(f"  [改写 重试{attempt+1}] 长度异常: {result_words} 词")
            continue
        if result.lower().startswith(("the video", "this video", "in this video")):
            logger.warning(f"  [改写 重试{attempt+1}] preamble 开头")
            continue
        return result

    logger.warning(f"  改写所有重试未通过")
    return result


def step_supervise(original_caption: str, rewritten_prompt: str,
                   model: str = "qwen-plus") -> dict:
    """Step 3: 监督LLM 审核运动保真度"""
    user_msg = (
        f"## ORIGINAL CAPTION (ground truth):\n{original_caption}\n\n"
        f"## REWRITTEN PROMPT (to audit):\n{rewritten_prompt}\n\n"
        f"Compare motion descriptions. Report fabricated, reversed, or upgraded motion."
    )

    result = call_llm(user_msg, SUPERVISOR_SYSTEM, model, temperature=0.3)

    passed = "VERDICT: PASS" in result.upper()
    issues = []
    hints = []

    if not passed:
        in_issues = False
        in_hints = False
        for line in result.split("\n"):
            line = line.strip()
            if "ISSUES" in line.upper():
                in_issues, in_hints = True, False
                continue
            if "CORRECTION" in line.upper() and "HINT" in line.upper():
                in_hints, in_issues = True, False
                continue
            if line.startswith("- ") and in_issues:
                issues.append(line[2:])
            elif line.startswith("- ") and in_hints:
                hints.append(line[2:])

    return {"passed": passed, "feedback": result, "issues": issues, "hints": hints}


def step_rewrite_with_feedback(original_caption: str, previous_output: str,
                                supervisor_feedback: str,
                                model: str = "qwen-plus",
                                max_retries: int = 2) -> str:
    """Step 4: 改写LLM 根据监督反馈重写"""
    user_msg = (
        f"## ORIGINAL CAPTION:\n{original_caption}\n\n"
        f"## YOUR PREVIOUS OUTPUT (has motion errors):\n{previous_output}\n\n"
        f"## SUPERVISOR FEEDBACK:\n{supervisor_feedback}\n\n"
        f"Rewrite fixing ALL motion issues. Use EXACTLY the motion language from ORIGINAL.\n\nOUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.5 if attempt == 0 else max(0.3, 0.5 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_WITH_FEEDBACK_SYSTEM, model, temperature=temp)

        result_words = len(result.split())
        if result_words < 80 or result_words > 160:
            logger.warning(f"  [重写 重试{attempt+1}] 长度异常: {result_words} 词")
            continue
        return result

    logger.warning(f"  重写所有重试未通过")
    return result


def step_vlm_verify(video_path: str, prompt: str, vlm_client) -> str:
    """Step 5: VLM 校验（仅非运动维度）"""
    try:
        num_frames = 16 if getattr(vlm_client, 'use_video_mode', True) else 8
        frames_pil = vlm_client._extract_frames_pil(video_path, num_frames=num_frames)
        if not frames_pil:
            return "accurate"

        content_list = []
        for img in frames_pil:
            content_list.append({"type": "image", "image": img})

        content_list.append({
            "type": "text",
            "text": f'{VLM_VERIFY_PROMPT_V8}\n\n## Text prompt to verify:\n"{prompt}"'
        })

        messages = [{"role": "user", "content": content_list}]
        response_text = vlm_client._generate(messages)
        if response_text and len(response_text.strip()) > 10:
            return response_text.strip()
    except Exception as e:
        logger.warning(f"  VLM 校验失败: {e}")

    return "accurate"


def step_refine(current_prompt: str, vlm_feedback: str, model: str = "qwen-plus",
                max_retries: int = 2) -> str:
    """Step 6: LLM 修复（仅非运动反馈）"""
    word_count = len(current_prompt.split())
    user_msg = (
        f"## Current Prompt:\n{current_prompt}\n\n"
        f"## VLM Feedback (non-motion issues):\n{vlm_feedback}\n\n"
        f"Fix ONLY the flagged issues. Do NOT change motion. Output ONLY the fixed prompt:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.4 if attempt == 0 else max(0.2, 0.4 - attempt * 0.1)
        result = call_llm(user_msg, REFINE_SYSTEM, model, temperature=temp)

        result_words = len(result.split())
        if result_words / max(word_count, 1) < 0.70:
            continue
        if _compute_edit_ratio(current_prompt, result) > 0.35:
            continue
        return result

    return result


def has_real_issues(vlm_feedback: str) -> bool:
    """判断 VLM 反馈是否有实质性问题"""
    if not vlm_feedback or vlm_feedback == "accurate":
        return False
    for line in vlm_feedback.lower().split("\n"):
        line = line.strip()
        if ":" in line:
            value = line.split(":", 1)[1].strip().strip("[] ")
            if value and value != "accurate":
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 自适应 SVD 跳过
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_energy_ratio(noise_inv_path: str, rho_s: float = 0.1) -> float:
    """
    计算 SVD 滤波后 temporal 成分的能量占比。
    
    值越高表示运动信号越强，SVD 越有效。
    对于静态/弱运动视频，此值很低（<1%），应跳过 SVD。
    """
    import torch
    from src.svd_filter import SVDFilter

    if not Path(noise_inv_path).exists():
        logger.warning(f"  反演噪声不存在: {noise_inv_path}")
        return 0.0

    noise_inv = torch.load(noise_inv_path, map_location="cpu")
    if noise_inv.dim() == 5:
        noise_inv = noise_inv[0]

    original_energy = (noise_inv.float() ** 2).sum().item()
    if original_energy == 0:
        return 0.0

    svd_filter = SVDFilter(rho_s=rho_s, rho_m=0.9)
    filtered = svd_filter._filter_single(noise_inv.float())
    filtered_energy = (filtered ** 2).sum().item()

    return filtered_energy / original_energy


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════════════════

def make_flat_dir(output_dir: str, sample_ids: list) -> str:
    """创建 flat 目录供评测使用"""
    flat_dir = Path(output_dir) / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        src = Path(output_dir) / f"sample_{sid}" / f"{sid}.mp4"
        dst = flat_dir / f"{sid}.mp4"
        if src.exists():
            dst.unlink(missing_ok=True)
            os.symlink(src.resolve(), dst)
    return str(flat_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="v8 — 双LLM改写+监督 + 自适应SVD")

    # I/O
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--caption_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--sample_ids", type=int, nargs="+", required=True)

    # 生成参数
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.001,
                   help="SVD 噪声先验混合权重 (默认 0.001)")

    # LLM
    p.add_argument("--llm_model", type=str, default="qwen-plus")

    # VLM
    p.add_argument("--skip_vlm", action="store_true",
                   help="跳过 VLM 校验（仅双LLM机制）")
    p.add_argument("--vlm_provider", type=str, default="local")
    p.add_argument("--vlm_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct")

    # 自适应 SVD
    p.add_argument("--temporal_energy_threshold", type=float, default=0.01,
                   help="Temporal 能量占比阈值，低于此值跳过 SVD (默认 0.01)")
    p.add_argument("--noise_inv_dir", type=str, default="",
                   help="反演噪声缓存目录 (含 {sid}_noise_inv.pt 文件)")
    p.add_argument("--skip_adaptive", action="store_true",
                   help="禁用自适应SVD跳过，对所有样本使用SVD")

    # 控制
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("需要设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: 读取 VLM caption
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1: 读取 VLM caption")
    logger.info("=" * 60)

    caption_dir = Path(args.caption_dir)
    captions = {}
    for sid in args.sample_ids:
        cap_file = caption_dir / f"{sid}.txt"
        if not cap_file.exists():
            logger.error(f"  找不到 caption: {cap_file}")
            continue
        captions[sid] = cap_file.read_text(encoding="utf-8").strip()
        logger.info(f"  [{sid}] caption ({len(captions[sid].split())} 词)")

    if not captions:
        logger.error("没有找到任何 caption，退出")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: 改写LLM 第一次改写
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 2: 改写LLM — 第一次改写")
    logger.info("=" * 60)

    rewrite_dir = out_dir / "captions_rewritten"
    rewrite_dir.mkdir(exist_ok=True)

    rewritten = {}
    for sid, caption in captions.items():
        out_file = rewrite_dir / f"{sid}.txt"
        if args.resume and out_file.exists():
            rewritten[sid] = out_file.read_text(encoding="utf-8").strip()
            logger.info(f"  [{sid}] (resume)")
            continue

        result = step_rewrite(caption, args.llm_model)
        rewritten[sid] = result
        out_file.write_text(result, encoding="utf-8")
        logger.info(f"  [{sid}] 改写完成: {result[:60]}...")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: 监督LLM — 运动保真审核
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 3: 监督LLM — 运动保真审核")
    logger.info("=" * 60)

    supervisor_dir = out_dir / "supervisor_feedback"
    supervisor_dir.mkdir(exist_ok=True)

    supervised = {}  # sid -> final prompt after supervision
    supervisor_results = {}  # sid -> {passed, issues, ...}

    for sid, prompt in rewritten.items():
        original = captions[sid]

        # 监督审核
        logger.info(f"  [{sid}] 监督LLM审核中...")
        sv_result = step_supervise(original, prompt, args.llm_model)
        supervisor_results[sid] = sv_result

        # 保存监督反馈
        (supervisor_dir / f"{sid}.txt").write_text(sv_result["feedback"], encoding="utf-8")

        if sv_result["passed"]:
            logger.info(f"  [{sid}] ✅ 监督通过 — 无运动虚构")
            supervised[sid] = prompt
        else:
            logger.info(f"  [{sid}] ❌ 监督发现问题: {len(sv_result['issues'])} 个")
            for issue in sv_result["issues"]:
                logger.info(f"      - {issue[:80]}")

            # ── Step 4: 改写LLM 根据监督反馈重写 ──
            logger.info(f"  [{sid}] 改写LLM 根据监督反馈重写...")
            corrected = step_rewrite_with_feedback(
                original, prompt, sv_result["feedback"], args.llm_model
            )
            supervised[sid] = corrected
            logger.info(f"  [{sid}] 重写完成: {corrected[:60]}...")

            # 二次监督验证（可选，确保修正有效）
            sv_check = step_supervise(original, corrected, args.llm_model)
            if not sv_check["passed"]:
                logger.warning(f"  [{sid}] ⚠️ 二次监督仍发现问题，但继续使用")
                (supervisor_dir / f"{sid}_round2.txt").write_text(
                    sv_check["feedback"], encoding="utf-8"
                )
            else:
                logger.info(f"  [{sid}] ✅ 二次监督通过")

    # 保存监督后的 prompt
    supervised_dir = out_dir / "captions_supervised"
    supervised_dir.mkdir(exist_ok=True)
    for sid, prompt in supervised.items():
        (supervised_dir / f"{sid}.txt").write_text(prompt, encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: VLM 校验（仅非运动维度）
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 5: VLM 校验（仅非运动维度）")
    logger.info("=" * 60)

    final_dir = out_dir / "captions_final"
    final_dir.mkdir(exist_ok=True)

    if args.skip_vlm:
        logger.info("  --skip_vlm: 跳过 VLM，直接使用监督后结果")
        for sid, prompt in supervised.items():
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

        for sid, prompt in supervised.items():
            video_path = str(Path(args.data_dir) / f"{sid}.mp4")
            if not Path(video_path).exists():
                logger.warning(f"  [{sid}] 视频不存在，跳过")
                (final_dir / f"{sid}.txt").write_text(prompt, encoding="utf-8")
                continue

            logger.info(f"  [{sid}] VLM 校验...")
            feedback = step_vlm_verify(video_path, prompt, vlm_client)
            (vlm_feedback_dir / f"{sid}.txt").write_text(feedback, encoding="utf-8")

            if has_real_issues(feedback):
                logger.info(f"  [{sid}] VLM发现非运动问题，LLM修复...")
                refined = step_refine(prompt, feedback, args.llm_model)
                (final_dir / f"{sid}.txt").write_text(refined, encoding="utf-8")
            else:
                logger.info(f"  [{sid}] VLM通过")
                (final_dir / f"{sid}.txt").write_text(prompt, encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: 自适应 SVD 跳过判断
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 6: 自适应 SVD 跳过判断")
    logger.info("=" * 60)

    svd_decisions = {"use": [], "skip": [], "energy_ratios": {}}

    if args.skip_adaptive:
        logger.info("  --skip_adaptive: 禁用自适应，所有样本使用 SVD")
        svd_decisions["use"] = list(supervised.keys())
    elif not args.noise_inv_dir:
        logger.info("  未指定 --noise_inv_dir，所有样本使用 SVD（需要反演噪声文件判断）")
        logger.info("  提示: 指定 --noise_inv_dir 可启用自适应跳过")
        svd_decisions["use"] = list(supervised.keys())
    else:
        noise_dir = Path(args.noise_inv_dir)
        for sid in supervised.keys():
            # 尝试多种可能的文件名
            candidates = [
                noise_dir / f"{sid}_noise_inv.pt",
                noise_dir / f"sample_{sid}" / "noise_inv.pt",
                noise_dir / f"{sid}.pt",
            ]
            noise_path = None
            for c in candidates:
                if c.exists():
                    noise_path = str(c)
                    break

            if noise_path is None:
                logger.warning(f"  [{sid}] 反演噪声不存在，默认使用 SVD")
                svd_decisions["use"].append(sid)
                continue

            ratio = compute_temporal_energy_ratio(noise_path, rho_s=0.1)
            svd_decisions["energy_ratios"][sid] = ratio

            if ratio < args.temporal_energy_threshold:
                logger.info(f"  [{sid}] temporal能量={ratio:.4f} < {args.temporal_energy_threshold} → 跳过 SVD ⏭️")
                svd_decisions["skip"].append(sid)
            else:
                logger.info(f"  [{sid}] temporal能量={ratio:.4f} ≥ {args.temporal_energy_threshold} → 使用 SVD ✅")
                svd_decisions["use"].append(sid)

    logger.info(f"  汇总: 使用SVD={svd_decisions['use']}, 跳过SVD={svd_decisions['skip']}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7: 生成视频
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 7: 生成视频")
    logger.info("=" * 60)

    gen_dir = str((out_dir / "generated").resolve())
    final_caption_dir = str(final_dir.resolve())
    data_dir_abs = str(Path(args.data_dir).resolve())

    # 7a: 有 SVD 的样本
    svd_ids = sorted(svd_decisions["use"])
    if svd_ids and args.alpha > 0:
        logger.info(f"  7a: SVD生成 (α={args.alpha}): {svd_ids}")
        cmd = [
            sys.executable, str(PROJECT_ROOT / "run.py"),
            "--data_dir", data_dir_abs,
            "--caption_dir", final_caption_dir,
            "--output_dir", gen_dir,
            "--sample_ids", *[str(s) for s in svd_ids],
            "--noise_prior",
            "--alpha", str(args.alpha),
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
            "--seed", str(args.seed),
            "--resume",
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # 7b: 跳过 SVD 的样本（纯随机噪声）
    skip_ids = sorted(svd_decisions["skip"])
    if skip_ids:
        logger.info(f"  7b: 无SVD生成 (纯随机): {skip_ids}")
        cmd = [
            sys.executable, str(PROJECT_ROOT / "run.py"),
            "--data_dir", data_dir_abs,
            "--caption_dir", final_caption_dir,
            "--output_dir", gen_dir,
            "--sample_ids", *[str(s) for s in skip_ids],
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
            "--seed", str(args.seed),
            "--resume",
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # ══════════════════════════════════════════════════════════════════════════
    # Step 8: 评测
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 8: 评测 CLIP/XCLIP")
    logger.info("=" * 60)

    all_ids = sorted(supervised.keys())
    flat_dir = make_flat_dir(gen_dir, all_ids)
    eval_output = str((out_dir / "eval_results").resolve())

    cmd = [
        sys.executable, str(PROJECT_ROOT / "evaluation" / "run_clip_xclip_eval.py"),
        "--orig-dir", data_dir_abs,
        "--gen-dir", flat_dir,
        "--caption-dir", final_caption_dir,
        "--output-dir", eval_output,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # ── 汇总 ──
    logger.info(f"\n{'=' * 60}")
    logger.info("汇总")
    logger.info("=" * 60)

    # 统计监督效果
    n_total = len(supervisor_results)
    n_failed = sum(1 for r in supervisor_results.values() if not r["passed"])
    logger.info(f"  监督LLM: {n_failed}/{n_total} 个样本发现运动虚构并修正")
    logger.info(f"  SVD决策: {len(svd_ids)} 使用 SVD, {len(skip_ids)} 跳过 SVD")

    summary = {
        "version": "v8_dual_llm_adaptive_svd",
        "pipeline": "改写LLM → 监督LLM → VLM校验(非运动) → 自适应SVD → 生成",
        "fixes": {
            "fix1_motion_fabrication": f"监督LLM检出 {n_failed}/{n_total} 样本并修正",
            "fix2_vlm_limitation": "VLM 仅负责非运动维度，运动由监督LLM把关",
            "fix3_weak_motion_svd": f"自适应跳过: {skip_ids} (threshold={args.temporal_energy_threshold})",
        },
        "svd_decisions": {
            "use_svd": svd_ids,
            "skip_svd": skip_ids,
            "energy_ratios": svd_decisions["energy_ratios"],
        },
        "supervisor_summary": {
            sid: {"passed": r["passed"], "issues": r["issues"]}
            for sid, r in supervisor_results.items()
        },
        "config": {
            "alpha": args.alpha,
            "temporal_energy_threshold": args.temporal_energy_threshold,
            "llm_model": args.llm_model,
            "vlm_skipped": args.skip_vlm,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"  汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
