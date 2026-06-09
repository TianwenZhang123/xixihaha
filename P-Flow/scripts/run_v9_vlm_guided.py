#!/usr/bin/env python3
"""
Hybrid Pipeline v9 - VLM前置引导 + 监督LLM + 条件性SVD

核心创新：让VLM从"事后纠错"变成"事前补充信息"
  - LLM不再盲猜细节，而是主动向VLM提问
  - VLM看视频帧后回答LLM的疑问（纯视觉事实）
  - LLM结合VLM的回答进行二次改写（有视觉依据的丰富化）
  - 监督LLM确保改写不超越信息源（原始caption + VLM回答）

流程：
  1. 读取 VLM caption
  2. 改写LLM 第一次改写 + 生成疑问清单（对细节提问）
  3. VLM 看视频回答疑问清单（纯事实回答）
  4. 改写LLM 第二次改写（结合 VLM 回答丰富 prompt）
  5. 监督LLM：对比「原始caption + VLM回答」vs「最终改写」
  6. 条件性 SVD：根据 VLM 对运动的回答决定是否使用 SVD
  7. 生成视频 + 评测

用法:
    cd /root/xixihaha/P-Flow
    export DASHSCOPE_API_KEY="sk-xxxxx"

    # 完整流程
    python scripts/run_v9_vlm_guided.py \\
        --data_dir data/videos \\
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \\
        --output_dir outputs/v9 \\
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \\
        --alpha 0.002

    # 跳过SVD（仅测prompt质量）
    python scripts/run_v9_vlm_guided.py \\
        --data_dir data/videos \\
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \\
        --output_dir outputs/v9_L1 \\
        --sample_ids 7 17 21 31 32 33 34 43 46 47 \\
        --alpha 0
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


# ===========================================================================
# 系统提示词
# ===========================================================================

REWRITE_AND_ASK_SYSTEM = """You are a text-to-video prompt optimizer. You restructure VLM captions into effective video generation prompts for the Wan2.1 model.

## YOUR ROLE
The input caption was written by a VLM that watched the video. You have NOT seen the video. Your job is:
1. Restructure the caption into a video generation prompt (DRAFT)
2. Identify visual details you are UNCERTAIN about and ask questions

## DRAFT RULES

1. SUBJECT-FIRST OPENING: Start with the main subject + action + key visual detail. Never start with "The video shows/depicts/features..."

2. NATURAL TEMPORAL FLOW: Weave temporal structure using "initially... then...", "at first... before...", "gradually...". This is natural narration scaffolding.

3. MOTION FIDELITY (CRITICAL): Copy motion descriptions VERBATIM from the input. You must NOT:
   - Infer or invent motion directions unless explicitly stated
   - Change motion speed descriptions
   - Add motion trajectories not in the input
   - Replace neutral motion verbs with specific ones ("moves" to "drifts" is WRONG)

4. MARK UNCERTAINTY: For visual details you would like to add but are unsure about, mark them with [?] in your draft. These become your questions.

5. CAMERA & CLOSING: End with a camera note + short closing phrase. Mark camera details with [?] if not stated in input.

## CONSTRAINTS
- Do NOT add objects, animals, people not in the input
- PRESERVE all nouns, counts, and attributes from the input
- Draft length: 100-150 words, 1-2 paragraphs

## OUTPUT FORMAT (strict - two sections separated by exact delimiter)

<DRAFT>
[Your draft prompt here, with [?] marks on uncertain details]
</DRAFT>

<QUESTIONS>
1. [Question about a specific visual detail you are uncertain about]
2. [Another question...]
3. [Up to 5 questions maximum]
</QUESTIONS>

Questions should ask about:
- Material/texture of objects (e.g., "Are the boat hulls wooden or plastic?")
- Specific colors not stated (e.g., "What color is the road surface?")
- Spatial layout (e.g., "Is the subject on the left or right of frame?")
- Camera angle/movement (e.g., "Is the camera static or tracking?")
- Lighting details (e.g., "Is the lighting warm/golden or cool/blue?")
- Motion direction if vague (e.g., "Which direction does the car move - left-to-right or right-to-left?")

Do NOT ask about:
- Things already clearly stated in the input
- Abstract concepts or interpretations
- Things that would not help video generation"""


VLM_ANSWER_PROMPT_TEMPLATE = """You are watching key frames from a video. A prompt writer has questions about specific visual details they cannot determine from the text description alone.

Answer each question based ONLY on what you can observe in these frames. Be factual and concise.

## RULES
- Answer ONLY based on what is visually observable
- If you cannot determine the answer from the frames, say "Cannot determine"
- Keep each answer to 1-2 sentences
- For motion direction questions: describe the apparent movement direction you observe across the frame sequence (early frames to late frames)
- For camera questions: infer from how the scene perspective changes across frames

## QUESTIONS:
{questions}

## OUTPUT FORMAT
Answer each question with the same numbering:
1. [Your answer]
2. [Your answer]
..."""


REWRITE_WITH_ANSWERS_SYSTEM = """You are a text-to-video prompt optimizer doing a REFINED REWRITE. You now have visual answers from a VLM that watched the actual video. Use these to enrich your prompt with VERIFIED visual details.

## CONTEXT
You will receive:
1. ORIGINAL CAPTION: The ground truth text description
2. YOUR DRAFT: Your initial rewrite (with [?] uncertainty marks)
3. VLM ANSWERS: Factual visual observations that resolve your uncertainties

## YOUR TASK
Produce a FINAL prompt that:
- Incorporates VLM-verified details (replacing [?] marks with confirmed facts)
- Maintains subject-first opening and natural temporal flow
- Uses temporal connectors (initially/then/gradually) for narrative structure
- Includes camera description IF the VLM confirmed camera behavior
- PRESERVES all motion verbs EXACTLY as stated in the original (highest priority)
- If VLM confirmed a motion direction, you MAY include it (it is now verified, not fabricated)

## WHAT YOU MAY ADD (from VLM answers only)
- Confirmed materials/textures
- Confirmed colors
- Confirmed spatial positions
- Confirmed camera angle/movement
- Confirmed motion directions (VLM-observed)
- Confirmed lighting conditions

## WHAT YOU MUST NOT DO
- Invent details the VLM did not confirm
- Change motion verbs from the original
- Add objects/subjects not in the original or VLM answers
- Exceed 150 words

## OUTPUT
Output ONLY the final refined prompt. No explanations, no headers."""


SUPERVISOR_V9_SYSTEM = """You are a strict factual auditor. Your job is to verify that a REWRITTEN prompt does not contain information beyond its two allowed sources:
1. ORIGINAL CAPTION (the VLM text description of the video)
2. VLM ANSWERS (factual visual observations that answered specific questions)

## YOUR TASK
Check whether the REWRITTEN prompt introduces ANY claims not supported by either source.

## WHAT TO CHECK (priority order)

### 1. MOTION FABRICATION (highest priority)
Does the rewrite contain motion directions, speeds, or patterns NOT in either source?
- If original says "moves" and VLM answer says "moves left-to-right" then rewrite can say "moves left-to-right" (OK)
- If original says "moves" and VLM says nothing about direction then rewrite says "drifts leftward" (FABRICATED)
- If original says "swims" then rewrite says "glides" (VERB UPGRADE)

### 2. UNSUPPORTED DETAILS
Does the rewrite add visual details (color, material, texture) that appear in NEITHER the original NOR the VLM answers?
- VLM confirmed "wooden hulls" then rewrite says "wooden hulls" (OK)
- Neither source mentions "mahogany" then rewrite says "mahogany hulls" (FABRICATED)

### 3. INVENTED CAMERA
Does the rewrite add camera descriptions not confirmed by VLM answers?
- VLM says "camera is static" then rewrite says "static shot" (OK)
- VLM says "Cannot determine" then rewrite says "tracking shot" (FABRICATED)

### 4. TEMPORAL CONNECTORS (ACCEPTABLE - do NOT flag)
- Adding "initially... then... gradually..." to existing descriptions (OK)
- Reordering sentences for flow (OK)
- These are structural improvements, NOT fabrication

## OUTPUT FORMAT

If problems found:
VERDICT: FAIL

ISSUES:
- [specific issue with quote from rewrite and explanation of why it is unsupported]

CORRECTION_HINTS:
- [what should be used instead, citing the source]

If no problems:
VERDICT: PASS

Be STRICT on motion and unsupported details. Be LENIENT on temporal connectors and sentence restructuring."""


REWRITE_CORRECTION_SYSTEM = """You fix video generation prompts based on auditor feedback. The auditor found claims in your prompt that exceed your allowed information sources.

## CONTEXT
You will receive:
1. ORIGINAL CAPTION: Ground truth text
2. VLM ANSWERS: Verified visual facts
3. YOUR PROMPT: The version that has issues
4. AUDITOR FEEDBACK: Specific problems identified

## YOUR TASK
Fix ALL flagged issues by:
- Replacing fabricated details with wording from original caption or VLM answers
- Removing unsupported claims entirely if no source confirms them
- Keeping motion verbs EXACTLY as they appear in the original
- Using VLM-confirmed motion directions ONLY if the VLM explicitly stated them

## RULES
- Subject-first opening, natural temporal flow
- 100-150 words
- Temporal connectors (initially/then/gradually) are FINE to keep
- VLM-confirmed details are FINE to keep
- Output ONLY the fixed prompt. No explanations."""


# ===========================================================================
# 工具函数
# ===========================================================================

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


def _compute_edit_ratio(text_a: str, text_b: str) -> float:
    """Token-level edit distance ratio"""
    from difflib import SequenceMatcher
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    return 1.0 - SequenceMatcher(None, tokens_a, tokens_b).ratio()


# ===========================================================================
# 核心步骤
# ===========================================================================

def step_rewrite_and_ask(caption: str, model: str = "qwen-plus",
                         max_retries: int = 2) -> dict:
    """
    Step 2: 改写LLM 第一次改写 + 生成疑问清单

    Returns: {"draft": str, "questions": list[str], "raw": str}
    """
    word_count = len(caption.split())
    user_msg = (
        f"Rewrite this VLM caption ({word_count} words) into a video generation prompt draft, "
        f"and generate questions about uncertain visual details.\n\n"
        f"INPUT CAPTION:\n{caption}\n\n"
        f"OUTPUT (use exact <DRAFT> and <QUESTIONS> format):"
    )

    for attempt in range(max_retries + 1):
        temp = 0.7 if attempt == 0 else max(0.5, 0.7 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_AND_ASK_SYSTEM, model, temperature=temp)

        # Parse DRAFT
        draft = ""
        questions = []

        if "<DRAFT>" in result and "</DRAFT>" in result:
            draft = result.split("<DRAFT>")[1].split("</DRAFT>")[0].strip()
        elif "DRAFT" in result:
            parts = result.split("QUESTIONS")
            draft = parts[0].replace("DRAFT", "").replace("<", "").replace(">", "").strip()

        if "<QUESTIONS>" in result and "</QUESTIONS>" in result:
            q_text = result.split("<QUESTIONS>")[1].split("</QUESTIONS>")[0].strip()
        elif "QUESTIONS" in result:
            parts = result.split("QUESTIONS")
            q_text = parts[-1].replace("<", "").replace(">", "").replace("/", "").strip()
        else:
            q_text = ""

        # Parse question list
        for line in q_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                q = line.lstrip("0123456789.-) ").strip()
                if q and len(q) > 10:
                    questions.append(q)

        # Validate
        if draft and len(draft.split()) >= 60:
            return {"draft": draft, "questions": questions[:5], "raw": result}

        logger.warning(f"  [rewrite+ask retry {attempt+1}] parse failed or draft too short")

    return {"draft": result, "questions": [], "raw": result}


def step_vlm_answer(video_path: str, questions: list, vlm_client) -> dict:
    """
    Step 3: VLM 看视频回答疑问

    Returns: {"answers": list[str], "raw": str, "has_motion_direction": bool,
              "motion_description": str}
    """
    if not questions:
        return {"answers": [], "raw": "", "has_motion_direction": False,
                "motion_description": ""}

    q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt_text = VLM_ANSWER_PROMPT_TEMPLATE.format(questions=q_text)

    try:
        num_frames = 16
        frames_pil = vlm_client._extract_frames_pil(video_path, num_frames=num_frames)
        if not frames_pil:
            return {"answers": ["Cannot determine"] * len(questions), "raw": "",
                    "has_motion_direction": False, "motion_description": ""}

        content_list = []
        for img in frames_pil:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content_list}]
        response_text = vlm_client._generate(messages)

        # Parse answers
        answers = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                ans = line.lstrip("0123456789.-) ").strip()
                if ans:
                    answers.append(ans)

        while len(answers) < len(questions):
            answers.append("Cannot determine")

        # Detect motion direction info
        has_motion_direction = False
        motion_description = ""
        motion_keywords = ["left-to-right", "right-to-left", "leftward", "rightward",
                           "upward", "downward", "forward", "toward the camera",
                           "away from camera", "clockwise", "counter-clockwise",
                           "approaching", "receding", "from left", "from right",
                           "moves to the left", "moves to the right"]

        full_text = response_text.lower()
        for kw in motion_keywords:
            if kw in full_text:
                has_motion_direction = True
                motion_description = kw
                break

        return {
            "answers": answers[:len(questions)],
            "raw": response_text,
            "has_motion_direction": has_motion_direction,
            "motion_description": motion_description,
        }

    except Exception as e:
        logger.warning(f"  VLM answer failed: {e}")
        return {"answers": ["Cannot determine"] * len(questions), "raw": str(e),
                "has_motion_direction": False, "motion_description": ""}


def step_rewrite_with_answers(caption: str, draft: str, questions: list,
                               answers: list, model: str = "qwen-plus",
                               max_retries: int = 2) -> str:
    """
    Step 4: 改写LLM 结合VLM回答做二次改写
    """
    qa_text = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        qa_text += f"{i+1}. Q: {q}\n   A: {a}\n"

    if not qa_text:
        qa_text = "(No questions were asked / no VLM answers available)"

    user_msg = (
        f"## ORIGINAL CAPTION:\n{caption}\n\n"
        f"## YOUR DRAFT (with uncertainty marks):\n{draft}\n\n"
        f"## VLM ANSWERS (verified visual facts):\n{qa_text}\n\n"
        f"Produce the FINAL refined prompt incorporating VLM-verified details.\n\nOUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.6 if attempt == 0 else max(0.4, 0.6 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_WITH_ANSWERS_SYSTEM, model, temperature=temp)

        result_words = len(result.split())
        if result_words < 80 or result_words > 170:
            logger.warning(f"  [refine retry {attempt+1}] word count: {result_words}")
            continue
        if result.lower().startswith(("the video", "this video", "in this video")):
            logger.warning(f"  [refine retry {attempt+1}] preamble opening")
            continue
        return result

    logger.warning("  refine all retries failed")
    return result


def step_supervise_v9(original_caption: str, vlm_answers_text: str,
                      rewritten_prompt: str, model: str = "qwen-plus") -> dict:
    """
    Step 5: 监督LLM - 对比信息源（原始caption + VLM回答）vs 最终改写
    """
    user_msg = (
        f"## SOURCE 1 - ORIGINAL CAPTION (ground truth):\n{original_caption}\n\n"
        f"## SOURCE 2 - VLM ANSWERS (verified visual facts):\n{vlm_answers_text}\n\n"
        f"## REWRITTEN PROMPT (to audit):\n{rewritten_prompt}\n\n"
        f"Check whether the rewritten prompt exceeds these two sources. "
        f"Temporal connectors and sentence restructuring are ACCEPTABLE."
    )

    result = call_llm(user_msg, SUPERVISOR_V9_SYSTEM, model, temperature=0.3)

    passed = "VERDICT: PASS" in result.upper()
    issues = []
    hints = []

    if not passed:
        in_issues = False
        in_hints = False
        for line in result.split("\n"):
            line = line.strip()
            if "ISSUES" in line.upper() and ":" in line:
                in_issues, in_hints = True, False
                continue
            if "CORRECTION" in line.upper() and "HINT" in line.upper():
                in_hints, in_issues = True, False
                continue
            if "VERDICT" in line.upper():
                in_issues, in_hints = False, False
                continue
            if line.startswith("- ") and in_issues:
                issues.append(line[2:])
            elif line.startswith("- ") and in_hints:
                hints.append(line[2:])

    return {"passed": passed, "feedback": result, "issues": issues, "hints": hints}


def step_correct(original_caption: str, vlm_answers_text: str,
                 current_prompt: str, supervisor_feedback: str,
                 model: str = "qwen-plus", max_retries: int = 2) -> str:
    """
    Step 5b: 监督发现问题后的修正改写
    """
    user_msg = (
        f"## ORIGINAL CAPTION:\n{original_caption}\n\n"
        f"## VLM ANSWERS (verified facts you may use):\n{vlm_answers_text}\n\n"
        f"## YOUR PROMPT (has issues):\n{current_prompt}\n\n"
        f"## AUDITOR FEEDBACK:\n{supervisor_feedback}\n\n"
        f"Fix ALL flagged issues using ONLY information from the original caption and VLM answers.\n\nOUTPUT:"
    )

    for attempt in range(max_retries + 1):
        temp = 0.4 if attempt == 0 else max(0.2, 0.4 - attempt * 0.1)
        result = call_llm(user_msg, REWRITE_CORRECTION_SYSTEM, model, temperature=temp)

        result_words = len(result.split())
        if result_words < 80 or result_words > 170:
            continue
        return result

    return result


# ===========================================================================
# SVD 条件判断
# ===========================================================================

def extract_motion_clarity(vlm_answers_text: str, original_caption: str) -> str:
    """
    从 VLM 回答和原始 caption 中判断运动方向的清晰程度。

    Returns: "clear" / "moderate" / "vague"
      - clear: VLM 或原始 caption 明确指出了运动方向
      - moderate: 有运动描述但方向不明确
      - vague: 基本静态或极弱运动
    """
    direction_keywords = [
        "left to right", "right to left", "left-to-right", "right-to-left",
        "moves left", "moves right", "moves forward", "moves backward",
        "from left", "from right", "toward", "away from",
        "upward", "downward", "ascending", "descending",
        "clockwise", "counter-clockwise",
    ]

    motion_verbs = [
        "moves", "runs", "walks", "drives", "flies", "swims",
        "floats", "flows", "spins", "rotates", "jumps", "jogs",
        "chases", "drifts", "glides", "bounces",
    ]

    combined_text = (vlm_answers_text + " " + original_caption).lower()

    # Check for explicit direction
    for kw in direction_keywords:
        if kw in combined_text:
            return "clear"

    # Check for motion verbs (direction not specified)
    for v in motion_verbs:
        if v in combined_text:
            return "moderate"

    return "vague"


# ===========================================================================
# 辅助
# ===========================================================================

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


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="v9 - VLM前置引导 + 监督LLM + 条件性SVD")

    # I/O
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--caption_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--sample_ids", type=int, nargs="+", required=True)

    # Generation params
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.002,
                   help="SVD noise prior base blend weight (default 0.002)")

    # LLM
    p.add_argument("--llm_model", type=str, default="qwen-plus")

    # VLM
    p.add_argument("--vlm_provider", type=str, default="local")
    p.add_argument("--vlm_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct")

    # Control
    p.add_argument("--skip_svd", action="store_true",
                   help="Skip SVD entirely (all samples use random noise)")
    p.add_argument("--force_svd", action="store_true",
                   help="Force SVD for all samples (ignore motion clarity)")
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.error("Need DASHSCOPE_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1: Read VLM captions
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Read VLM caption")
    logger.info("=" * 60)

    caption_dir = Path(args.caption_dir)
    captions = {}
    for sid in args.sample_ids:
        cap_file = caption_dir / f"{sid}.txt"
        if not cap_file.exists():
            logger.error(f"  caption not found: {cap_file}")
            continue
        captions[sid] = cap_file.read_text(encoding="utf-8").strip()
        logger.info(f"  [{sid}] caption ({len(captions[sid].split())} words)")

    if not captions:
        logger.error("No captions found, exiting")
        sys.exit(1)

    # ==================================================================
    # Step 2: Rewrite LLM - first rewrite + questions
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 2: Rewrite LLM - draft + questions")
    logger.info("=" * 60)

    draft_dir = out_dir / "drafts"
    draft_dir.mkdir(exist_ok=True)
    questions_dir = out_dir / "questions"
    questions_dir.mkdir(exist_ok=True)

    drafts = {}      # sid -> draft text
    all_questions = {}  # sid -> list of questions

    for sid, caption in captions.items():
        d_file = draft_dir / f"{sid}.txt"
        q_file = questions_dir / f"{sid}.json"

        if args.resume and d_file.exists() and q_file.exists():
            drafts[sid] = d_file.read_text(encoding="utf-8").strip()
            all_questions[sid] = json.loads(q_file.read_text(encoding="utf-8"))
            logger.info(f"  [{sid}] (resume)")
            continue

        result = step_rewrite_and_ask(caption, args.llm_model)
        drafts[sid] = result["draft"]
        all_questions[sid] = result["questions"]

        d_file.write_text(result["draft"], encoding="utf-8")
        q_file.write_text(json.dumps(result["questions"], ensure_ascii=False, indent=2),
                          encoding="utf-8")
        logger.info(f"  [{sid}] draft ({len(result['draft'].split())} words) + "
                    f"{len(result['questions'])} questions")

    # ==================================================================
    # Step 3: VLM answers questions
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 3: VLM answers LLM questions")
    logger.info("=" * 60)

    from src.vlm_client import LocalVLMClient
    vlm_client = LocalVLMClient(
        model_path=args.vlm_path,
        temperature=0.3,
        max_tokens=2048,
        max_retries=3,
        use_video_mode=True,
        lazy_load=True,
    )

    answers_dir = out_dir / "vlm_answers"
    answers_dir.mkdir(exist_ok=True)

    vlm_results = {}  # sid -> full vlm result dict

    for sid in captions.keys():
        ans_file = answers_dir / f"{sid}.json"

        if args.resume and ans_file.exists():
            vlm_results[sid] = json.loads(ans_file.read_text(encoding="utf-8"))
            logger.info(f"  [{sid}] (resume)")
            continue

        video_path = str(Path(args.data_dir) / f"{sid}.mp4")
        if not Path(video_path).exists():
            logger.warning(f"  [{sid}] video not found, skipping VLM")
            vlm_results[sid] = {"answers": [], "raw": "", "has_motion_direction": False,
                                "motion_description": ""}
            ans_file.write_text(json.dumps(vlm_results[sid], ensure_ascii=False, indent=2),
                                encoding="utf-8")
            continue

        logger.info(f"  [{sid}] VLM answering {len(all_questions[sid])} questions...")
        result = step_vlm_answer(video_path, all_questions[sid], vlm_client)
        vlm_results[sid] = result

        ans_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # Log answers briefly
        for i, (q, a) in enumerate(zip(all_questions[sid], result["answers"])):
            logger.info(f"      Q{i+1}: {q[:50]}...")
            logger.info(f"      A{i+1}: {a[:80]}")

        if result["has_motion_direction"]:
            logger.info(f"      [motion detected: {result['motion_description']}]")

    # Release VLM VRAM
    vlm_client.unload_model()
    logger.info("  VLM unloaded, GPU memory released")

    # ==================================================================
    # Step 4: Rewrite LLM - second rewrite with VLM answers
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 4: Rewrite LLM - refine with VLM answers")
    logger.info("=" * 60)

    refined_dir = out_dir / "captions_refined"
    refined_dir.mkdir(exist_ok=True)

    refined = {}  # sid -> refined prompt

    for sid, caption in captions.items():
        ref_file = refined_dir / f"{sid}.txt"

        if args.resume and ref_file.exists():
            refined[sid] = ref_file.read_text(encoding="utf-8").strip()
            logger.info(f"  [{sid}] (resume)")
            continue

        result = step_rewrite_with_answers(
            caption=caption,
            draft=drafts[sid],
            questions=all_questions[sid],
            answers=vlm_results[sid]["answers"],
            model=args.llm_model,
        )
        refined[sid] = result
        ref_file.write_text(result, encoding="utf-8")
        logger.info(f"  [{sid}] refined ({len(result.split())} words): {result[:60]}...")

    # ==================================================================
    # Step 5: Supervisor LLM - audit against sources
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 5: Supervisor LLM - audit (caption + VLM answers) vs rewrite")
    logger.info("=" * 60)

    supervisor_dir = out_dir / "supervisor_feedback"
    supervisor_dir.mkdir(exist_ok=True)
    final_dir = out_dir / "captions_final"
    final_dir.mkdir(exist_ok=True)

    final_prompts = {}
    supervisor_results = {}

    for sid, prompt in refined.items():
        original = captions[sid]
        vlm_raw = vlm_results[sid]["raw"]

        # Format VLM answers as readable text for supervisor
        vlm_answers_text = ""
        for i, (q, a) in enumerate(zip(all_questions[sid], vlm_results[sid]["answers"])):
            vlm_answers_text += f"{i+1}. Q: {q}\n   A: {a}\n"
        if not vlm_answers_text:
            vlm_answers_text = "(No VLM answers available)"

        logger.info(f"  [{sid}] Supervisor auditing...")
        sv_result = step_supervise_v9(original, vlm_answers_text, prompt, args.llm_model)
        supervisor_results[sid] = sv_result
        (supervisor_dir / f"{sid}.txt").write_text(sv_result["feedback"], encoding="utf-8")

        if sv_result["passed"]:
            logger.info(f"  [{sid}] PASS")
            final_prompts[sid] = prompt
        else:
            logger.info(f"  [{sid}] FAIL ({len(sv_result['issues'])} issues)")
            for issue in sv_result["issues"][:3]:
                logger.info(f"      - {issue[:80]}")

            # Correct
            logger.info(f"  [{sid}] Correcting...")
            corrected = step_correct(
                original, vlm_answers_text, prompt, sv_result["feedback"], args.llm_model
            )
            final_prompts[sid] = corrected
            logger.info(f"  [{sid}] Corrected: {corrected[:60]}...")

        (final_dir / f"{sid}.txt").write_text(final_prompts[sid], encoding="utf-8")

    # ==================================================================
    # Step 6: SVD conditional decision
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 6: SVD conditional decision (based on VLM motion answers)")
    logger.info("=" * 60)

    svd_decisions = {"use": [], "skip": [], "clarity": {}}

    for sid in final_prompts.keys():
        vlm_raw = vlm_results[sid]["raw"]
        original = captions[sid]
        clarity = extract_motion_clarity(vlm_raw, original)
        svd_decisions["clarity"][sid] = clarity

        if args.skip_svd or args.alpha == 0:
            svd_decisions["skip"].append(sid)
            logger.info(f"  [{sid}] skip (alpha=0 or --skip_svd)")
        elif args.force_svd:
            svd_decisions["use"].append(sid)
            logger.info(f"  [{sid}] force SVD (alpha={args.alpha})")
        elif clarity == "clear":
            svd_decisions["use"].append(sid)
            logger.info(f"  [{sid}] motion=clear -> SVD (alpha={args.alpha})")
        elif clarity == "moderate":
            svd_decisions["use"].append(sid)
            logger.info(f"  [{sid}] motion=moderate -> SVD (alpha={args.alpha})")
        else:
            svd_decisions["skip"].append(sid)
            logger.info(f"  [{sid}] motion=vague -> skip SVD")

    logger.info(f"  Summary: use_SVD={sorted(svd_decisions['use'])}, "
                f"skip_SVD={sorted(svd_decisions['skip'])}")

    # ==================================================================
    # Step 7: Generate videos
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 7: Generate videos")
    logger.info("=" * 60)

    gen_dir = str((out_dir / "generated").resolve())
    final_caption_dir = str(final_dir.resolve())
    data_dir_abs = str(Path(args.data_dir).resolve())

    # 7a: SVD samples
    svd_ids = sorted(svd_decisions["use"])
    if svd_ids and args.alpha > 0:
        logger.info(f"  7a: SVD generation (alpha={args.alpha}): {svd_ids}")
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

    # 7b: No-SVD samples
    skip_ids = sorted(svd_decisions["skip"])
    if skip_ids:
        logger.info(f"  7b: No-SVD generation (random noise): {skip_ids}")
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

    # ==================================================================
    # Step 8: Evaluation
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 8: Evaluate CLIP/XCLIP")
    logger.info("=" * 60)

    all_ids = sorted(final_prompts.keys())
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

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Summary")
    logger.info("=" * 60)

    n_total = len(supervisor_results)
    n_failed = sum(1 for r in supervisor_results.values() if not r["passed"])
    logger.info(f"  Supervisor: {n_failed}/{n_total} samples corrected")
    logger.info(f"  SVD: {len(svd_ids)} use SVD, {len(skip_ids)} skip SVD")

    summary = {
        "version": "v9_vlm_guided",
        "pipeline": "Rewrite+Ask -> VLM Answer -> Rewrite+Merge -> Supervisor -> Conditional SVD -> Generate",
        "innovation": "VLM provides factual answers BEFORE rewrite, not after. LLM has visual evidence to enrich prompt.",
        "svd_decisions": svd_decisions,
        "supervisor_summary": {
            str(sid): {"passed": r["passed"], "issues": r["issues"]}
            for sid, r in supervisor_results.items()
        },
        "config": {
            "alpha": args.alpha,
            "llm_model": args.llm_model,
            "vlm_path": args.vlm_path,
            "skip_svd": args.skip_svd,
            "force_svd": args.force_svd,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
