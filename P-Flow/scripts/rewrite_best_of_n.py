#!/usr/bin/env python3
"""
Best-of-N Prompt Rewrite + CLIP 选择

路线B：生成 N 个候选 caption，用 CLIP(text, video) 打分选最佳。
核心思想来自 ARPO (2025)：多候选 + reward-guided 选择。

流程:
1. 对每个原始 caption，用 LLM 生成 N 个改写候选 (温度 0.3~0.6 梯度)
2. 用 CLIP text encoder 对每个候选编码
3. 与参考视频的 CLIP visual embedding 计算余弦相似度
4. 选择得分最高的候选作为最终输出

用法:
    # 基本用法（需要视频目录来做 CLIP 打分）
    python scripts/rewrite_best_of_n.py \
        --input-dir data/captions_baseline \
        --output-dir data/captions_best_of_n \
        --video-dir data/reference_videos \
        --n-candidates 5 \
        --backend dashscope \
        --model qwen-plus

    # 不带 CLIP 打分（退化为多次采样 + edit-distance 最小化选择）
    python scripts/rewrite_best_of_n.py \
        --input-dir data/captions_baseline \
        --output-dir data/captions_best_of_n \
        --n-candidates 5 \
        --no-clip \
        --backend dashscope \
        --model qwen-plus
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 复用 rewrite_hybrid 中的 prompt 模板和工具函数
sys.path.insert(0, str(Path(__file__).parent))
from rewrite_hybrid import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    call_dashscope,
    call_openai_compatible,
    _compute_edit_ratio,
    validate_rewrite,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP 打分模块
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_model(device: str = "cuda"):
    """加载 CLIP 模型用于 text-video 相似度打分"""
    try:
        import torch
        import open_clip
    except ImportError:
        raise ImportError("需要安装: pip install open_clip_torch torch")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def get_video_embedding(video_path: str, model, preprocess, device: str = "cuda"):
    """从视频中均匀采样帧，取 CLIP visual embedding 的均值"""
    import torch
    import numpy as np

    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        # 均匀采样 8 帧
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (8, H, W, 3)
    except ImportError:
        # 退而使用 opencv
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.array(frames)

    from PIL import Image
    embeddings = []
    with torch.no_grad():
        for i in range(len(frames)):
            img = Image.fromarray(frames[i])
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            emb = model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)

    # 取均值作为视频的视觉表示
    video_emb = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
    video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
    return video_emb


def score_candidates_clip(
    candidates: List[str],
    video_emb,  # (1, D) tensor
    model,
    tokenizer,
    device: str = "cuda",
) -> List[float]:
    """用 CLIP 计算每个候选 caption 与视频的相似度"""
    import torch

    tokens = tokenizer(candidates).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # 余弦相似度
    scores = (text_embs @ video_emb.T).squeeze(-1).cpu().tolist()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 候选生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(
    original: str,
    n_candidates: int,
    backend: str,
    model: str,
    api_base: str = "",
    api_key: str = "",
    base_temperature: float = 0.2,
    temp_spread: float = 0.3,
) -> List[str]:
    """生成 N 个候选改写，温度从 base 到 base+spread 线性递增"""
    word_count = len(original.split())
    user_msg = USER_TEMPLATE.format(
        word_count=word_count,
        original_caption=original,
    )

    candidates = []
    for i in range(n_candidates):
        # 温度梯度：让前几个候选更保守，后几个更有创造性
        temp = base_temperature + (temp_spread * i / max(n_candidates - 1, 1))

        try:
            if backend == "dashscope":
                result = call_dashscope(user_msg, SYSTEM_PROMPT, model, api_key, temp)
            elif backend == "openai":
                result = call_openai_compatible(user_msg, SYSTEM_PROMPT, model, api_base, api_key, temp)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            # 清理引号
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            if result.startswith("'") and result.endswith("'"):
                result = result[1:-1]

            candidates.append(result)
        except Exception as e:
            logger.warning(f"    候选 {i+1} 生成失败: {e}")
            continue

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 选择策略
# ─────────────────────────────────────────────────────────────────────────────

def select_best_no_clip(
    candidates: List[str], original: str
) -> Tuple[str, int, dict]:
    """无 CLIP 时的选择策略：综合 edit_ratio 最小 + 验证通过"""
    best_idx = 0
    best_score = -1.0
    scores_info = []

    for i, cand in enumerate(candidates):
        validation = validate_rewrite(original, cand)
        edit_ratio = _compute_edit_ratio(original, cand)

        # 分数 = 验证通过加分 + (1 - edit_ratio) 作为保留率
        score = 0.0
        if validation["valid"]:
            score += 0.5
        preservation = 1.0 - edit_ratio
        score += preservation * 0.5

        scores_info.append({
            "idx": i,
            "edit_ratio": round(edit_ratio, 4),
            "preservation": round(preservation, 4),
            "valid": validation["valid"],
            "score": round(score, 4),
        })

        if score > best_score:
            best_score = score
            best_idx = i

    return candidates[best_idx], best_idx, {"method": "edit_distance", "scores": scores_info}


def select_best_with_clip(
    candidates: List[str],
    original: str,
    video_emb,
    model,
    tokenizer,
    device: str = "cuda",
    clip_weight: float = 0.6,
    preservation_weight: float = 0.4,
) -> Tuple[str, int, dict]:
    """CLIP + preservation 联合打分"""
    clip_scores = score_candidates_clip(candidates, video_emb, model, tokenizer, device)

    best_idx = 0
    best_combined = -1.0
    scores_info = []

    for i, cand in enumerate(candidates):
        edit_ratio = _compute_edit_ratio(original, cand)
        preservation = 1.0 - edit_ratio

        # 归一化 CLIP 分 (通常在 0.2~0.4 之间，做 min-max)
        clip_s = clip_scores[i]

        # 联合分数
        combined = clip_weight * clip_s + preservation_weight * preservation

        scores_info.append({
            "idx": i,
            "clip_score": round(clip_s, 4),
            "edit_ratio": round(edit_ratio, 4),
            "preservation": round(preservation, 4),
            "combined": round(combined, 4),
        })

        if combined > best_combined:
            best_combined = combined
            best_idx = i

    return candidates[best_idx], best_idx, {"method": "clip+preservation", "scores": scores_info}


# ─────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N Prompt Rewrite：生成多候选 + CLIP 打分选最优"
    )

    # I/O
    parser.add_argument("--input-dir", type=str, required=True,
                        help="原始 caption 目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录（最优候选）")
    parser.add_argument("--video-dir", type=str, default="",
                        help="参考视频目录（用于 CLIP 打分，包含 {id}.mp4）")
    parser.add_argument("--sample-ids", type=int, nargs="+",
                        help="只处理指定样本 ID")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的输出文件")

    # Best-of-N 参数
    parser.add_argument("--n-candidates", type=int, default=5,
                        help="每个样本生成的候选数量 (默认: 5)")
    parser.add_argument("--base-temperature", type=float, default=0.2,
                        help="最低温度 (默认: 0.2)")
    parser.add_argument("--temp-spread", type=float, default=0.3,
                        help="温度递增范围 (默认: 0.3, 即 0.2~0.5)")

    # CLIP 参数
    parser.add_argument("--no-clip", action="store_true",
                        help="不使用 CLIP 打分（退化为 edit-distance 选择）")
    parser.add_argument("--clip-weight", type=float, default=0.6,
                        help="CLIP 分数权重 (默认: 0.6)")
    parser.add_argument("--preservation-weight", type=float, default=0.4,
                        help="保留率权重 (默认: 0.4)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="CLIP 推理设备 (默认: cuda)")

    # LLM 后端
    parser.add_argument("--backend", type=str, default="dashscope",
                        choices=["dashscope", "openai"])
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")

    # 其他
    parser.add_argument("--delay", type=float, default=0.5,
                        help="候选间请求间隔 (默认: 0.5s)")
    parser.add_argument("--save-all-candidates", action="store_true",
                        help="保存所有候选及打分详情到 JSON")

    args = parser.parse_args()

    # API Key
    api_key = args.api_key
    if not api_key:
        if args.backend == "dashscope":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if args.backend == "dashscope" and not api_key:
        logger.error("需要设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    # 目录
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    # CLIP 模型 (按需加载)
    clip_model = clip_preprocess = clip_tokenizer = None
    if not args.no_clip:
        if not args.video_dir:
            logger.error("使用 CLIP 打分需要指定 --video-dir")
            sys.exit(1)
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            logger.error(f"视频目录不存在: {video_dir}")
            sys.exit(1)

        logger.info("加载 CLIP 模型...")
        clip_model, clip_preprocess, clip_tokenizer = load_clip_model(args.device)
        logger.info("CLIP 模型加载完成")

    # 收集文件
    caption_files = sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.sample_ids:
        id_set = set(args.sample_ids)
        caption_files = [f for f in caption_files if int(f.stem) in id_set]

    if not caption_files:
        logger.error(f"未找到 caption 文件")
        sys.exit(1)

    logger.info(f"待处理: {len(caption_files)} 个样本, N={args.n_candidates}")
    logger.info(f"选择策略: {'CLIP + preservation' if not args.no_clip else 'edit-distance only'}")

    # 主循环
    all_results = []
    success = 0

    for idx, cap_file in enumerate(caption_files, 1):
        sample_id = cap_file.stem
        out_file = output_dir / f"{sample_id}.txt"

        if args.skip_existing and out_file.exists():
            logger.info(f"  [{idx}/{len(caption_files)}] 跳过 {sample_id}")
            continue

        original = cap_file.read_text(encoding="utf-8").strip()
        if not original:
            continue

        logger.info(f"  [{idx}/{len(caption_files)}] {sample_id}: 生成 {args.n_candidates} 个候选...")

        # 生成候选
        candidates = generate_candidates(
            original=original,
            n_candidates=args.n_candidates,
            backend=args.backend,
            model=args.model,
            api_base=args.api_base,
            api_key=api_key,
            base_temperature=args.base_temperature,
            temp_spread=args.temp_spread,
        )

        if not candidates:
            logger.error(f"    {sample_id}: 未生成任何有效候选")
            continue

        # 打分选择
        if args.no_clip:
            best, best_idx, info = select_best_no_clip(candidates, original)
        else:
            # 加载视频 embedding
            video_path = None
            video_dir = Path(args.video_dir)
            for ext in [".mp4", ".avi", ".mov", ".webm"]:
                candidate_path = video_dir / f"{sample_id}{ext}"
                if candidate_path.exists():
                    video_path = str(candidate_path)
                    break

            if video_path is None:
                logger.warning(f"    {sample_id}: 未找到视频，退化为 edit-distance 选择")
                best, best_idx, info = select_best_no_clip(candidates, original)
            else:
                video_emb = get_video_embedding(video_path, clip_model, clip_preprocess, args.device)
                best, best_idx, info = select_best_with_clip(
                    candidates, original, video_emb,
                    clip_model, clip_tokenizer, args.device,
                    args.clip_weight, args.preservation_weight,
                )

        # 保存最佳候选
        out_file.write_text(best + "\n", encoding="utf-8")
        success += 1

        edit_ratio = _compute_edit_ratio(original, best)
        logger.info(
            f"    → 选择候选 {best_idx+1}/{len(candidates)}, "
            f"edit_ratio={edit_ratio:.1%}, method={info['method']}"
        )

        result = {
            "sample_id": sample_id,
            "n_generated": len(candidates),
            "selected_idx": best_idx,
            "edit_ratio": round(edit_ratio, 4),
            "selection_info": info,
        }
        if args.save_all_candidates:
            result["candidates"] = candidates
        all_results.append(result)

        time.sleep(args.delay)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! 成功={success}/{len(caption_files)}")
    logger.info(f"输出目录: {output_dir}")

    # 保存详细日志
    log_file = output_dir / "best_of_n_log.json"
    log_data = {
        "config": {
            "n_candidates": args.n_candidates,
            "base_temperature": args.base_temperature,
            "temp_spread": args.temp_spread,
            "clip_enabled": not args.no_clip,
            "clip_weight": args.clip_weight,
            "preservation_weight": args.preservation_weight,
            "backend": args.backend,
            "model": args.model,
        },
        "results": all_results,
    }
    log_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"详细日志: {log_file}")


if __name__ == "__main__":
    main()
