#!/usr/bin/env python3
"""
多分镜短片批量生成脚本 — P-Flow 数据集构建。

从 data/storyboards/ 目录读取三分镜剧本 txt 文件，
使用 Wan2.1-T2V-1.3B 逐分镜生成视频，再用 ffmpeg 拼接为 ~15s 短片。

Usage:
    # 生成全部 50 条（3 个类别）
    python scripts/generate_storyboard_videos.py

    # 只生成某个类别
    python scripts/generate_storyboard_videos.py --category human_action

    # 限制数量（调试用）
    python scripts/generate_storyboard_videos.py --limit 2

    # 自定义参数
    python scripts/generate_storyboard_videos.py --steps 30 --guidance-scale 5.0

    # 断点续跑（跳过已存在的视频）
    python scripts/generate_storyboard_videos.py --resume
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STORYBOARD_DIR = PROJECT_ROOT / "data" / "storyboards"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated_videos"

# 模型路径（AutoDL 服务器）
MODEL_DIR = Path("/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers")

from src.constants import NEGATIVE_PROMPT

# 类别列表
CATEGORIES = ["human_action", "animal_nature", "physics_motion"]


# ============================================================
# 数据结构
# ============================================================
@dataclass
class StoryConfig:
    """三分镜剧本配置。"""
    story_name: str
    scenes: list[dict[str, str]]
    steps: int = 30
    guidance_scale: float = 5.0
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 15
    seed: int = 2026


# ============================================================
# 剧本解析（复用 Video2Prompt/wan_storyboard_lib.py 的逻辑）
# ============================================================
def parse_storyboard_text_file(path: Path) -> StoryConfig:
    """解析三分镜剧本 txt 文件为 StoryConfig。"""
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]

    story_name = path.stem
    scenes: list[dict[str, str]] = []
    current_title = None
    current_prompt_lines: list[str] = []

    # 解析生成参数
    steps = 30
    guidance_scale = 5.0
    height = 480
    width = 832
    num_frames = 81
    fps = 15
    seed = 2026

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("项目名称："):
            story_name = line.split("：", 1)[1].strip() or story_name
            continue
        if line.startswith("生成参数：") or line.startswith("分镜剧本："):
            continue
        # 解析参数行
        if line.startswith("- "):
            param = line[2:].strip()
            if param.startswith("steps:"):
                steps = int(param.split(":")[1].strip())
            elif param.startswith("guidance_scale:"):
                guidance_scale = float(param.split(":")[1].strip())
            elif param.startswith("分辨率:"):
                res = param.split(":")[1].strip()
                if "x" in res:
                    width, height = int(res.split("x")[0]), int(res.split("x")[1])
            elif param.startswith("帧数:"):
                num_frames = int(param.split(":")[1].strip())
            elif param.startswith("帧率:"):
                fps = int(param.split(":")[1].strip())
            elif param.startswith("随机种子基值:"):
                seed = int(param.split(":")[1].strip())
            continue
        # 解析分镜
        if line[0].isdigit() and ". " in line:
            if current_title and current_prompt_lines:
                scenes.append({
                    "title": current_title,
                    "prompt": "".join(current_prompt_lines).strip(),
                })
            current_title = line.split(". ", 1)[1].strip()
            current_prompt_lines = []
            continue
        if current_title:
            current_prompt_lines.append(line)

    if current_title and current_prompt_lines:
        scenes.append({
            "title": current_title,
            "prompt": "".join(current_prompt_lines).strip(),
        })

    if not scenes:
        raise ValueError(f"剧本文件中未解析出有效分镜：{path}")

    return StoryConfig(
        story_name=story_name,
        scenes=scenes,
        steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
    )


def collect_storyboard_files(
    storyboard_dir: Path,
    category: str | None = None,
) -> list[tuple[str, Path]]:
    """
    收集剧本文件。

    Returns:
        list of (category_name, file_path) tuples, sorted by filename.
    """
    results = []
    categories = [category] if category else CATEGORIES

    for cat in categories:
        cat_dir = storyboard_dir / cat
        if not cat_dir.exists():
            print(f"  ⚠️  类别目录不存在，跳过: {cat_dir}")
            continue
        for txt_file in sorted(cat_dir.glob("*.txt")):
            if txt_file.is_file():
                results.append((cat, txt_file))

    return results


# ============================================================
# 视频拼接
# ============================================================
def concat_videos(video_paths: list[Path], output_path: Path) -> None:
    """使用 ffmpeg 拼接多段视频。"""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    list_path = output_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join([f"file '{path.as_posix()}'" for path in video_paths]),
        encoding="utf-8",
    )

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # 清理临时文件
    list_path.unlink(missing_ok=True)


# ============================================================
# 模型加载
# ============================================================
def check_model_files(model_dir: Path) -> list[str]:
    """检查模型文件完整性。"""
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    missing = []
    for rel in required:
        if not (model_dir / rel).exists():
            missing.append(rel)
    return missing


def build_pipeline(model_dir: Path):
    """加载 Wan2.1-T2V-1.3B pipeline。"""
    from diffusers import AutoencoderKLWan, WanPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    vae = AutoencoderKLWan.from_pretrained(
        model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipe = WanPipeline.from_pretrained(
        model_dir,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to(device)
    return pipe, device


# ============================================================
# 单条剧本生成
# ============================================================
def generate_story(
    pipe,
    device: str,
    config: StoryConfig,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """
    生成单条三分镜短片。

    Args:
        pipe: WanPipeline instance
        device: cuda/cpu
        config: 剧本配置
        output_dir: 输出目录（如 data/generated_videos/human_action/HA01_xxx/）
        force: 是否强制重新生成

    Returns:
        生成结果字典
    """
    from diffusers.utils import export_to_video

    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths: list[Path] = []
    for idx, scene in enumerate(config.scenes, start=1):
        output_path = output_dir / f"scene_{idx}.mp4"

        if output_path.exists() and output_path.stat().st_size > 0 and not force:
            print(f"    跳过已存在分镜: scene_{idx}.mp4")
            video_paths.append(output_path)
            continue

        print(f"    生成分镜 {idx}/{len(config.scenes)}: {scene['title']}")
        generator = torch.Generator(device=device).manual_seed(config.seed + idx)

        frames = pipe(
            prompt=scene["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        ).frames[0]

        export_to_video(frames, str(output_path), fps=config.fps)
        print(f"    已保存: {output_path.name}")
        video_paths.append(output_path)

    # 拼接成完整短片
    merged_path = output_dir / "full_video.mp4"
    if merged_path.exists() and merged_path.stat().st_size > 0 and not force:
        print(f"    跳过已存在成片: full_video.mp4")
    else:
        concat_videos(video_paths, merged_path)
        print(f"    已合并成片: {merged_path.name}")

    return {
        "story_name": config.story_name,
        "output_dir": str(output_dir),
        "scene_videos": [str(p) for p in video_paths],
        "merged_video": str(merged_path),
    }


# ============================================================
# 主流程
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P-Flow 多分镜短片批量生成"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=CATEGORIES,
        help="只生成指定类别，默认全部"
    )
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条，0=全部")
    parser.add_argument("--steps", type=int, default=None, help="覆盖推理步数")
    parser.add_argument("--guidance-scale", type=float, default=None, help="覆盖引导系数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子基值")
    parser.add_argument("--resume", action="store_true", help="跳过已有输出的剧本")
    parser.add_argument("--force", action="store_true", help="强制重新生成所有视频")
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help="Wan2.1 模型目录路径"
    )
    parser.add_argument(
        "--storyboard-dir", type=str, default=str(STORYBOARD_DIR),
        help="剧本文件目录"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="视频输出目录"
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    """清理文件名中的特殊字符。"""
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            cleaned.append(ch)
        elif ch in {" ", "　"}:
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    return result or "未命名短片"


def main() -> None:
    args = parse_args()

    storyboard_dir = Path(args.storyboard_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    print("=" * 60)
    print("  P-Flow 多分镜短片批量生成")
    print("=" * 60)
    print(f"  剧本目录: {storyboard_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  模型目录: {model_dir}")
    print(f"  类别: {args.category or '全部'}")
    print()

    # 1. 收集剧本文件
    story_files = collect_storyboard_files(storyboard_dir, args.category)
    if args.limit > 0:
        story_files = story_files[:args.limit]

    if not story_files:
        print("❌ 未找到任何剧本文件！")
        sys.exit(1)

    print(f"  找到 {len(story_files)} 条剧本待生成")
    print()

    # 2. 检查模型
    print("检查模型文件...")
    missing = check_model_files(model_dir)
    if missing:
        print(f"❌ 模型文件不完整，缺少: {missing}")
        print(f"   请确认模型路径: {model_dir}")
        sys.exit(2)
    print("  ✅ 模型文件完整")
    print()

    # 3. 加载 Pipeline
    print("加载 Wan2.1-T2V-1.3B Pipeline...")
    pipe, device = build_pipeline(model_dir)
    print("  ✅ Pipeline 加载完成")
    print()

    # 4. 逐条生成
    results = []
    total = len(story_files)
    start_time = time.time()

    for idx, (category, story_file) in enumerate(story_files, start=1):
        print(f"{'='*60}")
        print(f"[{idx}/{total}] {category}/{story_file.name}")
        print(f"{'='*60}")

        # 解析剧本
        config = parse_storyboard_text_file(story_file)

        # 覆盖 CLI 参数
        if args.steps is not None:
            config.steps = args.steps
        if args.guidance_scale is not None:
            config.guidance_scale = args.guidance_scale
        if args.seed is not None:
            config.seed = args.seed

        # 输出目录: data/generated_videos/{category}/{story_name}/
        story_output_dir = output_dir / category / sanitize_name(config.story_name)

        # 断点续跑检查
        merged_path = story_output_dir / "full_video.mp4"
        if args.resume and merged_path.exists() and merged_path.stat().st_size > 0:
            print(f"  ⏭️  跳过（已存在成片）")
            continue

        # 生成
        result = generate_story(
            pipe=pipe,
            device=device,
            config=config,
            output_dir=story_output_dir,
            force=args.force,
        )
        results.append(result)

        elapsed = time.time() - start_time
        avg_per_story = elapsed / idx
        remaining = avg_per_story * (total - idx)
        print(f"  ⏱️  已用 {elapsed/60:.1f}min, 预计剩余 {remaining/60:.1f}min")
        print()

    # 5. 汇总
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  ✅ 批量生成完成!")
    print(f"  总计: {len(results)}/{total} 条")
    print(f"  耗时: {total_time/60:.1f} 分钟")
    print(f"  输出: {output_dir}")
    print("=" * 60)

    # 保存生成记录
    import json
    log_path = output_dir / "generation_log.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_data = {
        "total": total,
        "completed": len(results),
        "time_minutes": round(total_time / 60, 1),
        "results": results,
    }
    log_path.write_text(
        json.dumps(log_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  生成日志: {log_path}")


if __name__ == "__main__":
    main()
