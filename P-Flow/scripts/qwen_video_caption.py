#!/usr/bin/env python3
"""使用本地 Qwen2.5-VL 对目录中的视频批量生成 caption。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# 某些运行环境会注入非法的 OMP_NUM_THREADS，先兜底为 1，避免 libgomp 警告。
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch caption videos with local Qwen2.5-VL.")
    parser.add_argument(
        "--video-dir",
        default="/root/video/generated/video",
        help="待 caption 的视频目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/baseline/caption",
        help="caption 输出目录。",
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct",
        help="本地 Qwen2.5-VL 模型目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="推理设备，例如 cuda:0 或 cpu。",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="只处理指定视频 id，支持逗号和连续范围；例如 a005-a010 或 a005,a007。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="caption 最大生成 token 数。",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="每个视频送入 VLM 的 OpenCV 均匀采样帧数；0 表示不采样、使用完整视频。默认 32。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已存在的 caption 文件。",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the video content in English.",
    )
    return parser.parse_args()


def split_id_parts(value: str) -> tuple[str, str]:
    prefix = value.rstrip("0123456789")
    digits = value[len(prefix):]
    return prefix, digits


def expand_id_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []

    if "-" not in token:
        return [token]

    start, end = [part.strip() for part in token.split("-", 1)]
    start_prefix, start_digits = split_id_parts(start)
    end_prefix, end_digits = split_id_parts(end)
    if (
        not start_digits
        or not end_digits
        or start_prefix != end_prefix
        or len(start_digits) != len(end_digits)
    ):
        return [token]

    start_num = int(start_digits)
    end_num = int(end_digits)
    step = 1 if end_num >= start_num else -1
    width = len(start_digits)
    return [
        f"{start_prefix}{number:0{width}d}"
        for number in range(start_num, end_num + step, step)
    ]


def parse_ids(ids_arg: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for token in ids_arg.split(","):
        for item in expand_id_token(token):
            if item not in seen:
                ids.append(item)
                seen.add(item)
    return ids


def filter_paths_by_ids(paths: list[Path], ids_arg: str, kind: str) -> list[Path]:
    target_ids = parse_ids(ids_arg)
    if not target_ids:
        return paths

    by_stem = {path.stem: path for path in paths}
    missing = [item for item in target_ids if item not in by_stem]
    if missing:
        raise FileNotFoundError(f"Missing {kind} ids: {', '.join(missing)}")
    return [by_stem[item] for item in target_ids]


def discover_videos(video_dir: Path, ids_arg: str = "") -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    videos = sorted(path for path in video_dir.glob("*.mp4") if path.is_file())
    return filter_paths_by_ids(videos, ids_arg, "video")


def build_model(model_path: Path, device: str) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    torch_dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map=device if device.startswith("cuda") else None,
        local_files_only=True,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, processor


def move_inputs_to_device(inputs: dict, device: str, torch_dtype: torch.dtype) -> dict:
    moved: dict = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue

        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=torch_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def sample_video_with_opencv(video_path: Path, num_frames: int) -> tuple[list[np.ndarray | str], dict]:
    if num_frames <= 0:
        return [str(video_path)], {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read frame count with OpenCV: {video_path}")

    sample_count = min(num_frames, total_frames)
    if sample_count > 1 and sample_count % 2:
        sample_count -= 1
    sample_count = max(1, sample_count)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    frames: list[np.ndarray] = []
    used_indices: list[int] = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        used_indices.append(int(frame_index))
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to sample frames with OpenCV: {video_path}")

    if len(frames) % 2:
        frames.append(frames[-1].copy())
        used_indices.append(used_indices[-1])

    video = np.stack(frames)
    metadata = {
        "total_num_frames": total_frames,
        "fps": fps if fps > 0 else None,
        "duration": total_frames / fps if fps > 0 else None,
        "frames_indices": used_indices,
        "height": int(video.shape[1]),
        "width": int(video.shape[2]),
        "video_backend": "opencv",
    }
    return [video], {"video_metadata": [metadata]}


def caption_video(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: Path,
    prompt: str,
    max_new_tokens: int,
    num_frames: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    videos, video_kwargs = sample_video_with_opencv(video_path, num_frames)
    inputs = processor(
        text=[text],
        videos=videos,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    model_device = str(model.device)
    model_dtype = next(model.parameters()).dtype
    inputs = move_inputs_to_device(inputs, device=model_device, torch_dtype=model_dtype)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def write_text(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_caption(text: str) -> str:
    return " ".join(text.strip().split())


def caption_videos(
    *,
    args: argparse.Namespace,
    videos: list[Path],
    output_dir: Path,
    model_path: Path,
) -> list[dict[str, str]]:
    model, processor = build_model(model_path, args.device)
    results: list[dict[str, str]] = []

    for video_path in videos:
        text_path = output_dir / f"{video_path.stem}.txt"
        if text_path.exists() and not args.overwrite:
            caption = text_path.read_text(encoding="utf-8").strip()
            print(f"[skip] {video_path.name} -> {text_path.name}")
        else:
            print(f"[caption] {video_path.name}")
            caption = normalize_caption(
                caption_video(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_frames=args.num_frames,
                )
            )
            write_text(text_path, caption)
            print(f"[done] {video_path.name}: {caption}")

        results.append(
            {
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "video_path": str(video_path),
                "caption": caption,
            }
        )

    return results


def write_caption_summary(
    *,
    args: argparse.Namespace,
    videos: list[Path],
    output_dir: Path,
    model_path: Path,
) -> None:
    results: list[dict[str, str]] = []
    for video_path in videos:
        text_path = output_dir / f"{video_path.stem}.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"Caption output missing: {text_path}")
        results.append(
            {
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "video_path": str(video_path),
                "caption": text_path.read_text(encoding="utf-8").strip(),
            }
        )

    summary: dict[str, object] = {
        "model_path": str(model_path),
        "video_dir": str(Path(args.video_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "ids": parse_ids(args.ids),
        "num_frames": args.num_frames,
        "captions": results,
    }
    write_json(output_dir / "captions.json", summary)
    print(f"[saved] {output_dir / 'captions.json'}")


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(video_dir, args.ids)
    if not videos:
        raise FileNotFoundError(f"No mp4 videos found in {video_dir}")

    caption_videos(args=args, videos=videos, output_dir=output_dir, model_path=model_path)
    write_caption_summary(
        args=args,
        videos=videos,
        output_dir=output_dir,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
