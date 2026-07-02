#!/usr/bin/env python3
"""使用本地 Qwen2.5-VL 对目录中的视频批量生成 caption（取帧模式）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
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
        "--max-new-tokens",
        type=int,
        default=200,
        help="caption 最大生成 token 数。",
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
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="从视频中均匀采样的帧数。",
    )
    return parser.parse_args()


def discover_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    return sorted(path for path in video_dir.glob("*.mp4") if path.is_file())


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


def caption_video(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: Path,
    prompt: str,
    max_new_tokens: int,
    num_frames: int = 32,
) -> str:
    # 1. 从视频均匀提取 num_frames 帧
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(frame_count - 1, 0), num=num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to extract any frame from {video_path}")

    # 2. 构造 messages（图像 + 文本）
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in frames],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")

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


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No mp4 videos found in {video_dir}")

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

    summary = {
        "model_path": str(model_path),
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "captions": results,
    }
    write_json(output_dir / "captions.json", summary)
    print(f"[saved] {output_dir / 'captions.json'}")


if __name__ == "__main__":
    main()
