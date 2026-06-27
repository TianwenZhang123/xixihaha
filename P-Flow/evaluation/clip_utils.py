"""
Shared CLIP/X-CLIP evaluation utilities.

Extracted from run_clip_xclip_eval.py and eval_caption_quality.py
to eliminate duplicate implementations.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, XCLIPModel


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_numeric_id(path: Path) -> str | None:
    import re
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else None


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return float(np.dot(a, b))


def sample_video_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    """Uniformly sample frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video frame count: {video_path}")

    indices = np.linspace(0, max(frame_count - 1, 0), num=num_frames, dtype=int)
    frames = []
    wanted = set(int(i) for i in indices.tolist())
    current = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        current += 1
        if len(frames) >= len(wanted):
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to sample frames from video: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


def build_models(device: str, clip_model_name: str, xclip_model_name: str):
    """Load CLIP and X-CLIP models."""
    print(f"Loading CLIP model from: {clip_model_name}", flush=True)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
    clip_model.eval()

    print(f"Loading X-CLIP model from: {xclip_model_name}", flush=True)
    xclip_processor = AutoProcessor.from_pretrained(xclip_model_name, local_files_only=True)
    xclip_model = XCLIPModel.from_pretrained(xclip_model_name, local_files_only=True).to(device)
    xclip_model.eval()

    return clip_processor, clip_model, xclip_processor, xclip_model


@torch.inference_mode()
def get_clip_text_feature(text: str, processor, model, device: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", truncation=True).to(device)
    features = model.get_text_features(**inputs)
    return features[0].detach().float().cpu().numpy()


@torch.inference_mode()
def get_clip_video_feature(frames: list[Image.Image], processor, model, device: str) -> np.ndarray:
    inputs = processor(images=frames, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    mean_feature = features.detach().float().cpu().numpy().mean(axis=0)
    return mean_feature


@torch.inference_mode()
def get_xclip_text_feature(text: str, processor, model, device: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", truncation=True).to(device)
    features = model.get_text_features(**inputs)
    return features[0].detach().float().cpu().numpy()


@torch.inference_mode()
def get_xclip_video_feature(frames: list[Image.Image], processor, model, device: str) -> np.ndarray:
    np_frames = [np.array(frame) for frame in frames]
    inputs = processor(videos=[np_frames], return_tensors="pt").to(device)
    features = model.get_video_features(**inputs)
    return features[0].detach().float().cpu().numpy()


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def mean_of(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(sum(row[key] for row in rows) / len(rows))
