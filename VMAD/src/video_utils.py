"""
Video I/O and Processing Utilities for VMAD.

复用 P-Flow 的视频工具，支持:
    - 视频加载 (decord/cv2/imageio)
    - 视频保存 (mp4)
    - 帧提取和缩放
    - 归一化/反归一化
"""

import os
import numpy as np
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn.functional as TF
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


def load_video(
    video_path: str,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    加载视频文件为 tensor。

    Args:
        video_path: 视频路径
        num_frames: 目标帧数 (均匀采样)
        height: 目标高度
        width: 目标宽度
        device: 目标设备

    Returns:
        Video tensor (C, F, H, W) in [0, 1]
    """
    if HAS_DECORD:
        return _load_video_decord(video_path, num_frames, height, width, device)
    elif HAS_CV2:
        return _load_video_cv2(video_path, num_frames, height, width, device)
    elif HAS_IMAGEIO:
        return _load_video_imageio(video_path, num_frames, height, width, device)
    else:
        raise ImportError("Need one of: decord, opencv-python, or imageio")


def _load_video_decord(video_path, num_frames, height, width, device):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if num_frames and num_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
    frames = vr.get_batch(indices).asnumpy()
    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(3, 0, 1, 2)  # (C, F, H, W)
    if height and width:
        video = resize_video(video, height, width)
    return video.to(device)


def _load_video_cv2(video_path, num_frames, height, width, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.array(frames)
    total = len(frames)
    if num_frames and num_frames < total:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = frames[indices]
    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(3, 0, 1, 2)
    if height and width:
        video = resize_video(video, height, width)
    return video.to(device)


def _load_video_imageio(video_path, num_frames, height, width, device):
    reader = imageio.get_reader(video_path)
    frames = [f for f in reader]
    reader.close()
    frames = np.array(frames)
    total = len(frames)
    if num_frames and num_frames < total:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = frames[indices]
    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(3, 0, 1, 2)
    if height and width:
        video = resize_video(video, height, width)
    return video.to(device)


def save_video_tensor(
    video: torch.Tensor,
    output_path: str,
    fps: int = 16,
):
    """
    保存 video tensor 为 mp4 文件。

    Args:
        video: (C, F, H, W) 或 (B, C, F, H, W) in [0, 1]
        output_path: 输出路径
        fps: 帧率
    """
    if video.dim() == 5:
        video = video[0]

    frames = video.permute(1, 2, 3, 0).cpu().float().numpy()
    frames = (frames * 255).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if HAS_IMAGEIO:
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    elif HAS_CV2:
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
    else:
        raise ImportError("Need imageio or opencv-python to save videos")


def resize_video(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """缩放视频 tensor (C, F, H, W)。"""
    C, F, H, W = video.shape
    if H == height and W == width:
        return video
    video_flat = video.reshape(C * F, 1, H, W)
    video_resized = TF.interpolate(
        video_flat, size=(height, width), mode="bilinear", align_corners=False
    )
    return video_resized.reshape(C, F, height, width)


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]"""
    return video * 2.0 - 1.0


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (video + 1.0) / 2.0
