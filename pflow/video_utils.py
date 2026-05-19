"""
Video I/O and Processing Utilities for P-Flow.

Handles video loading, saving, frame extraction, composite video creation,
and format conversions between different tensor formats.

Key addition for paper-faithful implementation:
- create_vertical_composite(): Vertical stacking (top/middle/bottom)
  for VLM input as described in Section 3.5.
"""

import os
import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as TF
from PIL import Image, ImageDraw, ImageFont

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
    fps: Optional[float] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load a video file and return as a tensor.

    Args:
        video_path: Path to the video file.
        num_frames: Target number of frames (uniformly sampled if video is longer).
        height: Target height (resize if specified).
        width: Target width (resize if specified).
        fps: Target FPS (resample if specified).
        device: Device to place tensor on.

    Returns:
        Video tensor of shape (C, F, H, W) in [0, 1] range.
    """
    if HAS_DECORD:
        return _load_video_decord(video_path, num_frames, height, width, device)
    elif HAS_CV2:
        return _load_video_cv2(video_path, num_frames, height, width, device)
    elif HAS_IMAGEIO:
        return _load_video_imageio(video_path, num_frames, height, width, device)
    else:
        raise ImportError("Need one of: decord, opencv-python, or imageio")


def _load_video_decord(
    video_path: str,
    num_frames: Optional[int],
    height: Optional[int],
    width: Optional[int],
    device: str,
) -> torch.Tensor:
    """Load video using decord (fastest)."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if num_frames is not None and num_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)

    frames = vr.get_batch(indices).asnumpy()  # (F, H, W, C) uint8

    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(3, 0, 1, 2)  # (F, H, W, C) -> (C, F, H, W)

    if height is not None and width is not None:
        video = resize_video(video, height, width)

    return video.to(device)


def _load_video_cv2(
    video_path: str,
    num_frames: Optional[int],
    height: Optional[int],
    width: Optional[int],
    device: str,
) -> torch.Tensor:
    """Load video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    total_frames = len(frames)

    if num_frames is not None and num_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = frames[indices]

    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(3, 0, 1, 2)  # (C, F, H, W)

    if height is not None and width is not None:
        video = resize_video(video, height, width)

    return video.to(device)


def _load_video_imageio(
    video_path: str,
    num_frames: Optional[int],
    height: Optional[int],
    width: Optional[int],
    device: str,
) -> torch.Tensor:
    """Load video using imageio."""
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()

    frames = np.array(frames)
    total_frames = len(frames)

    if num_frames is not None and num_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = frames[indices]

    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(3, 0, 1, 2)  # (C, F, H, W)

    if height is not None and width is not None:
        video = resize_video(video, height, width)

    return video.to(device)


def save_video_tensor(
    video: torch.Tensor,
    output_path: str,
    fps: int = 16,
):
    """
    Save a video tensor to file.

    Args:
        video: Video tensor of shape (C, F, H, W) or (B, C, F, H, W) in [0, 1].
        output_path: Output file path (e.g., "output.mp4").
        fps: Frames per second.
    """
    if video.dim() == 5:
        video = video[0]

    # (C, F, H, W) -> (F, H, W, C)
    frames = video.permute(1, 2, 3, 0).cpu().numpy()
    frames = (frames * 255).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

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


def save_frame(
    frame: torch.Tensor,
    output_path: str,
):
    """
    Save a single frame tensor as an image.

    Args:
        frame: Frame tensor of shape (C, H, W) in [0, 1].
        output_path: Output image path.
    """
    if frame.dim() == 4:
        frame = frame[0]

    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(frame_np)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    img.save(output_path)


def resize_video(
    video: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Resize a video tensor.

    Args:
        video: Video tensor (C, F, H, W).
        height: Target height.
        width: Target width.

    Returns:
        Resized video tensor (C, F, height, width).
    """
    C, num_f, H, W = video.shape

    if H == height and W == width:
        return video

    # Reshape to (C*F, 1, H, W) for batch interpolation
    video_flat = video.reshape(C * num_f, 1, H, W)
    video_resized = TF.interpolate(
        video_flat,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    video_resized = video_resized.reshape(C, num_f, height, width)

    return video_resized


def create_vertical_composite(
    videos: List[torch.Tensor],
    labels: Optional[List[str]] = None,
    padding: int = 4,
    label_height: int = 24,
) -> torch.Tensor:
    """
    Create a VERTICAL composite video from multiple video tensors.

    Paper Section 3.5 layout:
        Panel A (top): Reference video
        Panel B (middle): Previous generated video
        Panel C (bottom): Current generated video

    This vertical stacking ensures the VLM can compare videos
    in the natural top-to-bottom reading order.

    Args:
        videos: List of video tensors, each (C, F, H, W) in [0, 1].
        labels: Optional text labels for each panel (e.g., ["A: Reference", "B: Previous", "C: Current"]).
        padding: Pixels of padding between panels.
        label_height: Height reserved for text labels (pixels).

    Returns:
        Composite video tensor (C, F, H_total, W) with vertical stacking.
    """
    # Ensure all videos have the same number of frames and width
    num_frames = min(v.shape[1] for v in videos)
    width = max(v.shape[3] for v in videos)

    # Process videos: match frame count and width
    processed_videos = []
    for v in videos:
        # Match frame count
        if v.shape[1] > num_frames:
            indices = torch.linspace(0, v.shape[1] - 1, num_frames).long()
            v = v[:, indices]
        # Match width (resize if different)
        if v.shape[3] != width:
            v = resize_video(v, v.shape[2], width)
        processed_videos.append(v)

    # Calculate total height: sum of all video heights + padding + labels
    total_height = (
        sum(v.shape[2] for v in processed_videos)
        + padding * (len(videos) - 1)
        + label_height * len(videos)
    )

    # Create composite tensor
    C = processed_videos[0].shape[0]
    composite = torch.zeros(C, num_frames, total_height, width)

    # Stack videos vertically with labels
    y_offset = 0
    for i, v in enumerate(processed_videos):
        h = v.shape[2]

        # Add label area (white background with text rendered later)
        # For now, just leave a white strip
        composite[:, :, y_offset:y_offset + label_height, :] = 1.0  # White
        y_offset += label_height

        # Add video content
        composite[:, :, y_offset:y_offset + h, :] = v
        y_offset += h + padding

    return composite


def create_composite_video(
    videos: List[torch.Tensor],
    labels: Optional[List[str]] = None,
    padding: int = 4,
    label_height: int = 30,
) -> torch.Tensor:
    """
    Create a side-by-side (horizontal) composite video.

    KEPT for backward compatibility with API mode (run_pflow_api.py).
    The paper-faithful code uses create_vertical_composite() instead.

    Args:
        videos: List of video tensors, each (C, F, H, W) in [0, 1].
        labels: Optional text labels for each video.
        padding: Pixels of padding between videos.
        label_height: Height reserved for labels.

    Returns:
        Composite video tensor (C, F, H, W_total) with horizontal layout.
    """
    num_frames = min(v.shape[1] for v in videos)
    height = videos[0].shape[2]

    processed_videos = []
    for v in videos:
        if v.shape[1] > num_frames:
            indices = torch.linspace(0, v.shape[1] - 1, num_frames).long()
            v = v[:, indices]
        if v.shape[2] != height:
            v = resize_video(v, height, v.shape[3])
        processed_videos.append(v)

    total_width = sum(v.shape[3] for v in processed_videos) + padding * (len(videos) - 1)

    C = processed_videos[0].shape[0]
    composite = torch.zeros(C, num_frames, height, total_width)

    x_offset = 0
    for i, v in enumerate(processed_videos):
        w = v.shape[3]
        composite[:, :, :, x_offset:x_offset + w] = v
        x_offset += w + padding

    return composite


def extract_key_frames(
    video: torch.Tensor,
    num_frames: int = 8,
) -> List[torch.Tensor]:
    """
    Extract uniformly spaced key frames from a video.

    Args:
        video: Video tensor (C, F, H, W) or (B, C, F, H, W).
        num_frames: Number of key frames to extract.

    Returns:
        List of frame tensors, each (C, H, W).
    """
    if video.dim() == 5:
        video = video[0]

    total_frames = video.shape[1]
    indices = torch.linspace(0, total_frames - 1, num_frames).long()

    frames = [video[:, idx] for idx in indices]
    return frames


def video_to_pil_frames(
    video: torch.Tensor,
    num_frames: Optional[int] = None,
) -> List[Image.Image]:
    """
    Convert video tensor to list of PIL Images.

    Args:
        video: Video tensor (C, F, H, W) in [0, 1].
        num_frames: If specified, uniformly sample this many frames.

    Returns:
        List of PIL Images.
    """
    if video.dim() == 5:
        video = video[0]

    C, F, H, W = video.shape

    if num_frames is not None and num_frames < F:
        indices = torch.linspace(0, F - 1, num_frames).long()
    else:
        indices = torch.arange(F)

    pil_frames = []
    for idx in indices:
        frame = video[:, idx].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))

    return pil_frames


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """Normalize video tensor from [0, 1] to [-1, 1]."""
    return video * 2.0 - 1.0


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    """Denormalize video tensor from [-1, 1] to [0, 1]."""
    return (video + 1.0) / 2.0
