"""
Shot Boundary Detection Module for P-Flow.

将长视频切分为多个镜头片段（shot），每个片段可独立走 P-Flow pipeline。

支持两种后端：
    1. TransNetV2（推荐，深度学习方法，精度高、速度快）
    2. PySceneDetect（轻量备选，纯 CPU OpenCV）

典型用法：
    from src.shot_detect import ShotDetector

    detector = ShotDetector(backend="transnetv2", threshold=0.5)
    shots = detector.detect("long_video.mp4")
    # shots = [Shot(start_frame=0, end_frame=72, ...), Shot(...), ...]

    # 导出为独立视频文件
    detector.export_shots("long_video.mp4", shots, output_dir="./shots/")

    # 或直接获取每段的 tensor
    shot_tensors = detector.extract_shot_tensors("long_video.mp4", shots)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─── Optional Dependencies ────────────────────────────────────────────────

try:
    from transnetv2_pytorch import TransNetV2 as _TransNetV2Model
    HAS_TRANSNETV2 = True
except ImportError:
    try:
        # fallback: TensorFlow 原版
        from transnetv2 import TransNetV2 as _TransNetV2Model
        HAS_TRANSNETV2 = True
    except ImportError:
        HAS_TRANSNETV2 = False

try:
    from scenedetect import detect as _sd_detect
    from scenedetect import ContentDetector
    HAS_PYSCENEDETECT = True
except ImportError:
    HAS_PYSCENEDETECT = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


# ─── Data Structures ──────────────────────────────────────────────────────


@dataclass
class Shot:
    """表示一个镜头片段。"""

    start_frame: int          # 起始帧索引（含）
    end_frame: int            # 结束帧索引（含）
    start_time: float         # 起始时间 (秒)
    end_time: float           # 结束时间 (秒)
    duration: float           # 时长 (秒)
    num_frames: int           # 帧数
    confidence: float = 1.0   # 边界检测置信度
    transition_type: str = "cut"  # 转场类型: cut / dissolve / fade / wipe

    def __repr__(self):
        return (
            f"Shot(frames=[{self.start_frame}:{self.end_frame}], "
            f"time=[{self.start_time:.2f}s:{self.end_time:.2f}s], "
            f"duration={self.duration:.2f}s, type={self.transition_type})"
        )


@dataclass
class ShotDetectionResult:
    """切分镜完整结果。"""

    video_path: str
    total_frames: int
    fps: float
    duration: float
    shots: List[Shot]
    # 过滤后的可用片段
    valid_shots: List[Shot] = field(default_factory=list)

    @property
    def num_shots(self) -> int:
        return len(self.shots)

    @property
    def num_valid_shots(self) -> int:
        return len(self.valid_shots)

    def summary(self) -> str:
        lines = [
            f"视频: {self.video_path}",
            f"总帧数: {self.total_frames}, FPS: {self.fps:.1f}, 时长: {self.duration:.2f}s",
            f"检测到 {self.num_shots} 个镜头, 有效 {self.num_valid_shots} 个",
            "─" * 60,
        ]
        for i, shot in enumerate(self.shots):
            valid_mark = "✓" if shot in self.valid_shots else "✗"
            lines.append(f"  [{valid_mark}] Shot {i}: {shot}")
        return "\n".join(lines)


# ─── Shot Detector ────────────────────────────────────────────────────────


class ShotDetector:
    """
    镜头边界检测器。

    支持后端:
        - "transnetv2": TransNetV2 深度学习模型（推荐）
        - "pyscenedetect": PySceneDetect content-aware 检测
        - "auto": 自动选择可用的最优后端

    Args:
        backend: 检测后端名称
        threshold: 边界判定阈值
            - TransNetV2: 帧级概率阈值，默认 0.5
            - PySceneDetect: 内容变化阈值，默认 27.0
        min_shot_duration: 最短有效镜头时长（秒），过短的片段会被过滤
        max_shot_duration: 最长有效镜头时长（秒），过长的片段会被切分
        target_duration: 目标片段时长（秒），用于 pad/trim
        device: TransNetV2 推理设备
    """

    def __init__(
        self,
        backend: str = "auto",
        threshold: Optional[float] = None,
        min_shot_duration: float = 1.0,
        max_shot_duration: float = 8.0,
        target_duration: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.backend = self._resolve_backend(backend)
        self.min_shot_duration = min_shot_duration
        self.max_shot_duration = max_shot_duration
        self.target_duration = target_duration
        self.device = device

        # 设置阈值默认值
        if threshold is None:
            self.threshold = 0.5 if self.backend == "transnetv2" else 27.0
        else:
            self.threshold = threshold

        # 初始化模型
        self._model = None
        if self.backend == "transnetv2":
            self._init_transnetv2()

        logger.info(
            f"ShotDetector initialized: backend={self.backend}, "
            f"threshold={self.threshold}, target_duration={self.target_duration}s"
        )

    def _resolve_backend(self, backend: str) -> str:
        """自动选择可用后端。"""
        if backend == "auto":
            if HAS_TRANSNETV2:
                return "transnetv2"
            elif HAS_PYSCENEDETECT:
                return "pyscenedetect"
            else:
                raise ImportError(
                    "No SBD backend available. Install one of:\n"
                    "  pip install transnetv2-pytorch  (recommended)\n"
                    "  pip install scenedetect[opencv]"
                )
        elif backend == "transnetv2":
            if not HAS_TRANSNETV2:
                raise ImportError(
                    "TransNetV2 not installed. Run: pip install transnetv2-pytorch"
                )
            return "transnetv2"
        elif backend == "pyscenedetect":
            if not HAS_PYSCENEDETECT:
                raise ImportError(
                    "PySceneDetect not installed. Run: pip install scenedetect[opencv]"
                )
            return "pyscenedetect"
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'transnetv2' or 'pyscenedetect'")

    def _init_transnetv2(self):
        """加载 TransNetV2 模型。"""
        logger.info("Loading TransNetV2 model...")
        self._model = _TransNetV2Model()
        if hasattr(self._model, 'to'):
            self._model = self._model.to(self.device)
        if hasattr(self._model, 'eval'):
            self._model.eval()
        logger.info("TransNetV2 model loaded.")

    # ─── Core Detection ───────────────────────────────────────────────

    def detect(self, video_path: str) -> ShotDetectionResult:
        """
        检测视频中的镜头边界。

        Args:
            video_path: 视频文件路径

        Returns:
            ShotDetectionResult 包含所有检测到的镜头及过滤结果
        """
        video_path = str(Path(video_path).resolve())
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 获取视频元信息
        fps, total_frames = self._get_video_info(video_path)
        duration = total_frames / fps

        logger.info(
            f"Detecting shots in: {video_path} "
            f"(frames={total_frames}, fps={fps:.1f}, duration={duration:.2f}s)"
        )

        # 执行检测
        if self.backend == "transnetv2":
            boundaries = self._detect_transnetv2(video_path, total_frames)
        else:
            boundaries = self._detect_pyscenedetect(video_path)

        # 构建 Shot 列表
        shots = self._boundaries_to_shots(boundaries, total_frames, fps)

        # 过滤有效片段
        valid_shots = self._filter_shots(shots)

        result = ShotDetectionResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            shots=shots,
            valid_shots=valid_shots,
        )

        logger.info(f"Detection complete: {result.num_shots} shots, {result.num_valid_shots} valid")
        return result

    def _detect_transnetv2(
        self, video_path: str, total_frames: int
    ) -> List[Tuple[int, float]]:
        """
        使用 TransNetV2 检测边界。

        Returns:
            List of (frame_index, confidence) for each detected boundary.
        """
        # 读取视频帧（TransNetV2 需要原始帧，内部做 resize 到 48x27）
        frames = self._load_frames_for_transnetv2(video_path)

        # TransNetV2 推理
        with torch.no_grad():
            if hasattr(self._model, 'predict_frames'):
                # transnetv2-pytorch API
                predictions = self._model.predict_frames(frames)
                if isinstance(predictions, tuple):
                    single_frame_pred = predictions[0]  # (N,) 逐帧概率
                else:
                    single_frame_pred = predictions
            elif hasattr(self._model, 'predict_video'):
                # 原版 TF API
                result = self._model.predict_video(video_path)
                if isinstance(result, tuple):
                    single_frame_pred = result[0]
                else:
                    single_frame_pred = result
            else:
                # 通用 forward
                if isinstance(frames, np.ndarray):
                    frames_tensor = torch.from_numpy(frames).float()
                    if frames_tensor.dim() == 4:  # (N, H, W, C)
                        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
                    frames_tensor = frames_tensor.to(self.device)
                else:
                    frames_tensor = frames.to(self.device)
                output = self._model(frames_tensor)
                if isinstance(output, tuple):
                    single_frame_pred = output[0]
                else:
                    single_frame_pred = output
                if isinstance(single_frame_pred, torch.Tensor):
                    single_frame_pred = single_frame_pred.cpu().numpy()

        # 确保是 numpy array
        if isinstance(single_frame_pred, torch.Tensor):
            single_frame_pred = single_frame_pred.cpu().numpy()

        single_frame_pred = np.asarray(single_frame_pred).flatten()

        # 阈值过滤，提取边界帧
        boundaries = []
        for i, prob in enumerate(single_frame_pred):
            if prob >= self.threshold:
                boundaries.append((i, float(prob)))

        # NMS: 合并过于接近的边界（5帧内取最高）
        boundaries = self._nms_boundaries(boundaries, min_distance=5)

        logger.info(f"TransNetV2 detected {len(boundaries)} boundaries (threshold={self.threshold})")
        return boundaries

    def _detect_pyscenedetect(self, video_path: str) -> List[Tuple[int, float]]:
        """
        使用 PySceneDetect 检测边界。

        Returns:
            List of (frame_index, confidence) for each detected boundary.
        """
        scene_list = _sd_detect(video_path, ContentDetector(threshold=self.threshold))

        boundaries = []
        for i, (start, end) in enumerate(scene_list):
            if i > 0:  # 第一个 scene 的 start 是视频开头，不算边界
                frame_idx = start.get_frames()
                boundaries.append((frame_idx, 1.0))

        logger.info(f"PySceneDetect detected {len(boundaries)} boundaries (threshold={self.threshold})")
        return boundaries

    # ─── Post-Processing ──────────────────────────────────────────────

    def _boundaries_to_shots(
        self, boundaries: List[Tuple[int, float]], total_frames: int, fps: float
    ) -> List[Shot]:
        """将边界帧列表转换为 Shot 列表。"""
        shots = []
        boundary_frames = sorted([b[0] for b in boundaries])
        confidences = {b[0]: b[1] for b in boundaries}

        # 构建片段区间
        cut_points = [0] + boundary_frames + [total_frames - 1]

        for i in range(len(cut_points) - 1):
            start_f = cut_points[i]
            end_f = cut_points[i + 1]

            # 边界帧本身属于下一个 shot（除了第一个）
            if i > 0:
                start_f = cut_points[i]

            num_frames = end_f - start_f + 1
            start_time = start_f / fps
            end_time = end_f / fps
            duration = num_frames / fps
            confidence = confidences.get(cut_points[i], 1.0) if i > 0 else 1.0

            shots.append(Shot(
                start_frame=start_f,
                end_frame=end_f,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                num_frames=num_frames,
                confidence=confidence,
            ))

        return shots

    def _filter_shots(self, shots: List[Shot]) -> List[Shot]:
        """
        过滤并调整镜头片段，使之适配 P-Flow 处理。

        规则:
            1. 过短（< min_shot_duration）的片段合并到前一个或丢弃
            2. 过长（> max_shot_duration）的片段均匀切分为多个子片段
        """
        if not shots:
            return []

        valid = []

        for shot in shots:
            if shot.duration < self.min_shot_duration:
                # 过短片段：尝试合并到上一个
                if valid:
                    prev = valid[-1]
                    merged = Shot(
                        start_frame=prev.start_frame,
                        end_frame=shot.end_frame,
                        start_time=prev.start_time,
                        end_time=shot.end_time,
                        duration=prev.duration + shot.duration,
                        num_frames=prev.num_frames + shot.num_frames,
                        confidence=max(prev.confidence, shot.confidence),
                        transition_type=prev.transition_type,
                    )
                    valid[-1] = merged
                # 如果是第一个且过短，先暂存，后面可能被合并
                else:
                    valid.append(shot)
            elif shot.duration > self.max_shot_duration:
                # 过长片段：均匀切分
                sub_shots = self._split_long_shot(shot)
                valid.extend(sub_shots)
            else:
                valid.append(shot)

        # 二次过滤：确保合并后的片段不会过短
        final = [s for s in valid if s.duration >= self.min_shot_duration]

        return final

    def _split_long_shot(self, shot: Shot) -> List[Shot]:
        """将过长镜头均匀切分为 target_duration 大小的子片段。"""
        num_sub = max(1, round(shot.duration / self.target_duration))
        frames_per_sub = shot.num_frames // num_sub

        fps = shot.num_frames / shot.duration  # 局部 fps

        sub_shots = []
        for i in range(num_sub):
            start_f = shot.start_frame + i * frames_per_sub
            if i == num_sub - 1:
                end_f = shot.end_frame
            else:
                end_f = start_f + frames_per_sub - 1

            num_frames = end_f - start_f + 1
            start_time = start_f / fps if fps > 0 else 0
            end_time = end_f / fps if fps > 0 else 0
            duration = num_frames / fps if fps > 0 else 0

            sub_shots.append(Shot(
                start_frame=start_f,
                end_frame=end_f,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                num_frames=num_frames,
                confidence=shot.confidence,
                transition_type="split",
            ))

        return sub_shots

    # ─── Video I/O Helpers ────────────────────────────────────────────

    def _get_video_info(self, video_path: str) -> Tuple[float, int]:
        """获取视频的 fps 和总帧数。"""
        if HAS_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            del vr
            return fps, total_frames
        elif HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return fps, total_frames
        else:
            raise ImportError("Need decord or opencv-python to read video info")

    def _load_frames_for_transnetv2(self, video_path: str) -> np.ndarray:
        """加载全部帧为 numpy array (N, H, W, 3) uint8。"""
        if HAS_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = vr.get_batch(range(len(vr))).asnumpy()  # (N, H, W, 3)
            del vr
            return frames
        elif HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return np.array(frames, dtype=np.uint8)
        else:
            raise ImportError("Need decord or opencv-python")

    @staticmethod
    def _nms_boundaries(
        boundaries: List[Tuple[int, float]], min_distance: int = 5
    ) -> List[Tuple[int, float]]:
        """对过于接近的边界做 NMS，保留置信度最高的。"""
        if not boundaries:
            return []

        # 按帧索引排序
        boundaries = sorted(boundaries, key=lambda x: x[0])

        result = [boundaries[0]]
        for frame_idx, conf in boundaries[1:]:
            prev_frame, prev_conf = result[-1]
            if frame_idx - prev_frame < min_distance:
                # 过于接近，保留更高置信度的
                if conf > prev_conf:
                    result[-1] = (frame_idx, conf)
            else:
                result.append((frame_idx, conf))

        return result

    # ─── Export & Extraction ──────────────────────────────────────────

    def export_shots(
        self,
        video_path: str,
        shots: Optional[List[Shot]] = None,
        output_dir: str = "./shots",
        use_valid_only: bool = True,
    ) -> List[str]:
        """
        将检测到的镜头导出为独立视频文件。

        Args:
            video_path: 源视频路径
            shots: Shot 列表（如果 None 则先执行 detect）
            output_dir: 输出目录
            use_valid_only: 是否只导出有效片段

        Returns:
            导出的文件路径列表
        """
        if shots is None:
            result = self.detect(video_path)
            shots = result.valid_shots if use_valid_only else result.shots

        os.makedirs(output_dir, exist_ok=True)
        fps, _ = self._get_video_info(video_path)

        exported_paths = []
        video_stem = Path(video_path).stem

        for i, shot in enumerate(shots):
            output_path = os.path.join(
                output_dir, f"{video_stem}_shot{i:03d}_{shot.start_frame}-{shot.end_frame}.mp4"
            )
            self._export_single_shot(video_path, shot, output_path, fps)
            exported_paths.append(output_path)
            logger.debug(f"Exported shot {i}: {output_path}")

        logger.info(f"Exported {len(exported_paths)} shots to {output_dir}/")
        return exported_paths

    def _export_single_shot(
        self, video_path: str, shot: Shot, output_path: str, fps: float
    ):
        """导出单个 shot 为视频文件。"""
        if HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)

            # 获取视频尺寸
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for _ in range(shot.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            cap.release()
        elif HAS_DECORD:
            import imageio
            vr = VideoReader(video_path, ctx=cpu(0))
            indices = list(range(shot.start_frame, min(shot.end_frame + 1, len(vr))))
            frames = vr.get_batch(indices).asnumpy()
            del vr

            writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        else:
            raise ImportError("Need opencv-python or decord+imageio to export shots")

    def extract_shot_tensors(
        self,
        video_path: str,
        shots: Optional[List[Shot]] = None,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        device: str = "cpu",
    ) -> List[torch.Tensor]:
        """
        提取每个 shot 的帧为 P-Flow 格式的 tensor。

        Args:
            video_path: 源视频路径
            shots: Shot 列表
            num_frames: 每段采样帧数（默认 81 = Wan2.1 的 5s@16fps）
            height: 目标高度
            width: 目标宽度
            device: 输出设备

        Returns:
            List of tensors, each (C, F, H, W) in [0, 1]
        """
        if shots is None:
            result = self.detect(video_path)
            shots = result.valid_shots

        # 加载全部帧
        if HAS_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            raise ImportError("Need decord for tensor extraction")

        tensors = []
        for shot in shots:
            indices = list(range(shot.start_frame, min(shot.end_frame + 1, len(vr))))

            # 均匀采样到目标帧数
            if len(indices) > num_frames:
                sample_indices = np.linspace(0, len(indices) - 1, num_frames, dtype=int)
                indices = [indices[j] for j in sample_indices]
            elif len(indices) < num_frames:
                # 不足则重复最后一帧 pad
                pad_count = num_frames - len(indices)
                indices = indices + [indices[-1]] * pad_count

            frames = vr.get_batch(indices).asnumpy()  # (F, H, W, 3)
            tensor = torch.from_numpy(frames).float() / 255.0
            tensor = tensor.permute(3, 0, 1, 2)  # (C, F, H, W)

            # Resize
            if tensor.shape[2] != height or tensor.shape[3] != width:
                tensor = self._resize_video_tensor(tensor, height, width)

            tensors.append(tensor.to(device))

        if HAS_DECORD:
            del vr

        logger.info(f"Extracted {len(tensors)} shot tensors, shape: (3, {num_frames}, {height}, {width})")
        return tensors

    @staticmethod
    def _resize_video_tensor(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize video tensor (C, F, H, W)."""
        import torch.nn.functional as TF
        C, F, H, W = video.shape
        if H == height and W == width:
            return video
        video_flat = video.reshape(C * F, 1, H, W)
        video_resized = TF.interpolate(
            video_flat, size=(height, width), mode="bilinear", align_corners=False
        )
        return video_resized.reshape(C, F, height, width)


# ─── Convenience Functions ────────────────────────────────────────────────


def detect_shots(
    video_path: str,
    backend: str = "auto",
    threshold: Optional[float] = None,
    min_duration: float = 1.0,
    max_duration: float = 8.0,
) -> ShotDetectionResult:
    """
    便捷函数：一行代码完成切分镜。

    Args:
        video_path: 视频路径
        backend: "transnetv2" / "pyscenedetect" / "auto"
        threshold: 检测阈值
        min_duration: 最短有效时长
        max_duration: 最长有效时长

    Returns:
        ShotDetectionResult
    """
    detector = ShotDetector(
        backend=backend,
        threshold=threshold,
        min_shot_duration=min_duration,
        max_shot_duration=max_duration,
    )
    return detector.detect(video_path)


def split_video_to_shots(
    video_path: str,
    output_dir: str = "./shots",
    backend: str = "auto",
    threshold: Optional[float] = None,
) -> List[str]:
    """
    便捷函数：切分视频并导出为多个文件。

    Returns:
        导出的视频文件路径列表
    """
    detector = ShotDetector(backend=backend, threshold=threshold)
    result = detector.detect(video_path)
    return detector.export_shots(video_path, result.valid_shots, output_dir=output_dir)


# ─── CLI Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P-Flow Shot Boundary Detection")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--backend", default="auto", choices=["auto", "transnetv2", "pyscenedetect"])
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Min shot duration (s)")
    parser.add_argument("--max-duration", type=float, default=8.0, help="Max shot duration (s)")
    parser.add_argument("--target-duration", type=float, default=5.0, help="Target shot duration (s)")
    parser.add_argument("--output-dir", default="./shots", help="Output directory for exported shots")
    parser.add_argument("--export", action="store_true", help="Export shots as separate video files")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = ShotDetector(
        backend=args.backend,
        threshold=args.threshold,
        min_shot_duration=args.min_duration,
        max_shot_duration=args.max_duration,
        target_duration=args.target_duration,
    )

    result = detector.detect(args.video)
    print(result.summary())

    if args.export:
        paths = detector.export_shots(args.video, result.valid_shots, output_dir=args.output_dir)
        print(f"\nExported {len(paths)} shots to {args.output_dir}/")
        for p in paths:
            print(f"  {p}")
