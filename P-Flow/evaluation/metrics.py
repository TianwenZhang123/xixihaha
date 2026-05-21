"""
Automated Evaluation Metrics for P-Flow.

Implements:
1. FID-VID: Frame-level FID between generated and reference videos
2. FVD: Frechet Video Distance using I3D features
3. Dynamic Degree: Measures temporal complexity/motion intensity

Paper Section 4.1:
- FID-VID: Lower is better (measures visual quality)
- FVD: Lower is better (measures temporal coherence)
- Dynamic Degree: Higher is better (measures effect dynamism)
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class FIDVIDMetric:
    """
    FID-VID: Frechet Inception Distance computed on video frames.

    Extracts frames from videos, computes InceptionV3 features,
    then calculates FID between reference and generated distributions.

    Lower FID-VID = better visual quality per frame.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load InceptionV3."""
        if self._model is None:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self._model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self._model.fc = torch.nn.Identity()  # Remove final FC for features
            self._model = self._model.to(self.device).eval()
        return self._model

    @torch.no_grad()
    def extract_features(self, video_paths: List[str], frames_per_video: int = 16) -> np.ndarray:
        """
        Extract InceptionV3 features from video frames.

        Args:
            video_paths: List of video file paths.
            frames_per_video: Number of frames to sample per video.

        Returns:
            Feature array of shape (N_total_frames, 2048).
        """
        from torchvision import transforms
        from src.video_utils import load_video

        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_features = []

        for video_path in tqdm(video_paths, desc="Extracting FID-VID features"):
            if not os.path.exists(video_path):
                continue

            video = load_video(video_path, num_frames=frames_per_video, device=self.device)
            # video: (C, F, H, W) in [0, 1]

            # Process frames in batches
            frames = video.permute(1, 0, 2, 3)  # (F, C, H, W)
            frames = torch.stack([transform(f) for f in frames])

            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i + self.batch_size].to(self.device)
                features = self.model(batch)
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0) if all_features else np.array([])

    def compute_fid(self, features_ref: np.ndarray, features_gen: np.ndarray) -> float:
        """
        Compute FID between two feature distributions.

        FID = ||mu_ref - mu_gen||^2 + Tr(Sigma_ref + Sigma_gen - 2*sqrt(Sigma_ref*Sigma_gen))
        """
        from scipy.linalg import sqrtm

        mu_ref = np.mean(features_ref, axis=0)
        mu_gen = np.mean(features_gen, axis=0)
        sigma_ref = np.cov(features_ref, rowvar=False)
        sigma_gen = np.cov(features_gen, rowvar=False)

        diff = mu_ref - mu_gen
        covmean, _ = sqrtm(sigma_ref @ sigma_gen, disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.dot(diff, diff) + np.trace(sigma_ref + sigma_gen - 2 * covmean)
        return float(fid)

    def evaluate(self, reference_videos: List[str], generated_videos: List[str],
                 frames_per_video: int = 16) -> Dict[str, float]:
        """
        Compute FID-VID between reference and generated video sets.

        Args:
            reference_videos: Paths to reference videos.
            generated_videos: Paths to generated videos.
            frames_per_video: Frames to sample per video.

        Returns:
            {"fid_vid": float}
        """
        print(f"Computing FID-VID ({len(reference_videos)} ref, {len(generated_videos)} gen)...")

        features_ref = self.extract_features(reference_videos, frames_per_video)
        features_gen = self.extract_features(generated_videos, frames_per_video)

        if len(features_ref) < 2 or len(features_gen) < 2:
            return {"fid_vid": float("inf"), "error": "insufficient_samples"}

        fid = self.compute_fid(features_ref, features_gen)
        return {"fid_vid": fid, "num_ref_frames": len(features_ref),
                "num_gen_frames": len(features_gen)}


class FVDMetric:
    """
    Frechet Video Distance (FVD).

    Uses I3D model to extract spatiotemporal features from videos,
    then computes Frechet distance between distributions.

    Lower FVD = better temporal coherence and video quality.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load I3D model."""
        if self._model is None:
            # Use torchvision's video model as I3D proxy
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                self._model = r3d_18(weights=R3D_18_Weights.DEFAULT)
                self._model.fc = torch.nn.Identity()
                self._model = self._model.to(self.device).eval()
            except ImportError:
                # Fallback: simple 3D conv feature extractor
                self._model = self._build_simple_3d_encoder()
        return self._model

    def _build_simple_3d_encoder(self):
        """Fallback 3D encoder if pretrained model unavailable."""
        model = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
            torch.nn.Flatten(),
        )
        return model.to(self.device).eval()

    @torch.no_grad()
    def extract_features(self, video_paths: List[str], num_frames: int = 16) -> np.ndarray:
        """Extract I3D/R3D features from videos."""
        from src.video_utils import load_video

        all_features = []

        for video_path in tqdm(video_paths, desc="Extracting FVD features"):
            if not os.path.exists(video_path):
                continue

            video = load_video(video_path, num_frames=num_frames, height=112, width=112, device=self.device)
            # video: (C, F, H, W) -> (B, C, F, H, W) for 3D model
            video_input = video.unsqueeze(0)

            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1, 1)
            video_input = (video_input - mean) / std

            features = self.model(video_input)
            all_features.append(features.cpu().numpy().flatten())

        return np.array(all_features) if all_features else np.array([]).reshape(0, 512)

    def compute_fvd(self, features_ref: np.ndarray, features_gen: np.ndarray) -> float:
        """Compute FVD (same formula as FID but on video features)."""
        from scipy.linalg import sqrtm

        if len(features_ref) < 2 or len(features_gen) < 2:
            return float("inf")

        mu_ref = np.mean(features_ref, axis=0)
        mu_gen = np.mean(features_gen, axis=0)
        sigma_ref = np.cov(features_ref, rowvar=False)
        sigma_gen = np.cov(features_gen, rowvar=False)

        # Handle 1D case
        if sigma_ref.ndim == 0:
            sigma_ref = np.array([[sigma_ref]])
            sigma_gen = np.array([[sigma_gen]])

        diff = mu_ref - mu_gen
        covmean, _ = sqrtm(sigma_ref @ sigma_gen, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fvd = np.dot(diff, diff) + np.trace(sigma_ref + sigma_gen - 2 * covmean)
        return float(fvd)

    def evaluate(self, reference_videos: List[str], generated_videos: List[str]) -> Dict[str, float]:
        """Compute FVD between video sets."""
        print(f"Computing FVD ({len(reference_videos)} ref, {len(generated_videos)} gen)...")

        features_ref = self.extract_features(reference_videos)
        features_gen = self.extract_features(generated_videos)

        fvd = self.compute_fvd(features_ref, features_gen)
        return {"fvd": fvd, "num_ref": len(features_ref), "num_gen": len(features_gen)}


class DynamicDegreeMetric:
    """
    Dynamic Degree: Measures motion intensity and visual transformation.

    Computes optical flow magnitude and frame-to-frame differences
    to quantify how dynamic/active the visual effects are.

    Higher Dynamic Degree = more active/dynamic effects.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute_optical_flow_magnitude(self, video_path: str, num_frames: int = 81) -> float:
        """
        Compute average optical flow magnitude as motion metric.

        Uses frame differences as proxy when optical flow is expensive.
        """
        from src.video_utils import load_video

        if not os.path.exists(video_path):
            return 0.0

        video = load_video(video_path, num_frames=num_frames, device="cpu")
        # video: (C, F, H, W)

        frames = video.permute(1, 2, 3, 0).numpy()  # (F, H, W, C)

        # Compute frame differences as motion proxy
        diffs = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
            diffs.append(diff.mean())

        return float(np.mean(diffs)) if diffs else 0.0

    def compute_temporal_variance(self, video_path: str, num_frames: int = 81) -> float:
        """
        Compute temporal variance of pixel values.
        High variance = more dynamic visual changes.
        """
        from src.video_utils import load_video

        if not os.path.exists(video_path):
            return 0.0

        video = load_video(video_path, num_frames=num_frames, device="cpu")
        # Compute per-pixel temporal variance, then average
        temporal_var = video.var(dim=1).mean().item()
        return temporal_var

    def evaluate(self, video_paths: List[str]) -> Dict[str, float]:
        """
        Compute Dynamic Degree for a set of videos.

        Returns average motion magnitude and temporal variance.
        """
        print(f"Computing Dynamic Degree ({len(video_paths)} videos)...")

        motion_scores = []
        variance_scores = []

        for path in tqdm(video_paths, desc="Dynamic Degree"):
            motion = self.compute_optical_flow_magnitude(path)
            variance = self.compute_temporal_variance(path)
            motion_scores.append(motion)
            variance_scores.append(variance)

        return {
            "dynamic_degree_motion": float(np.mean(motion_scores)),
            "dynamic_degree_variance": float(np.mean(variance_scores)),
            "dynamic_degree_combined": float(np.mean(motion_scores) + np.mean(variance_scores)),
            "num_videos": len(video_paths),
        }


def run_full_evaluation(
    reference_videos: List[str],
    generated_videos: List[str],
    output_path: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run complete evaluation suite (FID-VID + FVD + Dynamic Degree).

    Args:
        reference_videos: Paths to reference videos.
        generated_videos: Paths to generated videos.
        output_path: Path to save evaluation results.
        device: Compute device.

    Returns:
        Complete evaluation results dict.
    """
    results = {}

    # FID-VID
    try:
        fid_metric = FIDVIDMetric(device=device)
        fid_results = fid_metric.evaluate(reference_videos, generated_videos)
        results.update(fid_results)
    except Exception as e:
        results["fid_vid_error"] = str(e)

    # FVD
    try:
        fvd_metric = FVDMetric(device=device)
        fvd_results = fvd_metric.evaluate(reference_videos, generated_videos)
        results.update(fvd_results)
    except Exception as e:
        results["fvd_error"] = str(e)

    # Dynamic Degree (on generated videos)
    try:
        dd_metric = DynamicDegreeMetric(device=device)
        dd_results = dd_metric.evaluate(generated_videos)
        results.update(dd_results)
    except Exception as e:
        results["dynamic_degree_error"] = str(e)

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation Results:")
    print(f"  FID-VID: {results.get('fid_vid', 'N/A')}")
    print(f"  FVD: {results.get('fvd', 'N/A')}")
    print(f"  Dynamic Degree: {results.get('dynamic_degree_combined', 'N/A')}")
    print(f"  Saved to: {output_path}")

    return results
