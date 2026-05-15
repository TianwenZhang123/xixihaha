"""
Evaluation Metrics for P-Flow.

Implements the metrics used in the paper:
- FID-VID: Fréchet Inception Distance for Video
- FVD: Fréchet Video Distance
- Dynamic Degree: Measures motion/dynamics in generated videos

Usage:
    python scripts/evaluate.py \
        --reference_dir path/to/reference_videos/ \
        --generated_dir path/to/generated_videos/ \
        --metrics fid_vid fvd dynamic_degree
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class FIDVIDCalculator:
    """
    FID-VID: Fréchet Inception Distance for Video.
    
    Computes FID between distributions of video features extracted
    using a pre-trained I3D or VideoMAE model.
    """
    
    def __init__(self, device: str = "cuda", model_name: str = "i3d"):
        """
        Args:
            device: Computation device.
            model_name: Feature extractor model ('i3d' or 'videomae').
        """
        self.device = device
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the feature extraction model."""
        if self.model_name == "i3d":
            # Use torchvision's video model as proxy
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                model = r3d_18(weights=R3D_18_Weights.DEFAULT)
                # Remove classification head, keep features
                model.fc = nn.Identity()
                model = model.to(self.device).eval()
                return model
            except ImportError:
                print("Warning: torchvision video models not available, using simple features")
                return None
        return None
    
    def extract_features(self, video: torch.Tensor) -> np.ndarray:
        """
        Extract features from a video tensor.
        
        Args:
            video: Video tensor (C, F, H, W) or (B, C, F, H, W) in [0, 1].
            
        Returns:
            Feature vector as numpy array.
        """
        if self.model is None:
            # Fallback: use simple statistics as features
            return self._simple_features(video)
        
        with torch.no_grad():
            if video.dim() == 4:
                video = video.unsqueeze(0)  # Add batch dim
            
            # R3D expects (B, C, F, H, W) with F >= 16
            video = video.to(self.device)
            
            # Resize to model's expected input
            if video.shape[3] != 112 or video.shape[4] != 112:
                video = torch.nn.functional.interpolate(
                    video.flatten(0, 1).unsqueeze(0),
                    size=(112, 112),
                    mode="bilinear",
                ).reshape(video.shape[0], video.shape[1], video.shape[2], 112, 112)
            
            features = self.model(video)
            return features.cpu().numpy()
    
    def _simple_features(self, video: torch.Tensor) -> np.ndarray:
        """Simple feature extraction fallback using statistics."""
        if video.dim() == 5:
            video = video[0]
        
        features = []
        # Spatial statistics per frame
        for f in range(video.shape[1]):
            frame = video[:, f]
            features.extend([
                frame.mean().item(),
                frame.std().item(),
                frame.max().item(),
                frame.min().item(),
            ])
        
        # Temporal statistics
        diff = video[:, 1:] - video[:, :-1]
        features.extend([
            diff.mean().item(),
            diff.std().item(),
            diff.abs().mean().item(),
        ])
        
        return np.array(features)
    
    def compute_fid(
        self,
        features_real: np.ndarray,
        features_gen: np.ndarray,
    ) -> float:
        """
        Compute FID between two sets of features.
        
        FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
        
        Args:
            features_real: Real video features (N, D).
            features_gen: Generated video features (N, D).
            
        Returns:
            FID score (lower is better).
        """
        from scipy.linalg import sqrtm
        
        mu_r = np.mean(features_real, axis=0)
        mu_g = np.mean(features_gen, axis=0)
        
        sigma_r = np.cov(features_real, rowvar=False)
        sigma_g = np.cov(features_gen, rowvar=False)
        
        # Ensure matrices are 2D
        if sigma_r.ndim < 2:
            sigma_r = np.array([[sigma_r]])
        if sigma_g.ndim < 2:
            sigma_g = np.array([[sigma_g]])
        
        diff = mu_r - mu_g
        
        # Product might be almost singular
        covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
        
        return float(fid)


class FVDCalculator:
    """
    Fréchet Video Distance.
    
    Similar to FID-VID but uses video-specific features that capture
    both spatial and temporal characteristics.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fid_calc = FIDVIDCalculator(device=device)
    
    def compute(
        self,
        real_videos: List[torch.Tensor],
        generated_videos: List[torch.Tensor],
    ) -> float:
        """
        Compute FVD between sets of real and generated videos.
        
        Args:
            real_videos: List of real video tensors.
            generated_videos: List of generated video tensors.
            
        Returns:
            FVD score (lower is better).
        """
        # Extract features for all videos
        real_features = np.array([
            self.fid_calc.extract_features(v) for v in tqdm(real_videos, desc="Real features")
        ])
        gen_features = np.array([
            self.fid_calc.extract_features(v) for v in tqdm(generated_videos, desc="Gen features")
        ])
        
        # Handle different feature shapes
        if real_features.ndim > 2:
            real_features = real_features.reshape(len(real_videos), -1)
        if gen_features.ndim > 2:
            gen_features = gen_features.reshape(len(generated_videos), -1)
        
        return self.fid_calc.compute_fid(real_features, gen_features)


class DynamicDegreeCalculator:
    """
    Dynamic Degree metric.
    
    Measures the amount of motion/dynamics in a video.
    Higher values indicate more dynamic content.
    
    Computed as the average optical flow magnitude across frames.
    """
    
    def __init__(self):
        pass
    
    def compute(self, video: torch.Tensor) -> float:
        """
        Compute dynamic degree for a single video.
        
        Uses frame-to-frame differences as a proxy for motion.
        
        Args:
            video: Video tensor (C, F, H, W) in [0, 1].
            
        Returns:
            Dynamic degree score (higher = more motion).
        """
        if video.dim() == 5:
            video = video[0]
        
        # Compute frame differences
        frame_diffs = video[:, 1:] - video[:, :-1]
        
        # L2 norm of differences per pixel, averaged
        motion_magnitude = torch.sqrt((frame_diffs ** 2).sum(dim=0))  # (F-1, H, W)
        dynamic_degree = motion_magnitude.mean().item()
        
        return dynamic_degree
    
    def compute_optical_flow(self, video: torch.Tensor) -> float:
        """
        Compute dynamic degree using optical flow (more accurate but slower).
        
        Requires OpenCV.
        
        Args:
            video: Video tensor (C, F, H, W) in [0, 1].
            
        Returns:
            Dynamic degree based on optical flow magnitude.
        """
        try:
            import cv2
        except ImportError:
            return self.compute(video)  # Fallback
        
        if video.dim() == 5:
            video = video[0]
        
        # Convert to grayscale frames
        frames = video.permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C)
        frames = (frames * 255).astype(np.uint8)
        
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_magnitudes.append(magnitude.mean())
        
        return float(np.mean(flow_magnitudes))


def evaluate_single(
    reference_path: str,
    generated_path: str,
    metrics: List[str] = ["dynamic_degree"],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate a single reference-generated pair.
    
    Args:
        reference_path: Path to reference video.
        generated_path: Path to generated video.
        metrics: List of metrics to compute.
        device: Computation device.
        
    Returns:
        Dictionary of metric scores.
    """
    from pflow.video_utils import load_video
    
    ref = load_video(reference_path, device=device)
    gen = load_video(generated_path, device=device)
    
    results = {}
    
    if "dynamic_degree" in metrics:
        dd_calc = DynamicDegreeCalculator()
        results["dynamic_degree_ref"] = dd_calc.compute(ref)
        results["dynamic_degree_gen"] = dd_calc.compute(gen)
        results["dynamic_degree_ratio"] = (
            results["dynamic_degree_gen"] / max(results["dynamic_degree_ref"], 1e-8)
        )
    
    return results


def evaluate_batch(
    reference_dir: str,
    generated_dir: str,
    metrics: List[str] = ["fid_vid", "fvd", "dynamic_degree"],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate a batch of generated videos against references.
    
    Args:
        reference_dir: Directory of reference videos.
        generated_dir: Directory of generated videos.
        metrics: Metrics to compute.
        device: Computation device.
        
    Returns:
        Dictionary of averaged metric scores.
    """
    from pflow.video_utils import load_video
    
    ref_files = sorted(Path(reference_dir).glob("*.mp4"))
    gen_files = sorted(Path(generated_dir).glob("*.mp4"))
    
    print(f"Found {len(ref_files)} reference and {len(gen_files)} generated videos.")
    
    # Load videos
    ref_videos = [load_video(str(f), device=device) for f in tqdm(ref_files, desc="Loading refs")]
    gen_videos = [load_video(str(f), device=device) for f in tqdm(gen_files, desc="Loading gens")]
    
    results = {}
    
    # FID-VID
    if "fid_vid" in metrics:
        print("Computing FID-VID...")
        fid_calc = FIDVIDCalculator(device=device)
        ref_features = np.array([fid_calc.extract_features(v) for v in ref_videos])
        gen_features = np.array([fid_calc.extract_features(v) for v in gen_videos])
        
        if ref_features.ndim > 2:
            ref_features = ref_features.reshape(len(ref_videos), -1)
        if gen_features.ndim > 2:
            gen_features = gen_features.reshape(len(gen_videos), -1)
        
        results["fid_vid"] = fid_calc.compute_fid(ref_features, gen_features)
    
    # FVD
    if "fvd" in metrics:
        print("Computing FVD...")
        fvd_calc = FVDCalculator(device=device)
        results["fvd"] = fvd_calc.compute(ref_videos, gen_videos)
    
    # Dynamic Degree
    if "dynamic_degree" in metrics:
        print("Computing Dynamic Degree...")
        dd_calc = DynamicDegreeCalculator()
        ref_dd = [dd_calc.compute(v) for v in ref_videos]
        gen_dd = [dd_calc.compute(v) for v in gen_videos]
        results["dynamic_degree_ref_mean"] = float(np.mean(ref_dd))
        results["dynamic_degree_gen_mean"] = float(np.mean(gen_dd))
        results["dynamic_degree_ratio_mean"] = float(
            np.mean([g / max(r, 1e-8) for g, r in zip(gen_dd, ref_dd)])
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate P-Flow generated videos")
    
    parser.add_argument("--reference_dir", type=str, help="Directory of reference videos.")
    parser.add_argument("--generated_dir", type=str, help="Directory of generated videos.")
    parser.add_argument("--reference_video", type=str, help="Single reference video path.")
    parser.add_argument("--generated_video", type=str, help="Single generated video path.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["dynamic_degree"],
        choices=["fid_vid", "fvd", "dynamic_degree"],
        help="Metrics to compute.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file.")
    
    args = parser.parse_args()
    
    if args.reference_video and args.generated_video:
        results = evaluate_single(
            args.reference_video,
            args.generated_video,
            metrics=args.metrics,
            device=args.device,
        )
    elif args.reference_dir and args.generated_dir:
        results = evaluate_batch(
            args.reference_dir,
            args.generated_dir,
            metrics=args.metrics,
            device=args.device,
        )
    else:
        parser.error("Provide either --reference_video/--generated_video or --reference_dir/--generated_dir")
    
    # Print results
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
