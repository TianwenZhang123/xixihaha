"""
Three-Layer Progressive Motion Asset: Packaging, Storage, and Application.

Information-Theoretic Perspective (BCD, ICCV 2025):
    The three-layer motion asset implements a progressive refinement encoding
    that follows rate-distortion optimal information allocation:

    Layer 1 - Motion Text (Discrete Tokens):
        Rate: ~50-100 bits (10-20 tokens)
        Distortion: Captures only coarse motion semantics
        Portability: Cross-model, cross-architecture, human-editable
        Example: "dancing hip-hop with fast arm movements left to right"

    Layer 2 - Motion Embedding delta_e (Continuous Vector):
        Rate: ~10K-50K bits (L*D float16 values)
        Distortion: Captures mid-frequency motion details
        Portability: Same text-encoder family (T5-based models)
        Example: Acceleration curves, rhythmic patterns, phase relationships

    Layer 3 - Motion Noise Prior eta_motion (Latent Tensor):
        Rate: ~1M+ bits (C*F*H*W float16 values)
        Distortion: Captures full structural motion detail
        Portability: Same architecture only (Wan2.1 family)
        Example: Precise spatial trajectories, sub-frame timing

    The optimal information allocation follows progressive refinement:
    each layer encodes the RESIDUAL information that previous layers cannot
    capture. This is analogous to progressive JPEG encoding where each
    refinement pass adds finer detail.

Connection to Spectral Autoregression:
    The three layers correspond to different spectral bands:
    - Text: lowest frequency (action category, gross direction)
    - Embedding: mid frequency (timing, acceleration, rhythm)
    - Noise: highest frequency (precise spatial structure)

    This spectral correspondence is not coincidental - it reflects the
    fundamental structure of how diffusion models generate motion.

Asset Operations:
    - create: Package optimization results into three-layer asset
    - save: Serialize to disk (JSON metadata + .pt tensors)
    - load: Deserialize from disk
    - apply: Inject three-layer guidance into new generation
    - blend: Linear interpolation of assets (RichSpace linearity)
    - scale: Adjust motion intensity

References:
    - BCD (ICCV 2025): Bitrate-controlled disentanglement
    - RichSpace (ICLR 2025): T5 embedding space local linearity
    - Textual Inversion (ICLR 2023): Embedding asset storage pattern
    - Progressive JPEG: Rate-distortion optimal layered encoding
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch

logger = logging.getLogger(__name__)


@dataclass
class MotionAsset:
    """
    Three-Layer Progressive Motion Asset.

    Implements the rate-distortion optimal motion representation:
    each layer provides progressively finer motion information at
    increasing bitrate cost and decreasing portability.

    The three layers are complementary, not redundant:
    - Layer 1 alone: coarse motion transfer (cross-model)
    - Layer 1+2: precise motion transfer (same encoder family)
    - Layer 1+2+3: full-fidelity motion reproduction (same architecture)
    """
    # === Layer 1: Interpretable Motion Text (lowest bitrate, highest portability) ===
    motion_text: str = ""
    motion_tokens: List[str] = field(default_factory=list)

    # === Layer 2: Semantic Motion Embedding (medium bitrate, medium portability) ===
    delta_e: Optional[torch.Tensor] = None

    # === Layer 3: Structural Motion Prior (highest bitrate, lowest portability) ===
    eta_motion: Optional[torch.Tensor] = None

    # === Metadata ===
    source_caption: str = ""
    source_video: str = ""
    extraction_params: Dict[str, Any] = field(default_factory=dict)

    # === Asset Properties ===
    version: str = "2.0"  # v2.0: three-layer progressive encoding
    extraction_model: str = "wan2.1-1.3b"
    text_encoder: str = "umt5-xxl"
    motion_type: str = ""
    intensity: float = 0.0
    duration_frames: int = 81
    compatible_models: list = field(default_factory=lambda: ["wan2.1-1.3b", "wan2.1-14b"])

    # === Layer Quality Metrics ===
    token_decode_score: float = 0.0  # Velocity preservation of Layer 1
    embedding_norm: float = 0.0      # ||delta_e|| - Layer 2 information content
    noise_energy: float = 0.0        # ||eta_motion|| - Layer 3 information content


class MotionAssetManager:
    """
    Three-Layer Motion Asset Manager.

    Manages the complete lifecycle of progressive motion assets:
    - Packaging: Create asset from optimization results
    - Persistence: Save/load with format versioning
    - Application: Three-layer injection into new generation
    - Composition: Linear blending based on RichSpace theory
    - Analysis: Per-layer contribution diagnostics
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    # =========================================================================
    # Asset Creation
    # =========================================================================

    def create_asset(
        self,
        delta_e: torch.Tensor,
        eta_motion: torch.Tensor,
        motion_text: str = "",
        source_caption: str = "",
        source_video: str = "",
        extraction_params: Optional[Dict[str, Any]] = None,
    ) -> MotionAsset:
        """
        Create a three-layer progressive motion asset from optimization results.

        Packages the outputs of the Iterative Spectral Refinement pipeline
        into a portable, reusable motion asset with quality metrics.

        Args:
            delta_e: Optimized motion embedding (B, L, D) or (L, D)
            eta_motion: Spectral motion prior (B, C, F, H, W) or (C, F, H, W)
            motion_text: Decoded motion text description (Layer 1)
            source_caption: Original video caption
            source_video: Source video path
            extraction_params: Hyperparameters used during extraction

        Returns:
            MotionAsset instance with all three layers populated
        """
        # Remove batch dimension if present
        if delta_e.dim() == 3 and delta_e.shape[0] == 1:
            delta_e = delta_e.squeeze(0)
        if eta_motion.dim() == 5 and eta_motion.shape[0] == 1:
            eta_motion = eta_motion.squeeze(0)

        # Extract motion tokens from params if available
        params = extraction_params or {}
        motion_tokens = params.pop("motion_tokens", [])
        token_score = params.pop("token_decode_score", 0.0)

        asset = MotionAsset(
            # Layer 1
            motion_text=motion_text,
            motion_tokens=motion_tokens if isinstance(motion_tokens, list) else [],
            # Layer 2
            delta_e=delta_e.detach().cpu(),
            # Layer 3
            eta_motion=eta_motion.detach().cpu(),
            # Metadata
            source_caption=source_caption,
            source_video=source_video,
            extraction_params=params,
            # Metrics
            intensity=delta_e.norm().item(),
            embedding_norm=delta_e.norm().item(),
            noise_energy=eta_motion.norm().item() if eta_motion.numel() > 1 else 0,
            token_decode_score=token_score,
            duration_frames=eta_motion.shape[1] if eta_motion.dim() == 4 else (
                eta_motion.shape[0] if eta_motion.dim() >= 1 else 81
            ),
        )

        logger.info(
            f"  [Asset] Created three-layer asset:"
            f" L1(text)='{motion_text[:30]}...',"
            f" L2(||delta_e||)={asset.embedding_norm:.4f},"
            f" L3(||eta||)={asset.noise_energy:.4f}"
        )

        return asset

    # =========================================================================
    # Persistence (Save / Load)
    # =========================================================================

    def save(self, asset: MotionAsset, save_dir: str) -> str:
        """
        Save three-layer motion asset to disk.

        File structure:
            save_dir/
                asset.json          # Metadata + Layer 1 (text)
                delta_e_motion.pt   # Layer 2 (embedding tensor)
                eta_motion.pt       # Layer 3 (noise prior tensor)

        The JSON file contains all information needed for Layer 1 transfer
        (cross-model), while .pt files are needed for Layer 2/3 (same-family).
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save tensor files (Layer 2 & 3)
        delta_e_path = save_path / "delta_e_motion.pt"
        eta_path = save_path / "eta_motion.pt"

        if asset.delta_e is not None:
            torch.save(asset.delta_e, str(delta_e_path))
        if asset.eta_motion is not None:
            torch.save(asset.eta_motion, str(eta_path))

        # Save metadata JSON (includes Layer 1 text)
        meta = {
            "version": asset.version,
            "asset_type": "three_layer_progressive_motion_asset",
            "framework": "VMAD_Iterative_Spectral_Refinement",
            "extraction_model": asset.extraction_model,
            "text_encoder": asset.text_encoder,
            # Layer 1: Interpretable
            "layer1_text": {
                "motion_text": asset.motion_text,
                "motion_tokens": asset.motion_tokens,
                "token_decode_score": asset.token_decode_score,
            },
            # Layer 2: Semantic
            "layer2_embedding": {
                "path": "delta_e_motion.pt",
                "norm": asset.embedding_norm,
                "shape": list(asset.delta_e.shape) if asset.delta_e is not None else None,
            },
            # Layer 3: Structural
            "layer3_noise": {
                "path": "eta_motion.pt",
                "energy": asset.noise_energy,
                "shape": list(asset.eta_motion.shape) if asset.eta_motion is not None else None,
            },
            # Source info
            "source": {
                "caption": asset.source_caption,
                "video": asset.source_video,
            },
            # Metadata
            "metadata": {
                "motion_type": asset.motion_type,
                "intensity": asset.intensity,
                "duration_frames": asset.duration_frames,
                "compatible_models": asset.compatible_models,
                "extraction_params": asset.extraction_params,
            },
        }

        json_path = save_path / "asset.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"  [Asset] Saved to: {save_dir}")
        return str(save_path)

    def load(self, load_dir: str) -> MotionAsset:
        """
        Load three-layer motion asset from disk.

        Supports both v1.0 (legacy) and v2.0 (three-layer) format.
        """
        load_path = Path(load_dir)

        # Load metadata
        json_path = load_path / "asset.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Asset metadata not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Load tensors
        delta_e = None
        eta_motion = None

        # Handle both v1 and v2 format
        if "layer2_embedding" in meta:
            # v2.0 format
            delta_e_file = load_path / meta["layer2_embedding"].get("path", "delta_e_motion.pt")
        else:
            # v1.0 legacy format
            delta_e_file = load_path / meta.get("motion_token_path", "delta_e_motion.pt")

        if delta_e_file.exists():
            delta_e = torch.load(str(delta_e_file), map_location="cpu", weights_only=True)

        if "layer3_noise" in meta:
            eta_file = load_path / meta["layer3_noise"].get("path", "eta_motion.pt")
        else:
            eta_file = load_path / meta.get("noise_prior_path", "eta_motion.pt")

        if eta_file.exists():
            eta_motion = torch.load(str(eta_file), map_location="cpu", weights_only=True)

        # Extract layer 1 info
        if "layer1_text" in meta:
            motion_text = meta["layer1_text"].get("motion_text", "")
            motion_tokens = meta["layer1_text"].get("motion_tokens", [])
            token_score = meta["layer1_text"].get("token_decode_score", 0.0)
        else:
            motion_text = meta.get("motion_text", "")
            motion_tokens = []
            token_score = 0.0

        # Extract source info
        if "source" in meta:
            source_caption = meta["source"].get("caption", "")
            source_video = meta["source"].get("video", "")
        else:
            source_caption = meta.get("source_caption", "")
            source_video = meta.get("source_video", "")

        # Extract metadata
        metadata = meta.get("metadata", {})

        asset = MotionAsset(
            motion_text=motion_text,
            motion_tokens=motion_tokens,
            delta_e=delta_e,
            eta_motion=eta_motion,
            source_caption=source_caption,
            source_video=source_video,
            extraction_params=metadata.get("extraction_params", {}),
            version=meta.get("version", "1.0"),
            extraction_model=meta.get("extraction_model", "wan2.1-1.3b"),
            text_encoder=meta.get("text_encoder", "umt5-xxl"),
            motion_type=metadata.get("motion_type", ""),
            intensity=metadata.get("intensity", 0.0),
            duration_frames=metadata.get("duration_frames", 81),
            compatible_models=metadata.get("compatible_models", []),
            token_decode_score=token_score,
            embedding_norm=delta_e.norm().item() if delta_e is not None else 0,
            noise_energy=eta_motion.norm().item() if eta_motion is not None else 0,
        )

        logger.info(
            f"  [Asset] Loaded from: {load_dir} (v{asset.version}),"
            f" L1='{asset.motion_text[:30]}',"
            f" L2={asset.embedding_norm:.4f},"
            f" L3={asset.noise_energy:.4f}"
        )

        return asset

    # =========================================================================
    # Application (Three-Layer Injection)
    # =========================================================================

    def apply_to_embedding(
        self,
        content_embedding: torch.Tensor,
        asset: MotionAsset,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Layer 2 Application: Inject motion embedding into content.

        e_final = e_content + strength * delta_e

        Based on RichSpace (ICLR 2025) local linearity property:
        linear operations in T5 embedding space preserve semantic coherence,
        making additive injection theoretically grounded.

        Args:
            content_embedding: New content prompt embedding (B, L, D)
            asset: Motion asset containing delta_e
            strength: Motion intensity [0, 1+]

        Returns:
            e_final: Combined embedding (B, L, D)
        """
        if asset.delta_e is None:
            logger.warning("Asset has no delta_e (Layer 2), returning content unchanged")
            return content_embedding

        delta_e = asset.delta_e.to(
            device=content_embedding.device,
            dtype=content_embedding.dtype,
        )

        # Handle dimension matching
        if delta_e.dim() == 2 and content_embedding.dim() == 3:
            delta_e = delta_e.unsqueeze(0).expand_as(content_embedding)

        e_final = content_embedding + strength * delta_e
        return e_final

    def apply_noise_prior(
        self,
        asset: MotionAsset,
        alpha: float = 0.001,
        strength: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        """
        Layer 3 Application: Minimal motion prior injection.

        eta_init = sqrt(alpha * strength) * eta_motion + sqrt(1 - alpha * strength) * eta_random

        The extremely small alpha (0.001) is theoretically motivated:
        according to spectral autoregression, the noise prior encodes the
        LOWEST frequency motion structure. Even minimal injection provides
        sufficient structural guidance because low-frequency components
        dominate early denoising steps where motion layout is determined.

        Args:
            asset: Motion asset containing eta_motion
            alpha: Base blending weight (default 0.001 = minimal injection)
            strength: Motion intensity scaling
            generator: Random number generator for reproducibility

        Returns:
            eta_init: Blended initial noise, or None if no noise prior
        """
        if asset.eta_motion is None:
            logger.warning("Asset has no eta_motion (Layer 3), returning None")
            return None

        eta_motion = asset.eta_motion.to(device=self.device)

        # Scale alpha by strength
        alpha_scaled = min(alpha * strength, 1.0)

        # Generate random noise
        eta_random = torch.randn(
            eta_motion.shape,
            dtype=eta_motion.dtype,
            device=self.device,
            generator=generator,
        )

        # Blend: preserves unit variance when alpha is small
        eta_init = (
            (alpha_scaled ** 0.5) * eta_motion
            + ((1 - alpha_scaled) ** 0.5) * eta_random
        )

        # Ensure 5D (B, C, F, H, W) for WanPipeline compatibility
        if eta_init.dim() == 4:
            eta_init = eta_init.unsqueeze(0)

        return eta_init

    # =========================================================================
    # Composition (Linear Operations in Embedding Space)
    # =========================================================================

    def blend_assets(
        self,
        asset_a: MotionAsset,
        asset_b: MotionAsset,
        weight_a: float = 0.5,
    ) -> MotionAsset:
        """
        Blend two motion assets via linear interpolation.

        Based on RichSpace (ICLR 2025): T5 embedding space exhibits local
        linearity, meaning linear interpolation between motion embeddings
        produces semantically meaningful intermediate motions.

        delta_e_blend = w_a * delta_e_A + w_b * delta_e_B
        eta_blend = w_a * eta_A + w_b * eta_B

        This enables motion composition: e.g., blending "walking" and "waving"
        to get "walking while waving".

        Args:
            asset_a: First motion asset
            asset_b: Second motion asset
            weight_a: Weight for asset A (asset B weight = 1 - weight_a)

        Returns:
            Blended motion asset
        """
        weight_b = 1.0 - weight_a

        # Blend Layer 2 (embedding)
        delta_e_mixed = None
        if asset_a.delta_e is not None and asset_b.delta_e is not None:
            delta_e_mixed = weight_a * asset_a.delta_e + weight_b * asset_b.delta_e

        # Blend Layer 3 (noise prior)
        eta_mixed = None
        if asset_a.eta_motion is not None and asset_b.eta_motion is not None:
            eta_mixed = weight_a * asset_a.eta_motion + weight_b * asset_b.eta_motion

        # Blend Layer 1 (text - concatenation with weights)
        text_mixed = (
            f"[{weight_a:.1f}x] {asset_a.motion_text} + "
            f"[{weight_b:.1f}x] {asset_b.motion_text}"
        )

        return MotionAsset(
            motion_text=text_mixed,
            motion_tokens=asset_a.motion_tokens + asset_b.motion_tokens,
            delta_e=delta_e_mixed,
            eta_motion=eta_mixed,
            source_caption=f"blend({asset_a.source_caption}, {asset_b.source_caption})",
            extraction_params={"blend_weight_a": weight_a},
            intensity=(delta_e_mixed.norm().item() if delta_e_mixed is not None else 0),
            embedding_norm=(delta_e_mixed.norm().item() if delta_e_mixed is not None else 0),
            noise_energy=(eta_mixed.norm().item() if eta_mixed is not None else 0),
        )

    def scale_asset(self, asset: MotionAsset, scale: float) -> MotionAsset:
        """
        Scale motion asset intensity.

        Applies uniform scaling to Layer 2 and Layer 3:
            delta_e_scaled = scale * delta_e
            eta_scaled = scale * eta_motion

        Note: For Layer 3, scaling changes the signal-to-noise ratio
        in the blended initialization. Larger scale = stronger structural
        guidance from the motion prior.

        Args:
            asset: Original motion asset
            scale: Intensity factor (0.5=half, 2.0=double)

        Returns:
            Scaled motion asset
        """
        delta_e_scaled = None
        if asset.delta_e is not None:
            delta_e_scaled = scale * asset.delta_e

        eta_scaled = None
        if asset.eta_motion is not None:
            eta_scaled = scale * asset.eta_motion

        return MotionAsset(
            motion_text=f"[{scale:.1f}x intensity] {asset.motion_text}",
            motion_tokens=asset.motion_tokens,
            delta_e=delta_e_scaled,
            eta_motion=eta_scaled,
            source_caption=asset.source_caption,
            source_video=asset.source_video,
            extraction_params={**asset.extraction_params, "scale": scale},
            intensity=(delta_e_scaled.norm().item() if delta_e_scaled is not None else 0),
            embedding_norm=(delta_e_scaled.norm().item() if delta_e_scaled is not None else 0),
            noise_energy=(eta_scaled.norm().item() if eta_scaled is not None else 0),
            duration_frames=asset.duration_frames,
        )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def analyze_layer_information(self, asset: MotionAsset) -> Dict[str, Any]:
        """
        Analyze information content of each layer.

        Provides metrics for validating the progressive refinement hypothesis:
        each layer should contain non-redundant information at different
        spectral resolutions.

        Returns:
            Per-layer information metrics
        """
        analysis = {
            "layer1_text": {
                "num_tokens": len(asset.motion_tokens),
                "text_length": len(asset.motion_text),
                "estimated_bits": len(asset.motion_text.encode("utf-8")) * 8,
                "decode_score": asset.token_decode_score,
            },
            "layer2_embedding": {
                "norm": asset.embedding_norm,
                "shape": list(asset.delta_e.shape) if asset.delta_e is not None else None,
                "estimated_bits": (
                    asset.delta_e.numel() * 16 if asset.delta_e is not None else 0
                ),  # float16
                "sparsity": (
                    (asset.delta_e.abs() < 1e-6).float().mean().item()
                    if asset.delta_e is not None else 0
                ),
            },
            "layer3_noise": {
                "energy": asset.noise_energy,
                "shape": list(asset.eta_motion.shape) if asset.eta_motion is not None else None,
                "estimated_bits": (
                    asset.eta_motion.numel() * 16 if asset.eta_motion is not None else 0
                ),
            },
            "total_estimated_bits": 0,
            "compression_ratio": 0,
        }

        total_bits = (
            analysis["layer1_text"]["estimated_bits"]
            + analysis["layer2_embedding"]["estimated_bits"]
            + analysis["layer3_noise"]["estimated_bits"]
        )
        analysis["total_estimated_bits"] = total_bits

        # Compression ratio vs raw video (480x832x81x3x8 bits)
        raw_video_bits = 480 * 832 * 81 * 3 * 8
        analysis["compression_ratio"] = raw_video_bits / (total_bits + 1) if total_bits > 0 else 0

        return analysis
