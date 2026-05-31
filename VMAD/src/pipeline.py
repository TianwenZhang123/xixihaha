"""
VMAD Pipeline — Video Prompt Inversion via Three-Layer Progressive Encoding.

Core Problem: Video Prompt Inversion
    Given a reference video V_ref and a frozen text-to-video model v_θ, find the
    optimal input conditioning (text, embedding, noise) such that the model
    faithfully reproduces the reference video. This is the inverse problem of
    text-to-video generation: instead of "text → video", we solve "video → prompt".

    The key insight is that a single text prompt cannot fully specify a video —
    there is an enormous information gap between what text can express (~100 bits)
    and what a video contains (~10M bits). VMAD bridges this gap through three
    complementary layers of progressive encoding.

Three-Layer Progressive Encoding (Information-Theoretic View):
    Each layer encodes a different spectral band of the video's information,
    motivated by the spectral autoregression property of flow matching models:

    Layer 1 - Text (~100 bits, highest portability):
        Human-readable prompt optimized via LLM rewriting (Subject-First +
        Temporal Action Chain). Cross-model transferable, user-editable.
        Captures: subject identity, scene layout, action category.

    Layer 2 - Embedding Δe (~100K bits, CORE CONTRIBUTION):
        Continuous embedding residual optimized via velocity field matching:
            min_{Δe} E_t[||v_θ(x_t, t, e₀+Δe) - v*||²]
        Captures: timing, acceleration, rhythm, phase, style details.
        This is what text CANNOT express but the model CAN understand.

    Layer 3 - Noise Prior η_inv (~1M bits, lowest portability):
        Inverted initial noise via Flow Matching Inversion (ODE reversal).
        Captures: exact spatial structure, pixel-level layout, fine texture.
        Provides structural guidance during earliest denoising steps.

    Together: Text provides semantic scaffold → Δe adds dynamic precision →
    η_inv locks spatial structure. Each layer encodes the RESIDUAL of the previous.

Architecture: Extract + Apply (builds on P-Flow)
    Extract: V_ref → [Inversion + VelocityMatch + LLM Rewrite] → VideoAsset
        - Phase A: Flow Matching Inversion → η_inv (Layer 3)
        - Phase B: Velocity Field Matching → Δe (Layer 2)
        - Phase C: LLM Caption Rewriting → optimized text (Layer 1)
        NOTE: Extract phase does NOT require video generation by default.
              VLM text decode (which generates comparison videos) is OPTIONAL.

    Apply: VideoAsset + prompt → [TextFusion + EmbedInject + NoisePrior] → V_new
        - Layer 1: content_prompt + motion_text fusion
        - Layer 2: e_content + strength * Δe injection
        - Layer 3: noise prior blending with η_inv

    Relationship to P-Flow:
        P-Flow = Layer 1 only (LLM rewriting, iterative VLM feedback)
        VMAD = P-Flow + Layer 2 (Δe) + Layer 3 (η_inv)
        VMAD's Layer 1 reuses P-Flow's proven V4 rewriting strategy.

Module Flags:
    --inversion     Flow Matching Inversion (video → noise, Layer 3)
    --spectral      Spectral Decomposition (noise → motion prior, Layer 3)
    --blend         Noise Prior Injection (motion prior → initial latent, Layer 3)
    --velocity      Position-Aware Velocity Field Matching (core, Layer 2)
    --disentangle   Cross-Content Consistency Regularization (Layer 2, motion transfer only)
    --token_decode  Velocity-Preserving Token Decoding (Δe → tokens, Layer 1↔2 bridge)
    --text_decode   VLM Comparative Text Decoding (OPTIONAL, requires video generation)

References:
    - Dieleman (2024): Spectral autoregression in diffusion models
    - Reenact Anything (SIGGRAPH 2025): Motion-textual inversion
    - FlowEdit (ICCV 2025): Inversion-free editing via velocity interpolation
    - RF-Solver (ICML 2025): High-order flow matching inversion
    - SiD-DiT (Apple 2025): Score identity distillation for flow matching
    - TPSO (2025): Token-Prompt dual-space optimization
"""

import os
import json
import time
import inspect
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field

import torch

from .distributed import setup_single_gpu, load_model_single_gpu, cleanup_gpu_memory
from .flow_matching import FlowMatchingInverter, encode_video_to_latents
from .spectral_decomposition import SpectralMotionDecomposer
from .velocity_matching import PositionAwareVelocityMatcher
from .token_decoder import VelocityPreservingTokenDecoder
from .content_augmentation import ContentAugmenter
from .motion_asset import MotionAssetManager, MotionAsset
from .vlm_client import create_vlm_client
from .video_utils import (
    load_video, save_video_tensor, normalize_video, denormalize_video,
)

logger = logging.getLogger(__name__)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, work, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, "
    "deformed, blurry, watermark"
)


@dataclass
class VMADConfig:
    """
    VMAD Configuration — Video Prompt Inversion Parameters.

    Organized by spectral layer to reflect the three-layer architecture:
    - Layer 3 (Structural): Inversion + Spectral Decomposition + Blending
    - Layer 2 (Semantic): Velocity Matching + Disentanglement + Position-Aware
    - Layer 1 (Interpretable): LLM Rewriting + Token Decoding
    """

    # -- Model --
    t2v_path: str = "/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers"
    dtype: str = "bfloat16"

    # -- Video Generation --
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 15
    guidance_scale: float = 5.0
    num_inference_steps: int = 30

    # -- Layer Activation Flags --
    use_inversion: bool = True       # Prerequisite for Layer 3
    use_spectral: bool = True        # Layer 3: Spectral decomposition
    use_blend: bool = True           # Layer 3: Noise prior injection
    use_velocity: bool = True        # Layer 2: Velocity field matching
    use_disentangle: bool = True     # Layer 2: Content disentanglement
    use_position_aware: bool = True  # Layer 2: Position-aware optimization
    use_token_decode: bool = True    # Layer 1<->2: Token decoding bridge
    use_text_decode: bool = False    # Layer 1: VLM text decoding (OPTIONAL, generates 2 videos!)
    use_midpoint: bool = False       # Inversion solver order

    # -- Layer 3: Structural Prior (Noise Space) --
    inversion_steps: int = 50        # ODE integration steps
    rho_s: float = 0.1              # Spatial filtering (P-Flow: 0.1 removes appearance priors)
    rho_m: float = 0.9              # Temporal retention (P-Flow: 0.9 keeps motion dynamics)
    alpha: float = 0.001            # Noise prior blend weight (P-Flow: 0.001 = minimal injection)

    # -- Layer 2: Semantic Embedding (Token Space) --
    T_m: float = 1.0               # Full timestep range for reproduction (0.3 for motion transfer)
    num_opt_steps: int = 100        # Optimization iterations
    opt_lr: float = 1e-3            # Peak learning rate
    lambda_dis: float = 0.0         # Disentanglement (0=reproduction, 0.1=motion transfer)
    lambda_pos: float = 0.01        # Position regularization weight
    dis_every_n_steps: int = 10     # Disentanglement computation frequency
    warmup_ratio: float = 0.2       # Warmup fraction (no disentanglement)
    num_augmentations: int = 5      # Cross-content augmentation count

    # -- Layer 1: Interpretable Tokens (Discrete Space) --
    token_decode_top_k: int = 50    # NN projection candidates
    token_decode_beam: int = 5      # Velocity reranking beam width
    token_decode_max_tokens: int = 10  # Maximum motion tokens

    # -- VLM --
    vlm_provider: str = "local"
    vlm_model_path: str = "/root/models/Qwen2.5-VL-7B-Instruct"

    # -- Content Augmentation --
    augmentation_provider: str = "mock"

    # -- Application --
    default_strength: float = 1.0

    # -- Runtime --
    seed: int = 42
    save_intermediates: bool = True

    def active_flags(self) -> List[str]:
        """Return list of active module flags."""
        flags = []
        if self.use_inversion:
            flags.append("inversion")
        if self.use_spectral:
            flags.append("spectral")
        if self.use_blend:
            flags.append("blend")
        if self.use_velocity:
            flags.append("velocity")
        if self.use_disentangle:
            flags.append("disentangle")
        if self.use_position_aware:
            flags.append("position_aware")
        if self.use_token_decode:
            flags.append("token_decode")
        if self.use_text_decode:
            flags.append("text_decode")
        if self.use_midpoint:
            flags.append("midpoint")
        return flags

    def experiment_name(self) -> str:
        """Generate experiment identifier."""
        flags = self.active_flags()
        if not flags:
            return "baseline"
        return "vmad_" + "_".join(flags)


class VMADPipeline:
    """
    VMAD Pipeline: Video Prompt Inversion via Three-Layer Progressive Encoding.

    Solves the inverse problem: given a reference video, find optimal conditioning
    (text + embedding + noise) that makes a frozen T2V model reproduce it faithfully.

    Core workflow (Extract) — NO video generation required by default:
        Phase A (Layer 3): Flow Matching Inversion → η_inv
        Phase B (Layer 2): Position-Aware Velocity Matching → Δe
        Phase C (Layer 1): Token Decoding (optional) + LLM Rewriting
        Package: (Δe, η_inv, optimized_text) → VideoAsset

    Core workflow (Apply):
        Layer 1: content_prompt + motion_text fusion
        Layer 2: e_content + strength * Δe injection
        Layer 3: noise prior blending with η_inv
        Generate: single T2V forward pass → output video

    Relationship to P-Flow:
        P-Flow handles Layer 1 (text optimization via LLM + VLM iteration).
        VMAD adds Layer 2 (Δe via velocity matching) and Layer 3 (η_inv via inversion).
        Combined, the three layers provide progressively finer video reproduction.
    """

    def __init__(self, config: VMADConfig):
        self.config = config
        self.device = setup_single_gpu()
        self.dtype = getattr(torch, config.dtype)

        # Lazy-loaded components
        self._pipe = None
        self._vlm_client = None
        self._augmenter = None
        self._token_decoder = None
        self._asset_manager = MotionAssetManager(device=self.device)

    @property
    def pipe(self):
        """Lazy load T2V pipeline."""
        if self._pipe is None:
            self._pipe = load_model_single_gpu(
                model_path=self.config.t2v_path,
                dtype=self.dtype,
            )
        return self._pipe

    @property
    def vlm_client(self):
        """Lazy load VLM client."""
        if self._vlm_client is None:
            self._vlm_client = create_vlm_client({
                "provider": self.config.vlm_provider,
                "model_path": self.config.vlm_model_path,
                "lazy_load": True,
            })
        return self._vlm_client

    @property
    def augmenter(self):
        """Lazy load content augmenter."""
        if self._augmenter is None:
            self._augmenter = ContentAugmenter(
                provider=self.config.augmentation_provider,
                num_augmentations=self.config.num_augmentations,
            )
        return self._augmenter

    @property
    def token_decoder(self):
        """Lazy load velocity-preserving token decoder."""
        if self._token_decoder is None:
            self._token_decoder = VelocityPreservingTokenDecoder(
                pipe=self.pipe,
                top_k=self.config.token_decode_top_k,
                beam_width=self.config.token_decode_beam,
                device=self.device,
            )
        return self._token_decoder

    # =========================================================================
    # Extract Mode: Iterative Spectral Refinement
    # =========================================================================

    def extract(
        self,
        video_path: str,
        output_dir: str,
        caption: str = "",
    ) -> Dict[str, Any]:
        """
        Extract video asset via Three-Layer Progressive Encoding.

        This is the core "Video Prompt Inversion" operation: given a reference
        video, compute the optimal conditioning that reproduces it. By default,
        NO video generation is performed during extraction — only mathematical
        optimization (inversion, velocity matching, token projection).

        Phases (each captures progressively finer information):
            Phase A (Layer 3 - Noise Space):
                1. Flow Matching Inversion: video → η_inv
                2. Spectral Decomposition: η_inv → η_motion (optional filtering)
            Phase B (Layer 2 - Embedding Space):
                3. Position-Aware Velocity Matching: (z₀, η_inv) → Δe
                4. Content Disentanglement: regularize Δe (motion transfer only)
            Phase C (Layer 1 - Token/Text Space):
                5. Token Decoding: Δe → motion tokens (optional)
                6. VLM Text Decoding: comparative description (OPTIONAL, costs 2 videos!)

        Args:
            video_path: Path to reference video
            output_dir: Output directory for extracted asset
            caption: Initial caption (from VLM or LLM rewrite; auto-generated if empty)

        Returns:
            Result dictionary with asset path, metrics, and diagnostics
        """
        t0 = time.time()
        cfg = self.config
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        seed = cfg.seed
        torch.manual_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        flags = cfg.active_flags()
        logger.info("[VMAD Extract] Video Prompt Inversion — Three-Layer Encoding")
        logger.info(f"  Active layers: {flags}")

        # -- Prerequisite: Load reference video --
        logger.info("  [Prep] Loading reference video...")
        ref_video = load_video(
            video_path,
            num_frames=cfg.num_frames,
            height=cfg.height,
            width=cfg.width,
            device=self.device,
        )

        # -- Prerequisite: Generate/use caption --
        if not caption:
            logger.info("  [Prep] Generating caption via VLM...")
            caption = self.vlm_client.describe_video(video_path)
            if not caption:
                caption = "a video scene with natural motion"
            logger.info(f"  Caption: {caption[:80]}...")

        (out / "caption.txt").write_text(caption, encoding="utf-8")

        # =================================================================
        # Phase A: Structural Layer (Noise Space - Highest Spectral Resolution)
        # Captures precise spatial trajectories and sub-frame timing
        # =================================================================
        eta_inv = None
        eta_motion = None

        if cfg.use_inversion:
            logger.info("  [Phase A] Structural Layer - Flow Matching Inversion...")
            eta_inv = self._compute_inversion(ref_video, caption)

            if cfg.save_intermediates:
                torch.save(eta_inv.cpu(), str(out / "eta_inv.pt"))

            if cfg.use_spectral:
                logger.info(
                    f"  [Phase A] Spectral Motion-Content Decomposition "
                    f"(rho_s={cfg.rho_s}, rho_m={cfg.rho_m})..."
                )
                decomposer = SpectralMotionDecomposer(
                    rho_s=cfg.rho_s, rho_m=cfg.rho_m
                )
                eta_motion = decomposer.decompose(eta_inv)

                if cfg.save_intermediates:
                    torch.save(eta_motion.cpu(), str(out / "eta_motion.pt"))

                    # Save spectral analysis
                    if eta_inv.dim() == 5:
                        analysis = decomposer.compute_spectral_energy_distribution(
                            eta_inv[0]
                        )
                    else:
                        analysis = decomposer.compute_spectral_energy_distribution(
                            eta_inv
                        )
                    with open(out / "spectral_analysis.json", "w") as f:
                        json.dump(analysis, f, indent=2)
            else:
                eta_motion = eta_inv

        # =================================================================
        # Phase B: Semantic Layer (Embedding Space - Mid Spectral Resolution)
        # Captures acceleration curves, rhythmic patterns, phase relationships
        # =================================================================
        delta_e = None
        opt_result = None
        e0 = None
        z0 = None

        if cfg.use_velocity:
            logger.info("  [Phase B] Semantic Layer - Position-Aware Velocity Matching...")

            # Encode video to latent space
            ref_norm = normalize_video(ref_video).unsqueeze(0)
            z0 = encode_video_to_latents(self.pipe, ref_norm, self.device)

            # Encode caption to embedding space
            e0 = self._encode_prompt(caption)

            # Prepare augmented embeddings for disentanglement
            aug_embeddings = None
            if cfg.use_disentangle:
                logger.info("  [Phase B] Generating cross-content augmentations...")
                aug_prompts = self.augmenter.augment(caption)
                aug_embeddings = [self._encode_prompt(p) for p in aug_prompts]
                logger.info(f"    Generated {len(aug_prompts)} augmented prompts")

                if cfg.save_intermediates:
                    (out / "augmented_prompts.json").write_text(
                        json.dumps(aug_prompts, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

            # Use inverted noise if available
            eta_for_opt = eta_inv if eta_inv is not None else torch.randn_like(z0)

            # Execute position-aware optimization
            matcher = PositionAwareVelocityMatcher(
                pipe=self.pipe,
                T_m=cfg.T_m,
                num_opt_steps=cfg.num_opt_steps,
                lr=cfg.opt_lr,
                lambda_dis=cfg.lambda_dis,
                lambda_pos=cfg.lambda_pos if cfg.use_position_aware else 0.0,
                dis_every_n_steps=cfg.dis_every_n_steps,
                position_aware=cfg.use_position_aware,
                warmup_ratio=cfg.warmup_ratio,
                device=self.device,
            )

            opt_result = matcher.optimize(
                z0=z0,
                e0=e0,
                eta_inv=eta_for_opt,
                aug_embeddings=aug_embeddings,
                use_disentangle=cfg.use_disentangle,
            )

            delta_e = opt_result["delta_e"]
            logger.info(
                f"    Optimization complete: ||delta_e||={delta_e.norm().item():.4f}, "
                f"L_vel={opt_result['final_loss_vel']:.6f}, "
                f"L_dis={opt_result['final_loss_dis']:.6f}"
            )

            if cfg.save_intermediates:
                torch.save(delta_e.cpu(), str(out / "delta_e_raw.pt"))
                with open(out / "optimization_log.json", "w") as f:
                    json.dump(opt_result["loss_history"], f, indent=2)
                # Save position energy distribution
                if "position_energy_distribution" in opt_result:
                    with open(out / "position_energy.json", "w") as f:
                        json.dump(opt_result["position_energy_distribution"], f, indent=2)

        # =================================================================
        # Phase C: Interpretable Layer (Token/Text Space - Coarsest Spectral)
        # Captures action type, direction, speed class (human-readable)
        # =================================================================
        motion_tokens = []
        motion_text = ""
        token_decode_result = None

        # Stage C.1: Velocity-Preserving Token Decoding (delta_e -> tokens)
        if cfg.use_token_decode and delta_e is not None and e0 is not None:
            logger.info("  [Phase C] Velocity-Preserving Token Decoding...")
            try:
                token_decode_result = self.token_decoder.decode(
                    delta_e=delta_e,
                    e0=e0,
                    z0=z0 if z0 is not None else None,
                    base_caption=caption,
                    max_motion_tokens=cfg.token_decode_max_tokens,
                )
                motion_tokens = token_decode_result["motion_tokens"]
                motion_text = token_decode_result["motion_text"]
                logger.info(f"    Decoded tokens: {motion_tokens[:5]}...")
                logger.info(
                    f"    Velocity preservation: "
                    f"{token_decode_result['velocity_preservation_score']:.4f}"
                )
            except Exception as e:
                logger.warning(f"    Token decoding failed: {e}, falling back to VLM")

        # Stage C.2: VLM Motion Text Decoding (comparative description)
        if cfg.use_text_decode and delta_e is not None and not motion_text:
            logger.info("  [Phase C] VLM Motion Text Decoding...")
            motion_text = self._decode_motion_text(
                ref_video, caption, delta_e, e0, eta_motion, generator, out
            )
            logger.info(f"    Motion text: {motion_text[:80]}...")

        # =================================================================
        # Package Three-Layer Motion Asset
        # =================================================================
        logger.info("  [Package] Creating three-layer motion asset...")

        asset = self._asset_manager.create_asset(
            delta_e=delta_e if delta_e is not None else torch.zeros(1),
            eta_motion=eta_motion if eta_motion is not None else torch.zeros(1),
            motion_text=motion_text,
            source_caption=caption,
            source_video=video_path,
            extraction_params={
                # Layer 3 params
                "rho_s": cfg.rho_s,
                "rho_m": cfg.rho_m,
                "alpha": cfg.alpha,
                "inversion_steps": cfg.inversion_steps,
                "use_midpoint": cfg.use_midpoint,
                # Layer 2 params
                "T_m": cfg.T_m,
                "num_opt_steps": cfg.num_opt_steps,
                "lr": cfg.opt_lr,
                "lambda_dis": cfg.lambda_dis,
                "lambda_pos": cfg.lambda_pos,
                "use_disentangle": cfg.use_disentangle,
                "use_position_aware": cfg.use_position_aware,
                # Layer 1 params
                "motion_tokens": motion_tokens,
                "token_decode_score": (
                    token_decode_result["velocity_preservation_score"]
                    if token_decode_result else 0
                ),
            },
        )

        asset_dir = str(out / "asset")
        self._asset_manager.save(asset, asset_dir)

        # -- Result Summary --
        elapsed = time.time() - t0
        result = {
            "asset_dir": asset_dir,
            "motion_text": motion_text,
            "motion_tokens": motion_tokens,
            "caption": caption,
            "flags": flags,
            "experiment": cfg.experiment_name(),
            "time_seconds": elapsed,
            "delta_e_norm": delta_e.norm().item() if delta_e is not None else 0,
            "optimization": {
                "final_loss_vel": opt_result["final_loss_vel"] if opt_result else 0,
                "final_loss_dis": opt_result["final_loss_dis"] if opt_result else 0,
            },
            "spectral_layers": {
                "layer3_structural": eta_motion is not None,
                "layer2_semantic": delta_e is not None,
                "layer1_interpretable": bool(motion_text or motion_tokens),
            },
        }

        with open(out / "extract_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"[VMAD Extract] Video Prompt Inversion complete in {elapsed:.1f}s")
        logger.info(f"  Asset: {asset_dir}")
        return result

    # =========================================================================
    # Apply Mode: Three-Layer Video Reproduction / Motion Transfer
    # =========================================================================

    def apply(
        self,
        content_prompt: str,
        asset_dir: str,
        output_dir: str,
        strength: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Apply three-layer motion asset to reproduce or transfer video.

        Combines all spectral layers for comprehensive motion guidance:
            Layer 1 (Text): Append motion description to content prompt
            Layer 2 (Embedding): e_final = e_content + strength * delta_e
            Layer 3 (Noise): eta_init = sqrt(alpha*s)*eta_motion + sqrt(1-alpha*s)*eta_random

        The three layers provide complementary guidance:
        - Text gives the model semantic understanding of the motion
        - Embedding provides precise mid-frequency motion control
        - Noise prior guides the earliest denoising steps with structural info

        Operating Modes:
        - Reproduction (content_prompt = original caption): faithfully reproduce the source video
        - Motion Transfer (content_prompt ≠ original caption): transfer motion to new content

        Args:
            content_prompt: Content description (original caption for reproduction,
                          new description for motion transfer)
            asset_dir: Path to motion asset directory
            output_dir: Output directory
            strength: Motion intensity [0, 1+]

        Returns:
            Result dictionary
        """
        t0 = time.time()
        cfg = self.config
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        seed = cfg.seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info("[VMAD Apply] Three-Layer Video Reproduction")
        logger.info(f"  Content: '{content_prompt[:60]}', strength={strength}")

        # -- Load three-layer asset --
        asset = self._asset_manager.load(asset_dir)

        # -- Layer 1: Combine text (semantic guidance) --
        full_prompt = content_prompt
        if asset.motion_text and cfg.use_token_decode:
            full_prompt = f"{content_prompt}, {asset.motion_text}"
            logger.info(f"  [Layer 1] Text fusion: {full_prompt[:80]}...")
        else:
            logger.info(f"  [Layer 1] Pure prompt (no motion_text): {full_prompt[:80]}...")

        # -- Layer 2: Prepare delta_e hook (embedding-level motion) --
        # IMPORTANT: We inject delta_e via a text encoder hook instead of passing
        # prompt_embeds directly. This preserves the pipeline's normal prompt
        # processing path (CFG, negative prompt encoding, attention masks, etc.)
        # which is critical for generation quality.
        # Reference: P-Flow (2025) uses prompt string path exclusively.
        delta_e = None
        if asset.delta_e is not None and cfg.use_velocity:
            delta_e = asset.delta_e.to(device=self.device, dtype=torch.bfloat16)
            if delta_e.dim() == 2:
                delta_e = delta_e.unsqueeze(0)
            # Scale strength to be very gentle (P-Flow insight: minimal guidance)
            effective_strength = strength * cfg.alpha  # e.g. 1.0 * 0.001 = 0.001
            logger.info(
                f"  [Layer 2] Embedding hook: strength={strength}, "
                f"effective={effective_strength:.6f}, ||delta_e||={delta_e.norm():.2f}"
            )
        else:
            effective_strength = 0.0
            logger.info("  [Layer 2] Skipped (no delta_e or velocity disabled)")

        # -- Layer 3: Noise prior blending (structural guidance) --
        # Following P-Flow: alpha=0.001 provides minimal but sufficient structural guidance
        if cfg.use_blend:
            latents = self._asset_manager.apply_noise_prior(
                asset, alpha=cfg.alpha, strength=strength, generator=generator
            )
            if latents is not None:
                logger.info(f"  [Layer 3] Noise prior: alpha={cfg.alpha}")
        else:
            latents = None
            logger.info("  [Layer 3] Skipped (use_blend=False)")

        # -- Generate video with three-layer guidance --
        # Key fix: Use prompt STRING path (not prompt_embeds) to preserve CFG behavior.
        # If delta_e exists, inject via hook on the text encoder output.
        logger.info("  Generating video with three-layer motion guidance...")
        if delta_e is not None and effective_strength > 0:
            gen_video = self._generate_with_embedding_hook(
                full_prompt, latents, generator, delta_e, effective_strength
            )
        else:
            gen_video = self._generate(full_prompt, latents, generator)

        # Save output
        video_path = str(out / "generated.mp4")
        save_video_tensor(gen_video, video_path, fps=cfg.fps)

        elapsed = time.time() - t0
        result = {
            "video_path": video_path,
            "content_prompt": content_prompt,
            "motion_text": asset.motion_text,
            "full_prompt": full_prompt,
            "strength": strength,
            "asset_dir": asset_dir,
            "time_seconds": elapsed,
            "layers_applied": {
                "text": bool(asset.motion_text),
                "embedding": asset.delta_e is not None,
                "noise_prior": asset.eta_motion is not None,
            },
        }

        with open(out / "apply_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"[VMAD Apply] Done in {elapsed:.1f}s -> {video_path}")
        return result

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _compute_inversion(self, ref_video: torch.Tensor, prompt: str) -> torch.Tensor:
        """Flow Matching Inversion: V_ref -> eta_inv."""
        ref_norm = normalize_video(ref_video).unsqueeze(0)
        ref_latents = encode_video_to_latents(self.pipe, ref_norm, self.device)

        # Use null prompt for unconditional inversion (avoid condition leakage)
        prompt_embeds = self._encode_prompt("")

        inverter = FlowMatchingInverter(
            pipe=self.pipe,
            num_inversion_steps=self.config.inversion_steps,
            guidance_scale=1.0,
            device=self.device,
        )

        if self.config.use_midpoint:
            eta_inv = inverter.invert_midpoint(ref_latents, prompt_embeds)
        else:
            eta_inv = inverter.invert(ref_latents, prompt_embeds)

        logger.info(f"    eta_inv: mean={eta_inv.mean():.4f}, std={eta_inv.std():.4f}")
        return eta_inv

    def _decode_motion_text(
        self,
        ref_video: torch.Tensor,
        caption: str,
        delta_e: torch.Tensor,
        e0: torch.Tensor,
        eta_motion: Optional[torch.Tensor],
        generator: torch.Generator,
        out_dir: Path,
    ) -> str:
        """
        VLM Motion Text Decoding via comparative video generation.

        Generates two videos (with/without delta_e) and uses VLM to describe
        the motion difference - producing a human-readable motion description.
        """
        cfg = self.config

        # Prepare noise
        if eta_motion is not None and cfg.use_blend:
            eta_random = torch.randn(
                eta_motion.shape, dtype=eta_motion.dtype,
                device=self.device, generator=generator,
            )
            alpha = cfg.alpha
            eta_init = (alpha ** 0.5) * eta_motion + ((1 - alpha) ** 0.5) * eta_random
        else:
            eta_init = None

        # Generate V_without (no delta_e - baseline)
        logger.info("    Generating V_without (baseline)...")
        v_without = self._generate(caption, eta_init, generator)
        path_without = str(out_dir / "v_without_delta_e.mp4")
        save_video_tensor(v_without, path_without, fps=cfg.fps)

        # Generate V_with (with delta_e - motion-enhanced)
        logger.info("    Generating V_with (motion-enhanced)...")
        enhanced_embeds = e0 + delta_e  # Inject delta_e into embedding
        v_with = self._generate(caption, eta_init, generator, prompt_embeds=enhanced_embeds)
        path_with = str(out_dir / "v_with_delta_e.mp4")
        save_video_tensor(v_with, path_with, fps=cfg.fps)

        # VLM comparative decoding
        motion_text = self.vlm_client.decode_motion_text(path_with, path_without)
        return motion_text

    def _generate(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call T2V model to generate video."""
        cfg = self.config
        kwargs = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "guidance_scale": cfg.guidance_scale,
            "num_inference_steps": cfg.num_inference_steps,
            "generator": generator,
            "output_type": "pt",
        }

        if latents is not None:
            kwargs["latents"] = latents

        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs.pop("prompt", None)

        output = self.pipe(**kwargs)

        # Handle output format
        if hasattr(output, "frames"):
            video = output.frames
            if isinstance(video, list):
                import torchvision.transforms as T
                frames = [T.ToTensor()(f) for f in video[0]]
                video = torch.stack(frames, dim=1)
            elif isinstance(video, torch.Tensor):
                if video.dim() == 5:
                    video = video[0]
                    if video.shape[0] == cfg.num_frames:
                        video = video.permute(1, 0, 2, 3)
        else:
            video = output[0]

        if video.min() < 0:
            video = denormalize_video(video)
        return video.clamp(0, 1)

    def _generate_with_embedding_hook(
        self,
        prompt: str,
        latents: Optional[torch.Tensor],
        generator: torch.Generator,
        delta_e: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """
        Generate video with delta_e injected via text encoder hook.

        This preserves the pipeline's normal prompt processing path (CFG,
        negative prompt encoding, attention masks) while adding a gentle
        delta_e perturbation to the encoded prompt embedding.

        The hook intercepts the text encoder's output and adds:
            e_final = e_original + strength * delta_e

        This is analogous to P-Flow's noise prior: minimal perturbation
        that provides guidance without disrupting the generation process.
        """
        cfg = self.config

        # Install hook on text encoder to inject delta_e
        text_encoder = self.pipe.text_encoder
        hook_handle = None
        hook_applied = [False]  # Use list for mutability in closure

        def text_encoder_hook(module, input, output):
            """Add delta_e to text encoder output (positive prompt only)."""
            if hook_applied[0]:
                return output  # Only apply once (for positive prompt, not negative)

            # output is typically (hidden_states,) or BaseModelOutput
            if isinstance(output, tuple):
                hidden_states = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output

            # Inject delta_e with gentle strength
            de = delta_e.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if de.shape[1] != hidden_states.shape[1]:
                # Truncate or pad delta_e to match sequence length
                min_len = min(de.shape[1], hidden_states.shape[1])
                de_aligned = torch.zeros_like(hidden_states)
                de_aligned[:, :min_len, :] = de[:, :min_len, :]
                de = de_aligned

            hidden_states = hidden_states + strength * de
            hook_applied[0] = True

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            elif hasattr(output, "last_hidden_state"):
                output.last_hidden_state = hidden_states
                return output
            else:
                return hidden_states

        # Register hook
        hook_handle = text_encoder.register_forward_hook(text_encoder_hook)

        try:
            # Generate normally with prompt string — pipeline handles everything
            video = self._generate(prompt, latents, generator)
        finally:
            # Always remove hook
            if hook_handle is not None:
                hook_handle.remove()

        logger.info(
            f"    Hook applied: {hook_applied[0]}, "
            f"delta_e injection strength={strength:.6f}"
        )
        return video

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text to embedding via T5 text encoder.

        For Wan2.1: output shape is (1, 512, 4096) — UMT5-XXL hidden_size=4096,
        max_sequence_length=512.
        """
        if hasattr(self.pipe, "encode_prompt"):
            sig = inspect.signature(self.pipe.encode_prompt)
            params = sig.parameters
            kwargs = {"prompt": prompt}
            if "device" in params:
                kwargs["device"] = self.device
            if "num_videos_per_prompt" in params:
                kwargs["num_videos_per_prompt"] = 1
            if "do_classifier_free_guidance" in params:
                kwargs["do_classifier_free_guidance"] = False
            if "max_sequence_length" in params:
                kwargs["max_sequence_length"] = 512
            result = self.pipe.encode_prompt(**kwargs)
            return result[0] if isinstance(result, tuple) else result
        else:
            inputs = self.pipe.tokenizer(
                prompt, padding="max_length",
                max_length=512,
                truncation=True, return_tensors="pt",
            )
            return self.pipe.text_encoder(inputs.input_ids.to(self.device))[0]
