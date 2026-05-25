"""
Baseline Pipeline: Direct Caption → T2V Generation.

This is the simplest approach (same as Video2Prompt):
    1. Use VLM (Qwen2.5-VL-7B) to caption the reference video
    2. Feed caption to Wan2.1-1.3B to generate video
    3. Done (no iteration, no noise prior)

This serves as the "Direct Caption" baseline in the paper.
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch

from .distributed import load_model_single_gpu, setup_single_gpu, cleanup_gpu_memory
from .video_utils import load_video, save_video_tensor

logger = logging.getLogger(__name__)

# Same negative prompt as Video2Prompt collaborator (wan_model_lib.py)
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


class BaselinePipeline:
    """
    Baseline: VLM Caption → Wan T2V Generation.

    No iteration, no noise prior. One-shot generation from caption.
    This is equivalent to the Video2Prompt approach.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vlm_model=None,
        vlm_processor=None,
    ):
        """
        Args:
            config: Configuration dict.
            vlm_model: Pre-loaded Qwen2.5-VL model (optional, for batch reuse).
            vlm_processor: Pre-loaded processor (optional, for batch reuse).
        """
        self.config = config
        self.device = setup_single_gpu()
        self.dtype = getattr(torch, config.get("model", {}).get("dtype", "bfloat16"))

        # Lazy-loaded components
        self._pipe = None
        self._vlm_model = vlm_model
        self._vlm_processor = vlm_processor

    @property
    def pipe(self):
        """Lazy-load the Wan T2V pipeline."""
        if self._pipe is None:
            model_path = self.config["model"]["t2v_path"]
            self._pipe = load_model_single_gpu(
                model_path=model_path,
                dtype=self.dtype,
                model_type="t2v",
            )
        return self._pipe

    def load_vlm(self):
        """Load Qwen2.5-VL-7B for captioning."""
        if self._vlm_model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        vlm_path = self.config["vlm"]["model_path"]
        logger.info(f"Loading VLM from {vlm_path}...")

        self._vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        self._vlm_processor = AutoProcessor.from_pretrained(
            vlm_path, local_files_only=True
        )
        logger.info("VLM loaded.")

    def unload_vlm(self):
        """Release VLM from GPU memory."""
        if self._vlm_model is not None:
            del self._vlm_model
            del self._vlm_processor
            self._vlm_model = None
            self._vlm_processor = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("VLM unloaded, GPU memory released.")

    def caption_video(self, video_path: str) -> str:
        """
        Use Qwen2.5-VL-7B to generate a caption for the video.

        Same approach as Video2Prompt: feed video directly to VLM,
        get a detailed English description.

        Args:
            video_path: Path to the reference video.

        Returns:
            Caption string (English).
        """
        from qwen_vl_utils import process_vision_info

        self.load_vlm()

        prompt_text = (
            "Describe this video in detail in English. Include: "
            "the main subject and their appearance, the action/motion happening, "
            "the scene/background, camera angle and movement, "
            "and lighting/atmosphere. Write as a single paragraph suitable "
            "for a text-to-video generation model."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self._vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0]

        inputs = self._vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self._vlm_model.generate(
                **inputs, max_new_tokens=512
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    @torch.no_grad()
    def generate_video(
        self,
        prompt: str,
        output_path: str,
        seed: int = 42,
        sample_id: int = 0,
    ) -> str:
        """
        Generate video from text prompt using Wan2.1-1.3B.

        Args:
            prompt: Text prompt (caption).
            output_path: Where to save the generated video.
            seed: Base random seed.
            sample_id: Sample ID (seed = base_seed + sample_id).

        Returns:
            Path to generated video.
        """
        from diffusers.utils import export_to_video

        video_cfg = self.config["video"]
        generator = torch.Generator(device=self.device).manual_seed(seed + sample_id)

        frames = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=video_cfg["height"],
            width=video_cfg["width"],
            num_frames=video_cfg["num_frames"],
            num_inference_steps=video_cfg["num_inference_steps"],
            guidance_scale=video_cfg["guidance_scale"],
            generator=generator,
        ).frames[0]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        export_to_video(frames, output_path, fps=video_cfg["fps"])

        return output_path

    def run_single(
        self,
        video_path: str,
        output_dir: str,
        sample_id: int = 0,
        seed: int = 42,
        caption: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run baseline pipeline for a single video.

        Steps:
            1. Caption the reference video (or use provided caption)
            2. Generate video from caption
            3. Save results

        Args:
            video_path: Path to reference video.
            output_dir: Output directory.
            sample_id: Sample identifier.
            seed: Random seed.
            caption: Pre-computed caption (skip VLM if provided).

        Returns:
            Dict with results.
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Caption
        if caption is None:
            logger.info(f"[Baseline] Captioning video: {video_path}")
            caption = self.caption_video(video_path)
            # Save caption
            caption_file = output_path / f"{sample_id}.txt"
            caption_file.write_text(caption + "\n", encoding="utf-8")
            logger.info(f"  Caption: {caption[:100]}...")

        # Step 2: Generate
        video_output = str(output_path / f"{sample_id}.mp4")
        logger.info(f"[Baseline] Generating video for sample {sample_id}...")
        self.generate_video(
            prompt=caption,
            output_path=video_output,
            seed=seed,
            sample_id=sample_id,
        )

        elapsed = time.time() - start_time
        logger.info(f"[Baseline] Done. Time: {elapsed:.1f}s. Output: {video_output}")

        return {
            "sample_id": sample_id,
            "video_path": video_path,
            "caption": caption,
            "generated_video": video_output,
            "time_seconds": elapsed,
            "method": "baseline",
        }
