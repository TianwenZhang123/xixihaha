"""
Diagnostic test: Does using prompt_embeds instead of prompt string
produce the same video?

Hypothesis: The quality degradation in VMAD Apply is caused by
the prompt_embeds kwarg path in WanPipeline behaving differently
from the prompt string path (e.g., missing negative prompt encoding,
different attention mask, different sequence length handling).

Test:
  A) Generate with prompt="..." (normal baseline path)
  B) Generate with prompt_embeds=encode_prompt("...") (VMAD Apply path)
  C) Compare CLIP scores of A vs B

If B is significantly worse than A, the bug is in how prompt_embeds
is handled by the pipeline, NOT in delta_e or eta_inv.
"""

import sys
import os
import torch
import json
import logging
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distributed import load_model_single_gpu
from src.video_utils import save_video_tensor, denormalize_video
from src.pipeline import VMADConfig, NEGATIVE_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda"
SEED = 42
MODEL_PATH = "/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers"

# Use sample 7's caption
PROMPT = "A man in a red jacket is walking on a dirt road surrounded by autumn trees. The leaves are orange and red, and the sky is overcast. He is wearing a backpack and walking away from the camera."

OUTPUT_DIR = "/root/autodl-tmp/outputs/prompt_embeds_diagnostic"


def encode_prompt(pipe, prompt: str, device: str) -> torch.Tensor:
    """Replicate VMAD's _encode_prompt method."""
    if hasattr(pipe, "encode_prompt"):
        sig = inspect.signature(pipe.encode_prompt)
        params = sig.parameters
        kwargs = {"prompt": prompt}
        if "device" in params:
            kwargs["device"] = device
        if "num_videos_per_prompt" in params:
            kwargs["num_videos_per_prompt"] = 1
        if "do_classifier_free_guidance" in params:
            kwargs["do_classifier_free_guidance"] = False
        if "max_sequence_length" in params:
            kwargs["max_sequence_length"] = 512
        result = pipe.encode_prompt(**kwargs)
        return result[0] if isinstance(result, tuple) else result
    else:
        inputs = pipe.tokenizer(
            prompt, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        )
        return pipe.text_encoder(inputs.input_ids.to(device))[0]


def extract_video(output, cfg):
    """Extract video tensor from pipeline output."""
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = VMADConfig()

    logger.info("Loading model...")
    pipe = load_model_single_gpu(MODEL_PATH, torch_dtype=torch.bfloat16)
    pipe = pipe.to(DEVICE)

    # === First, inspect what WanPipeline.__call__ accepts for prompt_embeds ===
    logger.info("=" * 60)
    logger.info("INSPECTING WanPipeline.__call__ signature")
    logger.info("=" * 60)
    call_sig = inspect.signature(pipe.__call__)
    for name, param in call_sig.parameters.items():
        if "prompt" in name.lower() or "embed" in name.lower() or "guidance" in name.lower() or "negative" in name.lower():
            logger.info(f"  {name}: default={param.default}")

    # Also inspect encode_prompt to understand CFG handling
    logger.info("  --- encode_prompt signature ---")
    if hasattr(pipe, "encode_prompt"):
        ep_sig = inspect.signature(pipe.encode_prompt)
        for name, param in ep_sig.parameters.items():
            logger.info(f"  encode_prompt.{name}: default={param.default}")

    # Check: what does encode_prompt return with do_classifier_free_guidance=True vs False?
    logger.info("  --- Testing encode_prompt CFG behavior ---")
    ep_kwargs_cfg_true = {"prompt": PROMPT}
    ep_sig = inspect.signature(pipe.encode_prompt)
    ep_params = ep_sig.parameters
    if "device" in ep_params:
        ep_kwargs_cfg_true["device"] = DEVICE
    if "num_videos_per_prompt" in ep_params:
        ep_kwargs_cfg_true["num_videos_per_prompt"] = 1
    if "do_classifier_free_guidance" in ep_params:
        ep_kwargs_cfg_true["do_classifier_free_guidance"] = True
    if "max_sequence_length" in ep_params:
        ep_kwargs_cfg_true["max_sequence_length"] = 512
    result_cfg_true = pipe.encode_prompt(**ep_kwargs_cfg_true)
    if isinstance(result_cfg_true, tuple):
        logger.info(f"  CFG=True returns tuple of {len(result_cfg_true)} elements:")
        for i, r in enumerate(result_cfg_true):
            if r is not None:
                logger.info(f"    [{i}] shape={r.shape}, dtype={r.dtype}")
            else:
                logger.info(f"    [{i}] None")
    else:
        logger.info(f"  CFG=True returns: shape={result_cfg_true.shape}")

    ep_kwargs_cfg_false = dict(ep_kwargs_cfg_true)
    ep_kwargs_cfg_false["do_classifier_free_guidance"] = False
    result_cfg_false = pipe.encode_prompt(**ep_kwargs_cfg_false)
    if isinstance(result_cfg_false, tuple):
        logger.info(f"  CFG=False returns tuple of {len(result_cfg_false)} elements:")
        for i, r in enumerate(result_cfg_false):
            if r is not None:
                logger.info(f"    [{i}] shape={r.shape}, dtype={r.dtype}")
            else:
                logger.info(f"    [{i}] None")
    else:
        logger.info(f"  CFG=False returns: shape={result_cfg_false.shape}")

    # === Test A: prompt string (baseline) ===
    logger.info("=" * 60)
    logger.info("TEST A: Generate with prompt STRING")
    logger.info("=" * 60)
    generator_a = torch.Generator(device=DEVICE).manual_seed(SEED)
    output_a = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.num_frames,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator_a,
        output_type="pt",
    )
    video_a = extract_video(output_a, cfg)
    path_a = os.path.join(OUTPUT_DIR, "A_prompt_string.mp4")
    save_video_tensor(video_a, path_a, fps=cfg.fps)
    logger.info(f"  Saved: {path_a}")
    logger.info(f"  Video shape: {video_a.shape}, range: [{video_a.min():.3f}, {video_a.max():.3f}]")

    # === Encode prompt to embedding ===
    logger.info("=" * 60)
    logger.info("ENCODING PROMPT")
    logger.info("=" * 60)
    prompt_embeds = encode_prompt(pipe, PROMPT, DEVICE)
    logger.info(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
    logger.info(f"  prompt_embeds norm: {prompt_embeds.norm():.4f}")
    logger.info(f"  prompt_embeds mean: {prompt_embeds.mean():.6f}, std: {prompt_embeds.std():.6f}")

    # === Test B: prompt_embeds (VMAD Apply path) ===
    logger.info("=" * 60)
    logger.info("TEST B: Generate with prompt_embeds TENSOR (same content)")
    logger.info("=" * 60)
    generator_b = torch.Generator(device=DEVICE).manual_seed(SEED)
    output_b = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt=NEGATIVE_PROMPT,
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.num_frames,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator_b,
        output_type="pt",
    )
    video_b = extract_video(output_b, cfg)
    path_b = os.path.join(OUTPUT_DIR, "B_prompt_embeds.mp4")
    save_video_tensor(video_b, path_b, fps=cfg.fps)
    logger.info(f"  Saved: {path_b}")
    logger.info(f"  Video shape: {video_b.shape}, range: [{video_b.min():.3f}, {video_b.max():.3f}]")

    # === Test C: prompt_embeds WITHOUT negative_prompt ===
    # (in case the pipeline ignores negative_prompt when prompt_embeds is given)
    logger.info("=" * 60)
    logger.info("TEST C: prompt_embeds WITHOUT negative_prompt")
    logger.info("=" * 60)
    generator_c = torch.Generator(device=DEVICE).manual_seed(SEED)
    try:
        output_c = pipe(
            prompt_embeds=prompt_embeds,
            # NO negative_prompt — to see if pipeline handles it differently
            height=cfg.height,
            width=cfg.width,
            num_frames=cfg.num_frames,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator_c,
            output_type="pt",
        )
        video_c = extract_video(output_c, cfg)
        path_c = os.path.join(OUTPUT_DIR, "C_embeds_no_neg.mp4")
        save_video_tensor(video_c, path_c, fps=cfg.fps)
        logger.info(f"  Saved: {path_c}")
    except Exception as e:
        logger.error(f"  Test C failed: {e}")
        video_c = None
        path_c = "FAILED"

    # === Compare ===
    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    diff_ab = (video_a - video_b).abs()
    logger.info(f"  A vs B pixel diff: mean={diff_ab.mean():.6f}, max={diff_ab.max():.6f}")

    if diff_ab.mean() < 1e-4:
        logger.info("  ✓ A≈B: prompt_embeds path produces SAME result as prompt string")
        logger.info("  → Bug is NOT in prompt_embeds path. Look at delta_e/eta_inv quality.")
    elif diff_ab.mean() < 0.01:
        logger.info("  ~ A≈B: Small numerical difference (likely floating point)")
        logger.info("  → prompt_embeds path is approximately equivalent")
    else:
        logger.info("  ✗ A≠B: SIGNIFICANT difference!")
        logger.info(f"  → Mean diff={diff_ab.mean():.4f} — prompt_embeds path IS broken!")
        logger.info("  → This is the root cause of VMAD Apply degradation")

    if video_c is not None:
        diff_bc = (video_b - video_c).abs()
        logger.info(f"  B vs C pixel diff: mean={diff_bc.mean():.6f}")
        if diff_bc.mean() > 0.01:
            logger.info("  → negative_prompt is IGNORED when prompt_embeds is used!")

    # Save results
    results = {
        "prompt": PROMPT,
        "seed": SEED,
        "paths": {"A_string": path_a, "B_embeds": path_b, "C_no_neg": path_c},
        "diff_AB_mean": diff_ab.mean().item(),
        "diff_AB_max": diff_ab.max().item(),
        "videos_identical": diff_ab.mean().item() < 1e-4,
        "conclusion": (
            "prompt_embeds path EQUIVALENT" if diff_ab.mean().item() < 0.01
            else "prompt_embeds path BROKEN — root cause confirmed"
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "diagnostic_result.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("Run CLIP eval: python P-Flow/evaluation/run_clip_xclip_eval.py --gen-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
