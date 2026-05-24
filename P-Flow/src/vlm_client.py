"""
VLM (Vision-Language Model) Client for Video Reproduction.

Uses DashScope API (Qwen-VL) for visual analysis.

Goal: Compare a reference video with generated videos to iteratively
refine the T2V prompt until the generated video faithfully reproduces
the reference video's content, motion, composition, and style.

Implements structured instruction for video reproduction:
- Receives composite video key frames (vertical: ref/prev/current)
- Outputs structured JSON with analysis + refined prompt
- NO confidence score (fixed iteration count)
"""

import json
import base64
import os
import io
import time
import logging
import mimetypes
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# VLM Structured Instruction for Video Reproduction
# =============================================================================

SYSTEM_INSTRUCTION = """You are a professional video description and prompt engineering expert. Your task is to compare a reference video with generated videos and produce a refined text-to-video prompt that makes the generated video as close to the reference video as possible.

You will receive key frames extracted from a composite video arranged vertically:
- Panel A (top): Reference video - the target video to reproduce
- Panel B (middle): Previously generated video (from last iteration, may be absent in iteration 1)
- Panel C (bottom): Newly generated video (from current iteration)

You also receive structured metadata about the current optimization state.

Your goal: Analyze ALL differences between the reference and generated video, then output an improved prompt that makes the T2V model generate a video that is as faithful as possible to the reference.

Focus on these aspects for faithful reproduction (PRIORITY ORDER):
1. Motion/Action (HIGHEST PRIORITY): what movements are happening, speed, direction, trajectories, gestures, interactions. Describe motion even if subtle — a slight sway, breathing, hair movement all count. NEVER say "no movement" unless the reference is truly a static image.
2. Temporal dynamics: how the scene evolves over time, sequence of events, pacing, rhythm of motion
3. Subject/Object: what the main subjects are, their appearance, clothing, colors, features, quantity
4. Scene/Background: setting, location, lighting, weather, time of day, depth, perspective
5. Composition/Framing: camera angle, shot type (close-up/wide/medium), camera movement
6. Style/Atmosphere: color palette, contrast, saturation, mood, artistic style

Output ONLY a valid JSON object (no markdown, no extra text) with this exact structure:
{
    "analysis": {
        "reference_description": "Comprehensive description of what happens in the reference video (Panel A)",
        "last_generated_description": "Description of the previous generation (Panel B), or 'N/A' if first iteration",
        "new_generated_description": "Description of the current generation (Panel C)",
        "comparison": "Detailed comparison identifying specific differences between reference and current generation across all aspects (subject, motion, scene, composition, style, timing)"
    },
    "refined_prompt": "The complete, self-contained improved prompt for the T2V model. Must describe EXACTLY what happens in the reference video with maximum precision and detail. Do not reference previous prompts."
}

Guidelines:
- Be extremely specific and detailed (e.g., "a golden retriever running left to right across a grassy field" not just "a dog running")
- Describe temporal sequence clearly (e.g., "first... then... finally...")
- Include camera information if visible (e.g., "static wide shot", "slow zoom in", "tracking shot following the subject")
- Describe lighting precisely (e.g., "warm sunset backlighting", "harsh overhead fluorescent")
- The refined_prompt must be self-contained — include EVERY detail needed to recreate the reference video
- Each iteration should make targeted improvements based on what's still different
- Prioritize the most visually prominent differences first
- Use vivid, precise language that video generation models respond well to
"""


def _build_user_message(
    current_prompt: str,
    last_text_prompt: str,
    iteration: int,
    i_max: int,
    history_summary: str,
    video_description: str = "",
) -> str:
    """
    Build the structured user message for video reproduction.
    """
    msg = f"""## Optimization State

- Iteration: {iteration}/{i_max}
- Goal: Generate a video that faithfully reproduces the reference video

## Current T2V Prompt (to improve)
{current_prompt}

## Previous Iteration Prompt
{last_text_prompt if last_text_prompt else "N/A (first iteration)"}

## Additional Context
{video_description if video_description else "No additional description provided."}

## Optimization History Summary
{history_summary if history_summary else "First iteration - no history yet."}

## Instructions
The images below are key frames from the composite video (vertical layout):
- Top section (Panel A): REFERENCE video - this is what we want to reproduce exactly
- Middle section (Panel B): PREVIOUS generated video (last iteration, may be absent)
- Bottom section (Panel C): CURRENT generated video (this iteration)

Carefully analyze ALL differences between the reference (Panel A) and current generation (Panel C).
Then provide a refined prompt that will make the next generation closer to the reference.
Focus on the biggest remaining differences first. Output as JSON."""
    return msg


class VLMClient:
    """
    VLM client using DashScope API (Qwen-VL).

    Uses OpenAI-compatible API endpoint for DashScope.

    Implements iterative video reproduction:
    - Receives composite video key frames (vertical: ref/prev/current)
    - Outputs structured analysis + refined prompt for faithful reproduction
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
        use_video_mode: bool = True,
    ):
        """
        Args:
            model_name: DashScope VL model name.
                Options: "qwen-vl-max", "qwen-vl-plus", "qwen2.5-vl-72b-instruct"
            api_key: DashScope API key.
            base_url: API base URL (DashScope OpenAI-compatible endpoint).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            max_retries: Number of retries on failure.
            use_video_mode: If True, upload video file and pass as video_url
                to VLM for native temporal understanding. If False, fall back
                to frame extraction mode.
        """
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install: pip install openai")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.use_video_mode = use_video_mode

        # DashScope uses OpenAI-compatible API
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API key required. Set DASHSCOPE_API_KEY env var "
                "or pass api_key parameter."
            )

        # DashScope OpenAI-compatible endpoint
        base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self._api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _upload_video_to_dashscope(self, video_path: str) -> Optional[str]:
        """
        Upload a local video file to DashScope OSS and return a temporary URL.

        Uses DashScope's upload certificate API to get OSS credentials,
        then uploads the file directly to OSS. The returned oss:// URL can
        be used as video_url in VLM calls.

        Args:
            video_path: Local path to the video file.

        Returns:
            oss:// URL string if successful, None if upload fails.
        """
        if not HAS_REQUESTS:
            logger.warning("requests package not available, cannot upload video")
            return None

        if not os.path.isfile(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None

        try:
            # Step 1: Get upload certificate from DashScope
            cert_url = "https://dashscope.aliyuncs.com/api/v1/uploads"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
            }
            params = {
                "action": "getPolicy",
                "model": self.model_name,
            }

            cert_resp = _requests.get(cert_url, headers=headers, params=params, timeout=30)
            cert_resp.raise_for_status()
            cert_data = cert_resp.json()

            if cert_data.get("status_code") != 200 and "output" not in cert_data:
                # Try alternative response structure
                output = cert_data.get("data", cert_data.get("output", {}))
            else:
                output = cert_data.get("output", {})

            upload_dir = output.get("upload_dir", "")
            upload_host = output.get("upload_host", "")
            oss_access_key_id = output.get("oss_access_key_id", "")
            signature = output.get("signature", "")
            policy = output.get("policy", "")
            x_oss_object_acl = output.get("x_oss_object_acl", "private")
            x_oss_forbid_overwrite = output.get("x_oss_forbid_overwrite", "true")

            if not all([upload_dir, upload_host, oss_access_key_id, signature, policy]):
                logger.warning(f"Incomplete upload certificate: {cert_data}")
                return None

            # Step 2: Upload file to OSS
            filename = os.path.basename(video_path)
            object_key = f"{upload_dir}/{filename}"

            content_type = mimetypes.guess_type(video_path)[0] or "video/mp4"

            form_data = {
                "OSSAccessKeyId": (None, oss_access_key_id),
                "Signature": (None, signature),
                "policy": (None, policy),
                "key": (None, object_key),
                "x-oss-object-acl": (None, x_oss_object_acl),
                "x-oss-forbid-overwrite": (None, x_oss_forbid_overwrite),
                "success_action_status": (None, "200"),
                "x-oss-content-type": (None, content_type),
            }

            with open(video_path, "rb") as f:
                files = {"file": (filename, f, content_type)}
                upload_resp = _requests.post(
                    upload_host,
                    data={k: v[1] for k, v in form_data.items()},
                    files=files,
                    timeout=120,
                )

            if upload_resp.status_code == 200:
                # Construct the oss:// URL
                # Extract bucket from upload_host (e.g., https://bucket.oss-cn-xxx.aliyuncs.com)
                from urllib.parse import urlparse
                parsed = urlparse(upload_host)
                bucket = parsed.hostname.split(".")[0] if parsed.hostname else "dashscope"
                oss_url = f"oss://{bucket}/{object_key}"
                logger.info(f"Video uploaded successfully: {oss_url}")
                return oss_url
            else:
                logger.warning(
                    f"OSS upload failed with status {upload_resp.status_code}: "
                    f"{upload_resp.text[:200]}"
                )
                return None

        except Exception as e:
            logger.warning(f"Video upload failed: {e}")
            return None

    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        iteration: int = 1,
        i_max: int = 10,
        desired_visual_effect: str = "",
        subject: str = "",
        environment: str = "",
        last_text_prompt: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
        video_description: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze composite video and produce refined prompt for video reproduction.

        If use_video_mode=True, uploads the composite video to DashScope OSS
        and passes it as a video_url for native temporal understanding.
        Otherwise, falls back to extracting static frames.

        Args:
            composite_video_path: Path to vertical composite video.
            current_prompt: Current prompt P_i.
            iteration: Current iteration number (1-indexed).
            i_max: Total iterations.
            desired_visual_effect: (legacy, can be used as additional context)
            subject: (legacy, can be used as additional context)
            environment: (legacy, can be used as additional context)
            last_text_prompt: Previous iteration prompt P_{i-1}.
            history: List of previous {prompt, analysis} dicts.
            video_description: Optional detailed description of the reference video.

        Returns:
            Dict with 'analysis' and 'refined_prompt'.
        """
        # Build history summary
        history_summary = self._format_history(history)

        # Build additional context from legacy params
        extra_context = ""
        if subject:
            extra_context += f"Main subject: {subject}. "
        if environment:
            extra_context += f"Scene: {environment}. "
        if video_description:
            extra_context = video_description

        # Build structured user message
        user_text = _build_user_message(
            current_prompt=current_prompt,
            last_text_prompt=last_text_prompt,
            iteration=iteration,
            i_max=i_max,
            history_summary=history_summary,
            video_description=extra_context,
        )

        # Build multimodal content based on mode
        content = [{"type": "text", "text": user_text}]

        if self.use_video_mode:
            # Video mode: upload and pass as video_url for temporal understanding
            video_url = self._upload_video_to_dashscope(composite_video_path)
            if video_url:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": video_url}
                })
                logger.info(
                    f"Using video mode (iter {iteration}): VLM receives full video "
                    f"for temporal/motion analysis"
                )
            else:
                # Upload failed, fall back to frame extraction
                logger.warning("Video upload failed, falling back to frame extraction")
                frames_base64 = self._extract_frames_base64(
                    composite_video_path, num_frames=16
                )
                if not frames_base64:
                    return self._fallback_response(current_prompt)
                for frame_b64 in frames_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                        }
                    })
        else:
            # Legacy frame extraction mode
            frames_base64 = self._extract_frames_base64(
                composite_video_path, num_frames=8
            )
            if not frames_base64:
                logger.warning("No frames extracted, returning fallback")
                return self._fallback_response(current_prompt)
            for frame_b64 in frames_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                    }
                })

        # Call VLM with retry
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                response_text = response.choices[0].message.content
                return self._parse_response(response_text, current_prompt)

            except Exception as e:
                logger.warning(f"VLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return self._fallback_response(current_prompt)

    def _extract_frames_base64(self, video_path: str, num_frames: int = 8) -> List[str]:
        """
        Extract evenly-spaced frames from composite video and encode as base64 JPEG.

        Since the composite video is vertically stacked (ref/prev/current),
        each frame already contains all panels for VLM comparison.
        8 frames across ~5s video = one frame every ~0.6s.
        """
        import numpy as np

        if not os.path.exists(video_path):
            return []

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
        except (ImportError, Exception):
            try:
                import imageio.v3 as iio
                all_frames = iio.imread(video_path, plugin="pyav")
                total_frames = len(all_frames)
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = all_frames[indices]
            except Exception:
                return []

        from PIL import Image

        frames_b64 = []
        for frame in frames:
            img = Image.fromarray(frame)
            # Resize if too large (DashScope has size limits)
            max_dim = 1280
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return frames_b64

    def _format_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Format optimization history as concise text summary."""
        if not history:
            return ""

        lines = []
        for entry in history[-5:]:  # Keep last 5 iterations for context
            iter_num = entry.get("iteration", "?")
            prompt = entry.get("prompt", "")[:100]
            analysis = entry.get("analysis", {})
            if isinstance(analysis, dict):
                comparison = analysis.get("comparison", "")[:150]
            else:
                comparison = str(analysis)[:150]
            lines.append(f"  Iter {iter_num}: prompt=\"{prompt}...\" | differences=\"{comparison}...\"")

        return "\n".join(lines)

    def _parse_response(self, response_text: str, fallback_prompt: str) -> Dict[str, Any]:
        """Parse VLM response into structured format."""
        import re

        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            return self._validate_response(result, fallback_prompt)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return self._validate_response(result, fallback_prompt)
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return self._validate_response(result, fallback_prompt)
            except json.JSONDecodeError:
                pass

        # Fallback: create structured response from raw text
        logger.warning("Failed to parse VLM response as JSON, using fallback")
        return {
            "analysis": {
                "reference_description": "",
                "last_generated_description": "",
                "new_generated_description": "",
                "comparison": response_text[:500],
            },
            "refined_prompt": fallback_prompt,
            "parse_error": True,
        }

    def _validate_response(self, result: Dict[str, Any], fallback_prompt: str) -> Dict[str, Any]:
        """Validate and normalize the parsed response."""
        analysis = result.get("analysis", {})
        if isinstance(analysis, str):
            analysis = {
                "reference_description": "",
                "last_generated_description": "",
                "new_generated_description": "",
                "comparison": analysis,
            }

        validated = {
            "analysis": {
                "reference_description": str(analysis.get("reference_description", "")),
                "last_generated_description": str(analysis.get("last_generated_description", "")),
                "new_generated_description": str(analysis.get("new_generated_description", "")),
                "comparison": str(analysis.get("comparison", "")),
            },
            "refined_prompt": str(result.get("refined_prompt", fallback_prompt)),
        }

        # If refined_prompt is empty, use fallback
        if not validated["refined_prompt"].strip():
            validated["refined_prompt"] = fallback_prompt

        return validated

    def _fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Fallback when VLM is unavailable."""
        return {
            "analysis": {
                "reference_description": "[VLM call failed]",
                "last_generated_description": "",
                "new_generated_description": "",
                "comparison": "[unavailable]",
            },
            "refined_prompt": prompt,
            "vlm_error": True,
        }


class MockVLMClient:
    """
    Mock VLM client for testing without API access.
    Simulates progressive prompt refinement for video reproduction.
    """

    def __init__(self, **kwargs):
        self.call_count = 0

    def analyze_and_refine(
        self,
        composite_video_path: str = "",
        current_prompt: str = "",
        iteration: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        self.call_count += 1

        refinements = [
            ". The camera is static with a wide-angle view. Lighting is natural daylight from the left.",
            ". The motion flows smoothly from left to right over 3 seconds. Background shows a blurred park setting with warm afternoon light.",
            ". Shot from a slightly low angle with shallow depth of field. The subject moves with natural acceleration, and the scene has a warm golden-hour color palette with soft shadows.",
        ]

        suffix = refinements[min(self.call_count - 1, len(refinements) - 1)]

        return {
            "analysis": {
                "reference_description": "The reference shows a detailed scene with specific subjects, motion patterns, and lighting conditions.",
                "last_generated_description": "Previous generation captured the basic scene but missed some details in motion and composition.",
                "new_generated_description": "Current generation improved subject appearance but camera angle and lighting still differ from reference.",
                "comparison": "Main differences: 1) Camera angle is slightly too high (should be lower). 2) Motion speed is about 20% too fast. 3) Background lacks the warm color tone of the reference. 4) Subject's position in frame is slightly off-center.",
            },
            "refined_prompt": f"{current_prompt}{suffix}",
        }


def create_vlm_client(config: Dict[str, Any]) -> Any:
    """
    Factory to create VLM client from config.

    Default: DashScope Qwen-VL.
    """
    provider = config.get("provider", "dashscope")
    model_name = config.get("model_name", "qwen-vl-max")

    if provider == "mock":
        return MockVLMClient()

    # Default: DashScope Qwen-VL
    try:
        return VLMClient(
            model_name=model_name,
            api_key=os.environ.get(config.get("api_key_env", "DASHSCOPE_API_KEY")),
            base_url=config.get("base_url"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
            max_retries=config.get("max_retries", 3),
            use_video_mode=config.get("use_video_mode", True),
        )
    except (ImportError, ValueError) as e:
        logger.warning(f"VLM client init failed: {e}, using MockVLMClient")
        return MockVLMClient()
