"""
VLM (Vision-Language Model) Client for P-Flow.

Implements the VLM instruction from Listing 1 of the paper (arXiv:2603.22091).

The VLM receives a composite video (vertical stacking: reference/previous/current)
as key frames and outputs structured JSON with:
- analysis: {reference_description, last_generated_description, new_generated_description, comparison}
- refined_prompt: the improved T2V prompt

Uses DashScope API (Qwen3-VL-Flash) for vision-language analysis.
NO confidence score — paper does NOT use confidence-based stopping.

Reference: Section 3.4, Listing 1, and Appendix A of the paper.
"""

import json
import base64
import os
import io
from typing import Optional, Dict, List, Any
from pathlib import Path

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# Paper Listing 1: VLM Structured Instruction
# =============================================================================

SYSTEM_INSTRUCTION = """You are a professional visual effects video prompt engineer. Your task is to compare a reference video with generated videos and produce a refined text-to-video prompt that better captures the visual effects shown in the reference.

You will receive key frames extracted from a composite video arranged vertically:
- Panel A (top): Reference video — the target visual effect to reproduce
- Panel B (middle): Previously generated video (from last iteration, may be absent in iteration 1)
- Panel C (bottom): Newly generated video (from current iteration)

You also receive structured metadata about the current optimization state.

Your goal: Analyze visual effect differences and output an improved prompt that makes the T2V model generate videos closer to the reference's visual effects.

Focus on these aspects of visual effects:
1. Motion patterns: speed, direction, trajectories, oscillation, acceleration
2. Visual appearance: colors, textures, opacity, glow, particles, shapes
3. Spatial distribution: where effects appear, density, spread patterns
4. Temporal dynamics: how effects evolve over time, periodicity, onset/offset
5. Interactions: how effects interact with the scene/subject

Output ONLY a valid JSON object (no markdown, no extra text) with this exact structure:
{
    "analysis": {
        "reference_description": "Describe the visual effects observed in the reference video (Panel A)",
        "last_generated_description": "Describe the visual effects in the previous generation (Panel B), or 'N/A' if first iteration",
        "new_generated_description": "Describe the visual effects in the current generation (Panel C)",
        "comparison": "Detailed comparison of differences between reference and current generation, identifying specific aspects to improve"
    },
    "refined_prompt": "The complete, self-contained improved prompt for the T2V model. Must include ALL necessary details to reproduce the reference effect. Do not reference previous prompts."
}

Guidelines:
- Be specific and quantitative (e.g., "particles move 2x faster", "glow radius is 30% smaller")
- The refined_prompt must be self-contained — include every detail needed
- Focus on visual effect characteristics, not general scene content
- Use vivid, precise language that video generation models respond well to
- Each iteration should make targeted improvements based on the comparison
"""


def _build_user_message(
    desired_visual_effect: str,
    subject: str,
    environment: str,
    current_prompt: str,
    last_text_prompt: str,
    iteration: int,
    history_summary: str,
) -> str:
    """
    Build the structured user message following paper's Listing 1 placeholder format.

    Args:
        desired_visual_effect: User's description of the target effect.
        subject: Main subject of the video.
        environment: Background/setting description.
        current_prompt: Current T2V prompt (P_i).
        last_text_prompt: Previous iteration's prompt (P_{i-1}).
        iteration: Current iteration number.
        history_summary: Summary of optimization history.

    Returns:
        Formatted user message string.
    """
    msg = f"""## Optimization State

- Iteration: {iteration}/10
- Desired visual effect: {desired_visual_effect}
- Subject: {subject}
- Environment: {environment}

## Current T2V Prompt (to improve)
{current_prompt}

## Previous Iteration Prompt
{last_text_prompt if last_text_prompt else "N/A (first iteration)"}

## Optimization History Summary
{history_summary if history_summary else "First iteration — no history yet."}

## Instructions
The images below are key frames from the composite video (vertical layout):
- Top section (Panel A): REFERENCE video — the target visual effect
- Middle section (Panel B): PREVIOUS generated video (last iteration)
- Bottom section (Panel C): CURRENT generated video (this iteration)

Analyze the visual effect differences and provide a refined prompt as JSON."""
    return msg


class VLMClient:
    """
    VLM client using DashScope API (Qwen3-VL-Flash).

    Implements paper's VLM interaction:
    - Receives composite video key frames (vertical: ref/prev/current)
    - Outputs structured analysis + refined prompt
    - NO confidence score
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Args:
            model_name: DashScope VL model name.
                Options: "qwen-vl-max", "qwen-vl-plus", "qwen2.5-vl-72b-instruct"
                For Qwen3-VL-Flash use: "qwen-vl-max" or the specific model id.
            api_key: DashScope API key.
            base_url: API base URL (DashScope OpenAI-compatible endpoint).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
        """
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install: pip install openai")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

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

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        iteration: int = 1,
        desired_visual_effect: str = "",
        subject: str = "",
        environment: str = "",
        last_text_prompt: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze composite video frames and produce refined prompt.

        Follows paper Algorithm 1 step 6c-6d:
        - Send composite [V_ref | V_{i-1} | V_i] to VLM with history
        - Get refined prompt P_{i+1} and analysis A_i

        Args:
            composite_video_path: Path to vertical composite video.
            current_prompt: Current prompt P_i.
            iteration: Current iteration number (1-indexed).
            desired_visual_effect: User's target effect description.
            subject: Main subject description.
            environment: Scene/background description.
            last_text_prompt: Previous iteration prompt P_{i-1}.
            history: List of previous {prompt, analysis} dicts.

        Returns:
            Dict with:
                - 'analysis': nested dict {reference_description, last_generated_description,
                               new_generated_description, comparison}
                - 'refined_prompt': improved T2V prompt string
        """
        # Extract key frames from composite video
        frames_base64 = self._extract_frames_base64(composite_video_path, num_frames=6)

        # Build history summary
        history_summary = self._format_history(history)

        # Build structured user message (Listing 1 format)
        user_text = _build_user_message(
            desired_visual_effect=desired_visual_effect or current_prompt,
            subject=subject or "visual effect subject",
            environment=environment or "scene environment",
            current_prompt=current_prompt,
            last_text_prompt=last_text_prompt,
            iteration=iteration,
            history_summary=history_summary,
        )

        # Build multimodal content (text + images)
        content = [{"type": "text", "text": user_text}]
        for frame_b64 in frames_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                }
            })

        # Call VLM
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

    def _extract_frames_base64(self, video_path: str, num_frames: int = 6) -> List[str]:
        """
        Extract evenly-spaced frames from video and encode as base64 JPEG.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to extract.

        Returns:
            List of base64-encoded JPEG strings.
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
            img.save(buffer, format="JPEG", quality=80)
            frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return frames_b64

    def _format_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Format optimization history as concise text summary."""
        if not history:
            return ""

        lines = []
        for entry in history[-5:]:  # Keep last 5 iterations for context
            iter_num = entry.get("iteration", "?")
            prompt = entry.get("prompt", "")[:80]
            analysis = entry.get("analysis", {})
            if isinstance(analysis, dict):
                comparison = analysis.get("comparison", "")[:100]
            else:
                comparison = str(analysis)[:100]
            lines.append(f"  Iter {iter_num}: prompt=\"{prompt}...\" | feedback=\"{comparison}...\"")

        return "\n".join(lines)

    def _parse_response(self, response_text: str, fallback_prompt: str) -> Dict[str, Any]:
        """
        Parse VLM response into structured format.

        Expected output (from paper):
        {
            "analysis": {
                "reference_description": "...",
                "last_generated_description": "...",
                "new_generated_description": "...",
                "comparison": "..."
            },
            "refined_prompt": "..."
        }
        """
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
        """Validate and normalize the parsed response to match paper format."""
        # Ensure analysis is a nested dict
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


class MockVLMClient:
    """
    Mock VLM client for testing without API access.
    Simulates progressive prompt refinement following paper's output format.
    """

    def __init__(self, **kwargs):
        self.iteration = 0

    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        iteration: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        self.iteration += 1

        refinements = [
            ", with smooth flowing particle trails and natural acceleration",
            ", featuring luminous particles with varying speeds and organic trajectories",
            ", with precisely timed particle bursts that accelerate outward then decelerate naturally, creating fluid motion trails with soft glowing edges",
        ]

        suffix = refinements[min(self.iteration - 1, len(refinements) - 1)]

        return {
            "analysis": {
                "reference_description": f"Reference shows dynamic particle effects with natural motion patterns and soft glow.",
                "last_generated_description": f"Previous generation had somewhat rigid particle movement with uniform speed.",
                "new_generated_description": f"Current generation improved motion variety but still lacks the organic flow of reference.",
                "comparison": f"Main gap: particle acceleration patterns differ. Reference particles accelerate/decelerate naturally while generated ones move at near-constant speed. Glow intensity also 30% weaker.",
            },
            "refined_prompt": f"{current_prompt}{suffix}",
        }
