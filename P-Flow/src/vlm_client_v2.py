"""
VLM Client V2 - Optimized for Video Reproduction with T2V Model Awareness.

Key improvements over V1:
1. Structured prompt template (subject→action→scene→camera→style)
2. Token budget control (60-80 words max for 1.3B model)
3. T2V capability awareness in instructions
4. Incremental refinement (modify top-1 difference per iteration)
5. Separated "what to describe" from "how to describe for T2V"

Compatible with DashScope API (Qwen-VL).
"""

import json
import base64
import os
import io
import time
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# V2 Structured Instruction - T2V Model Aware
# =============================================================================

SYSTEM_INSTRUCTION_V2 = """You are a text-to-video (T2V) prompt engineering expert. Your task: compare a reference video with a generated video, then output an improved T2V prompt.

## CRITICAL: T2V Model Constraints
The T2V model (Wan2.1-1.3B) has these limitations:
- Effective token window: ~60-80 English words (content beyond this is largely ignored)
- Cannot follow sequential instructions ("first X, then Y, finally Z" often fails)
- Responds best to: concrete nouns, vivid action verbs, spatial relationships, lighting descriptors
- Responds poorly to: abstract instructions, meta-commentary, quality adjectives ("detailed", "realistic")
- Single continuous action works best; multi-step narratives often collapse into the first action
- Camera descriptions are effective (e.g., "close-up", "wide shot", "tracking shot")

## Your Input
You receive 8 key frames from a composite video:
- Panel A (top): Reference video (TARGET to reproduce)
- Panel B (middle): Previous generation (may be absent in iter 1)  
- Panel C (bottom): Current generation

## Your Output Format
Output ONLY a valid JSON object with this exact structure:
{
    "top1_difference": "The single most prominent visual difference between reference and current generation (one sentence)",
    "modification": "What specific change to make in the prompt to fix the top1 difference (one sentence)",
    "refined_prompt": "The complete improved prompt (MUST follow the template below, 60-80 words max)"
}

## Prompt Template (MUST follow this order)
[SUBJECT]: who/what is the main focus, appearance details
[ACTION]: what motion/movement is happening, direction
[SCENE]: background, setting, objects in environment  
[CAMERA]: shot type, angle, camera movement
[STYLE]: lighting, color palette, atmosphere

Write the refined_prompt as a SINGLE flowing paragraph following this order. Do NOT use labels or bullet points in the prompt itself. Prioritize the first 40 words (they carry the most weight).

## Rules
- Keep prompt under 80 words. Shorter is better if it captures the essence.
- Front-load the most important visual elements (subject + action first)
- Use concrete visual descriptors, not instructions or meta-language
- Do NOT say "ensure", "maintain", "the scene should show" — just describe what IS there
- Each iteration: fix the TOP-1 difference only. Do not rewrite everything.
- If previous prompt was close, make minimal targeted edits
"""


SYSTEM_INSTRUCTION_INITIAL = """You are a video description expert. Given key frames from a video, write a concise T2V (text-to-video) generation prompt.

## Rules
- Maximum 60-80 words
- Follow this order: SUBJECT (who/what) → ACTION (motion) → SCENE (background) → CAMERA (angle/movement) → STYLE (lighting/mood)
- Use concrete visual descriptors: colors, shapes, directions, positions
- Describe ONE continuous action, not a sequence of events
- Front-load the most distinctive visual element
- Do NOT use instructions like "ensure" or "maintain"
- Write as a single flowing paragraph

## T2V Model Capabilities (Wan2.1-1.3B)
- Good at: single subject, consistent motion, natural scenes, specific lighting
- Limited at: multiple characters interacting, complex narratives, fine details, causal sequences
- If the video has a complex story, describe only the MOST visually dominant moment/action
"""


def _build_user_message_v2(
    current_prompt: str,
    iteration: int,
    i_max: int,
    top_differences_history: str = "",
) -> str:
    """Build concise user message for V2."""
    msg = f"""Iteration {iteration}/{i_max}

Current prompt (to improve):
"{current_prompt}"

Previous differences fixed: {top_differences_history if top_differences_history else "None (first iteration)"}

Look at the frames below. Panel A = reference (target). Panel C = current generation.
Identify the TOP-1 remaining visual difference and fix it in the prompt. Keep total prompt ≤80 words."""
    return msg


class VLMClientV2:
    """
    VLM Client V2 - Optimized prompt generation for T2V models.
    
    Key differences from V1:
    - Structured output with top1_difference tracking
    - Token budget enforcement
    - T2V capability-aware instructions
    - Incremental modification strategy
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.4,  # Lower temp for more focused output
        max_tokens: int = 1024,    # Shorter output needed
        max_retries: int = 3,
        max_prompt_words: int = 80,
    ):
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install: pip install openai")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_prompt_words = max_prompt_words

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DashScope API key required. Set DASHSCOPE_API_KEY env var.")

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
        i_max: int = 10,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs,  # Accept and ignore legacy params
    ) -> Dict[str, Any]:
        """
        Analyze composite video and produce refined prompt.
        
        V2: Focused on top-1 difference, token-budgeted output.
        """
        frames_base64 = self._extract_frames_base64(composite_video_path, num_frames=8)
        if not frames_base64:
            logger.warning("No frames extracted, returning fallback")
            return self._fallback_response(current_prompt)

        # Build history of what was fixed
        top_diffs_history = self._format_diff_history(history)

        # Build user message
        user_text = _build_user_message_v2(
            current_prompt=current_prompt,
            iteration=iteration,
            i_max=i_max,
            top_differences_history=top_diffs_history,
        )

        # Build multimodal content
        content = [{"type": "text", "text": user_text}]
        for frame_b64 in frames_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })

        # Call VLM
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION_V2},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_text = response.choices[0].message.content
                result = self._parse_response(response_text, current_prompt)
                
                # Enforce token budget
                result["refined_prompt"] = self._enforce_word_limit(
                    result["refined_prompt"], self.max_prompt_words
                )
                return result

            except Exception as e:
                logger.warning(f"VLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return self._fallback_response(current_prompt)

    def generate_initial_prompt(
        self,
        reference_video_path: str,
        user_hint: str = "",
    ) -> str:
        """
        Generate initial prompt from reference video (V2: structured, concise).
        """
        import numpy as np

        logger.info("Generating initial prompt from reference video (V2)...")

        frames_b64 = self._extract_video_frames(reference_video_path, num_frames=8)
        if not frames_b64:
            return user_hint or "A video scene."

        user_msg = "Describe this video for T2V generation. Follow the SUBJECT→ACTION→SCENE→CAMERA→STYLE order. Max 80 words, single paragraph."
        if user_hint:
            user_msg += f"\nContext hint: {user_hint}"

        content = [{"type": "text", "text": user_msg}]
        for frame_b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION_INITIAL},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            initial_prompt = response.choices[0].message.content.strip()
            # Remove any JSON wrapping if VLM outputs JSON
            if initial_prompt.startswith("{"):
                try:
                    parsed = json.loads(initial_prompt)
                    initial_prompt = parsed.get("refined_prompt", parsed.get("prompt", initial_prompt))
                except json.JSONDecodeError:
                    pass
            
            initial_prompt = self._enforce_word_limit(initial_prompt, self.max_prompt_words)
            logger.info(f"  Generated initial prompt ({len(initial_prompt.split())} words): {initial_prompt[:100]}...")
            return initial_prompt
        except Exception as e:
            logger.warning(f"Failed to generate initial prompt: {e}")
            return user_hint or "A video scene."

    def _extract_frames_base64(self, video_path: str, num_frames: int = 8) -> List[str]:
        """Extract frames from composite video as base64 JPEG."""
        return self._extract_video_frames(video_path, num_frames)

    def _extract_video_frames(self, video_path: str, num_frames: int = 8) -> List[str]:
        """Extract evenly-spaced frames and encode as base64."""
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
            max_dim = 1280
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return frames_b64

    def _format_diff_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Format history as list of previously fixed differences."""
        if not history:
            return ""
        
        diffs = []
        for entry in history[-5:]:
            analysis = entry.get("analysis", {})
            if isinstance(analysis, dict):
                top1 = analysis.get("top1_difference", "")
                if top1:
                    diffs.append(f"Iter {entry.get('iteration', '?')}: {top1}")
        
        return " | ".join(diffs) if diffs else ""

    def _enforce_word_limit(self, prompt: str, max_words: int) -> str:
        """Truncate prompt to max_words while keeping sentence integrity."""
        words = prompt.split()
        if len(words) <= max_words:
            return prompt
        
        # Truncate at last complete sentence within limit
        truncated = " ".join(words[:max_words])
        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        if last_period > len(truncated) * 0.6:  # At least 60% of content
            truncated = truncated[:last_period + 1]
        
        logger.info(f"  Prompt truncated: {len(words)} → {len(truncated.split())} words")
        return truncated

    def _parse_response(self, response_text: str, fallback_prompt: str) -> Dict[str, Any]:
        """Parse VLM response (V2 format)."""
        import re

        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            return self._validate_response_v2(result, fallback_prompt)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return self._validate_response_v2(result, fallback_prompt)
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return self._validate_response_v2(result, fallback_prompt)
            except json.JSONDecodeError:
                pass

        # Fallback
        logger.warning("Failed to parse VLM V2 response as JSON")
        return {
            "analysis": {
                "top1_difference": "parse_error",
                "modification": "",
                "reference_description": "",
                "comparison": response_text[:300],
            },
            "refined_prompt": fallback_prompt,
            "parse_error": True,
        }

    def _validate_response_v2(self, result: Dict[str, Any], fallback_prompt: str) -> Dict[str, Any]:
        """Validate V2 response format."""
        top1 = str(result.get("top1_difference", ""))
        modification = str(result.get("modification", ""))
        refined = str(result.get("refined_prompt", fallback_prompt))

        if not refined.strip():
            refined = fallback_prompt

        return {
            "analysis": {
                "top1_difference": top1,
                "modification": modification,
                "reference_description": "",
                "last_generated_description": "",
                "new_generated_description": "",
                "comparison": f"Top-1 diff: {top1}. Fix: {modification}",
            },
            "refined_prompt": refined,
        }

    def _fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Fallback when VLM is unavailable."""
        return {
            "analysis": {
                "top1_difference": "[VLM unavailable]",
                "modification": "",
                "reference_description": "",
                "last_generated_description": "",
                "new_generated_description": "",
                "comparison": "[unavailable]",
            },
            "refined_prompt": prompt,
            "vlm_error": True,
        }


def create_vlm_client_v2(config: Dict[str, Any]) -> Any:
    """Factory to create VLM V2 client."""
    provider = config.get("provider", "dashscope")
    model_name = config.get("model_name", "qwen-vl-max")

    if provider == "mock":
        from .vlm_client import MockVLMClient
        return MockVLMClient()

    try:
        return VLMClientV2(
            model_name=model_name,
            api_key=os.environ.get(config.get("api_key_env", "DASHSCOPE_API_KEY")),
            base_url=config.get("base_url"),
            temperature=config.get("temperature", 0.4),
            max_tokens=config.get("max_tokens", 1024),
            max_retries=config.get("max_retries", 3),
            max_prompt_words=config.get("max_prompt_words", 80),
        )
    except (ImportError, ValueError) as e:
        logger.warning(f"VLM V2 client init failed: {e}")
        from .vlm_client import MockVLMClient
        return MockVLMClient()
