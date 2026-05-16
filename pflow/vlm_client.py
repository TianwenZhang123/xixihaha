"""
VLM (Vision-Language Model) Client for P-Flow.

This module handles interaction with VLMs for test-time prompt optimization.
The VLM receives video frames (extracted as images) showing reference and
generated results, and outputs structured analysis with refined prompts.

Uses OpenAI-compatible API format via proxy/relay service (e.g., LinkAPI),
which supports Gemini, GPT-4o, Claude, etc. through a unified interface.

Reference: Section 3.4 and Appendix A of the paper.
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


# System instruction template for the VLM (from paper's Appendix A)
SYSTEM_INSTRUCTION = """You are a professional video generation prompt engineer. Your task is to analyze the differences between a reference video and a generated video, then provide an improved prompt that better captures the visual effects shown in the reference.

You will receive key frames from a composite video with the reference on the left and the generated result on the right (or specified otherwise). You also receive the history of previous prompts and analyses.

Your analysis should focus on:
1. **Motion patterns**: Speed, direction, trajectories, oscillation, acceleration
2. **Visual appearance**: Colors, textures, opacity, glow, particles, shapes
3. **Spatial distribution**: Where effects appear, density, spread patterns
4. **Temporal dynamics**: How effects evolve over time, periodicity, onset/offset
5. **Interactions**: How effects interact with the scene/subject

Output your response as a JSON object with the following structure:
{
    "analysis": "Detailed analysis of differences between reference and generated video",
    "improvements": ["List of specific improvements needed"],
    "refined_prompt": "The complete improved prompt for video generation",
    "confidence": 0.0-1.0,
    "key_differences": ["Most important visual differences to address"]
}

Guidelines:
- Be specific and quantitative when possible (e.g., "particles move 2x faster")
- Focus on visual effect characteristics, not scene content
- The refined prompt should be self-contained (don't reference previous prompts)
- Include all details needed to reproduce the effect in a single prompt
- Use vivid, precise language that video generation models respond well to
"""


class VLMClient:
    """
    VLM client using OpenAI-compatible API (via proxy/relay).
    
    Supports any model accessible through OpenAI-format API, including:
    - Gemini 2.0 Flash / 1.5 Pro (via LinkAPI or similar relay)
    - GPT-4o / GPT-4o-mini (via OpenAI directly or relay)
    - Claude 3.5 Sonnet (via relay)
    
    Video analysis is done by extracting key frames and sending them
    as base64 images to the vision model.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Args:
            model_name: Model identifier (e.g., "gemini-2.0-flash", "gpt-4o").
            api_key: API key for the relay service.
                     If None, reads from OPENAI_API_KEY env var.
            base_url: API base URL (e.g., "https://api.linkapi.org/v1").
                      If None, reads from OPENAI_BASE_URL env var or defaults
                      to OpenAI official endpoint.
            temperature: Sampling temperature for VLM.
            max_tokens: Maximum output tokens.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = openai.OpenAI(**client_kwargs)
    
    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze composite video (via extracted frames) and refine prompt.
        
        Extracts key frames from the video and sends them as images to
        the VLM for analysis.
        
        Args:
            composite_video_path: Path to the side-by-side composite video.
            current_prompt: The prompt used to generate the current video.
            history: List of previous {prompt, analysis} dictionaries.
            user_description: Original user description of desired effect.
            
        Returns:
            Dictionary with 'analysis', 'improvements', 'refined_prompt',
            'confidence', and 'key_differences'.
        """
        # Extract key frames from the composite video
        frames_base64 = self._extract_frames_base64(composite_video_path, num_frames=8)
        
        # Build messages
        text_prompt = self._build_prompt(current_prompt, history, user_description)
        
        # Build content with images
        content = [{"type": "text", "text": text_prompt}]
        for frame_b64 in frames_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "high",
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
        return self._parse_response(response_text)
    
    def analyze_with_frames(
        self,
        reference_frames: List[str],
        generated_frames: List[str],
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send key frames as images for analysis.
        
        Args:
            reference_frames: List of paths to reference video key frames.
            generated_frames: List of paths to generated video key frames.
            current_prompt: Current generation prompt.
            history: Previous prompt/analysis history.
            user_description: User's effect description.
            
        Returns:
            Structured analysis and refined prompt.
        """
        text_prompt = self._build_prompt(current_prompt, history, user_description)
        text_prompt += (
            "\n\nThe images below show key frames. "
            "First set is from the REFERENCE video, second set is from the GENERATED video. "
            "Analyze the visual effect differences between them."
        )
        
        content = [{"type": "text", "text": text_prompt}]
        
        # Add reference frames
        content.append({"type": "text", "text": "--- REFERENCE VIDEO FRAMES ---"})
        for frame_path in reference_frames:
            frame_b64 = self._encode_image(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"}
            })
        
        # Add generated frames
        content.append({"type": "text", "text": "--- GENERATED VIDEO FRAMES ---"})
        for frame_path in generated_frames:
            frame_b64 = self._encode_image(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"}
            })
        
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
        return self._parse_response(response_text)
    
    def _extract_frames_base64(self, video_path: str, num_frames: int = 8) -> List[str]:
        """
        Extract evenly-spaced frames from video and encode as base64 JPEG.
        
        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to extract (default 8 for good temporal coverage).
            
        Returns:
            List of base64-encoded JPEG strings.
        """
        import numpy as np
        
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
        except (ImportError, Exception):
            import imageio.v3 as iio
            all_frames = iio.imread(video_path, plugin="pyav")
            total_frames = len(all_frames)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = all_frames[indices]
        
        # Encode each frame as JPEG base64
        from PIL import Image
        
        frames_b64 = []
        for frame in frames:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        
        return frames_b64
    
    def _encode_image(self, image_path: str) -> str:
        """Encode a single image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _build_prompt(
        self,
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> str:
        """
        Build the text prompt for VLM including history context.
        
        Args:
            current_prompt: Current generation prompt.
            history: Previous iterations' prompts and analyses.
            user_description: User's original description.
            
        Returns:
            Formatted text prompt string.
        """
        parts = []
        
        if user_description:
            parts.append(f"## User's Desired Effect Description\n{user_description}\n")
        
        parts.append(f"## Current Generation Prompt\n{current_prompt}\n")
        
        if history:
            parts.append("## Optimization History")
            for i, h in enumerate(history):
                parts.append(f"\n### Iteration {i + 1}")
                if "prompt" in h:
                    parts.append(f"Prompt: {h['prompt']}")
                if "analysis" in h:
                    parts.append(f"Analysis: {h['analysis']}")
                if "improvements" in h:
                    parts.append(f"Improvements made: {', '.join(h['improvements'])}")
            parts.append("")
        
        parts.append(
            "## Task\n"
            "The images show key frames from the composite video with "
            "REFERENCE (left half) and GENERATED (right half) results side by side. "
            "Frames are ordered chronologically to show temporal progression.\n"
            "Analyze the differences in visual effects and provide an improved prompt.\n"
            "Focus on: motion patterns, visual appearance, spatial distribution, "
            "temporal dynamics, and effect interactions.\n"
            "Output as JSON with: analysis, improvements, refined_prompt, confidence, key_differences."
        )
        
        return "\n".join(parts)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse VLM response into structured format.
        
        Handles potential JSON formatting issues gracefully.
        """
        import re
        
        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            return self._validate_response(result)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return self._validate_response(result)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return self._validate_response(result)
            except json.JSONDecodeError:
                pass
        
        # Fallback: create structured response from raw text
        return {
            "analysis": response_text,
            "improvements": [],
            "refined_prompt": "",
            "confidence": 0.0,
            "key_differences": [],
            "parse_error": True,
        }
    
    def _validate_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the parsed response.
        
        Ensures all required fields are present with correct types.
        """
        validated = {
            "analysis": str(result.get("analysis", "")),
            "improvements": list(result.get("improvements", [])),
            "refined_prompt": str(result.get("refined_prompt", "")),
            "confidence": float(result.get("confidence", 0.5)),
            "key_differences": list(result.get("key_differences", [])),
        }
        
        # Ensure confidence is in [0, 1]
        validated["confidence"] = max(0.0, min(1.0, validated["confidence"]))
        
        return validated


class MockVLMClient:
    """
    Mock VLM client for testing without API access.
    Returns a slightly modified prompt each iteration.
    """
    
    def __init__(self, **kwargs):
        self.iteration = 0
        
    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.iteration += 1
        
        # Simulate progressive refinement
        refinements = [
            "with dynamic particle effects flowing smoothly",
            "featuring vibrant glowing particles with natural motion",
            "with luminous particles that accelerate and decelerate naturally, creating fluid motion trails",
        ]
        
        suffix = refinements[min(self.iteration - 1, len(refinements) - 1)]
        
        return {
            "analysis": f"Iteration {self.iteration}: Generated video lacks the precise motion dynamics of the reference.",
            "improvements": [f"Add motion detail: {suffix}"],
            "refined_prompt": f"{current_prompt}, {suffix}",
            "confidence": min(0.5 + self.iteration * 0.1, 0.95),
            "key_differences": ["Motion speed", "Particle density", "Glow intensity"],
        }
    
    def analyze_with_frames(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze_and_refine("", args[2] if len(args) > 2 else "")
