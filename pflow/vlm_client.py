"""
VLM (Vision-Language Model) Client for P-Flow.

This module handles interaction with Gemini 1.5 Pro (or compatible VLM)
for test-time prompt optimization. The VLM receives composite videos
showing reference and generated results, and outputs structured
analysis with refined prompts.

Reference: Section 3.4 and Appendix A of the paper.
"""

import json
import base64
import tempfile
import os
from typing import Optional, Dict, List, Any
from pathlib import Path

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# System instruction template for the VLM (from paper's Appendix A)
SYSTEM_INSTRUCTION = """You are a professional video generation prompt engineer. Your task is to analyze the differences between a reference video and a generated video, then provide an improved prompt that better captures the visual effects shown in the reference.

You will receive a composite video with the reference on the left and the generated result on the right (or specified otherwise). You also receive the history of previous prompts and analyses.

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
    Client for interacting with Vision-Language Models (Gemini 1.5 Pro).
    
    Handles:
    - Video upload and processing
    - Structured prompt engineering
    - Response parsing and validation
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Args:
            model_name: VLM model identifier.
            api_key: API key for Gemini. If None, reads from GOOGLE_API_KEY env var.
            temperature: Sampling temperature for VLM.
            max_tokens: Maximum output tokens.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not HAS_GENAI:
            raise ImportError(
                "google-generativeai is required. Install with: "
                "pip install google-generativeai"
            )
        
        # Configure API
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        
    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send composite video to VLM and get refined prompt.
        
        Args:
            composite_video_path: Path to the side-by-side composite video.
            current_prompt: The prompt used to generate the current video.
            history: List of previous {prompt, analysis} dictionaries.
            user_description: Original user description of desired effect.
            
        Returns:
            Dictionary with 'analysis', 'improvements', 'refined_prompt',
            'confidence', and 'key_differences'.
        """
        # Build the text prompt for VLM
        text_prompt = self._build_prompt(current_prompt, history, user_description)
        
        # Upload video
        video_file = genai.upload_file(composite_video_path)
        
        # Wait for video processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_file.state.name}")
        
        # Generate response
        response = self.model.generate_content(
            [video_file, text_prompt],
            request_options={"timeout": 120},
        )
        
        # Parse response
        result = self._parse_response(response.text)
        
        # Clean up uploaded file
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass
            
        return result
    
    def analyze_with_frames(
        self,
        reference_frames: List[str],
        generated_frames: List[str],
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Alternative: Send key frames as images instead of video.
        Useful when video upload is not supported or too slow.
        
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
        
        # Build content parts
        content_parts = [text_prompt]
        
        content_parts.append("\n--- REFERENCE VIDEO FRAMES ---\n")
        for frame_path in reference_frames:
            img_file = genai.upload_file(frame_path)
            content_parts.append(img_file)
            
        content_parts.append("\n--- GENERATED VIDEO FRAMES ---\n")
        for frame_path in generated_frames:
            img_file = genai.upload_file(frame_path)
            content_parts.append(img_file)
        
        # Generate response
        response = self.model.generate_content(
            content_parts,
            request_options={"timeout": 120},
        )
        
        return self._parse_response(response.text)
    
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
            "The video shows the REFERENCE (left/first) and GENERATED (right/second) results. "
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
        
        Args:
            response_text: Raw response text from VLM.
            
        Returns:
            Parsed dictionary with analysis results.
        """
        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            return self._validate_response(result)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
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
