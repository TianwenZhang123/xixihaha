"""
VISTA-Style Test-Time Self-Improving Prompt Optimization Module.

Implements the VISTA paper's multi-agent self-improvement framework
adapted for P-Flow's visual effects generation task.

Key Components (from VISTA, arXiv:2510.15831):
1. Structured Video Prompt Planning (SVPP) - decompose prompt into timed scenes
2. Binary Tournament Selection with Probing Critiques - pairwise comparison
3. MMAC (Multi-Dimensional Multi-Agent Critiques) - triadic court per dimension
4. DTPA (Deep Thinking Prompting Agent) - introspective prompt refinement

This module is designed as a drop-in replacement for prompt_optimizer.py
to enable ablation experiments comparing P-Flow's simple VLM optimization
with VISTA's multi-agent approach.

Usage in pipeline.py:
    # Replace:
    #   from .prompt_optimizer import PromptOptimizer
    # With:
    #   from .vista_optimizer import VISTAOptimizer as PromptOptimizer

Reference: "VISTA: Enhancing Long-Duration and High-Quality Video Generation
via Multi-Agent Test-Time Self-Improving" (arXiv:2510.15831)
"""

import os
import json
import copy
import random
import time
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

import torch

from .vlm_client import VLMClient, MockVLMClient
from .video_utils import (
    create_composite_video,
    save_video_tensor,
    extract_key_frames,
)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SceneSegment:
    """A single scene segment in the structured video prompt plan."""
    scene_id: int
    start_time: float  # seconds
    end_time: float  # seconds
    subject: str
    action: str
    environment: str
    camera_movement: str
    lighting: str
    color_palette: str
    visual_effects: str
    mood: str
    transition: str  # how this scene transitions to the next

    def to_prompt_text(self) -> str:
        """Convert scene segment to natural language prompt."""
        parts = []
        time_str = f"[{self.start_time:.1f}s - {self.end_time:.1f}s]"
        parts.append(time_str)
        if self.subject:
            parts.append(self.subject)
        if self.action:
            parts.append(self.action)
        if self.environment:
            parts.append(f"in {self.environment}")
        if self.visual_effects:
            parts.append(f"with {self.visual_effects}")
        if self.lighting:
            parts.append(f"under {self.lighting}")
        if self.color_palette:
            parts.append(f"({self.color_palette} color palette)")
        if self.camera_movement:
            parts.append(f"camera: {self.camera_movement}")
        if self.mood:
            parts.append(f"mood: {self.mood}")
        return ", ".join(parts)


@dataclass
class VideoCandidate:
    """A candidate video with its prompt and evaluation scores."""
    video: Optional[torch.Tensor]  # (C, F, H, W)
    video_path: Optional[str]
    prompt: str
    scene_plan: Optional[List[SceneSegment]]
    scores: Dict[str, float] = field(default_factory=dict)  # dimension -> score
    overall_score: float = 0.0
    critiques: Dict[str, str] = field(default_factory=dict)  # dimension -> critique text
    iteration: int = 0


@dataclass
class JudgeVerdict:
    """Verdict from a single judge in the MMAC system."""
    judge_role: str  # "normal", "adversarial", "meta"
    dimension: str  # "visual", "motion", "temporal"
    preferred: str  # "A" or "B"
    score_a: float
    score_b: float
    reasoning: str
    confidence: float


# ============================================================================
# VISTA System Prompts (adapted for visual effects generation)
# ============================================================================

SVPP_SYSTEM_PROMPT = """You are a professional video prompt engineer specializing in visual effects.
Your task is to decompose a visual effect description into a structured scene plan.

Given a user's description of a visual effect and a reference video analysis, create a structured
plan that breaks the effect into timed segments.

For each scene segment, specify:
1. start_time / end_time: Time range in seconds
2. subject: What is the main subject/element
3. action: What motion/change is happening
4. environment: Background/setting description
5. camera_movement: Camera motion (static, pan, zoom, etc.)
6. lighting: Lighting conditions and changes
7. color_palette: Dominant colors
8. visual_effects: Specific VFX details (particles, glow, distortion, etc.)
9. mood: Emotional tone
10. transition: How this flows into the next segment

Output as JSON array of scene segments.
"""

PAIRWISE_CRITIQUE_PROMPT = """You are an expert video quality judge. You will compare two generated videos (A and B)
against a reference video showing the desired visual effect.

Your task: Determine which video better captures the visual effect from the reference.

Evaluation criteria:
- Motion fidelity: Does the motion match the reference's dynamics?
- Visual appearance: Do colors, textures, and effects match?
- Temporal consistency: Is the effect smooth and coherent over time?
- Effect accuracy: Does it capture the specific VFX characteristics?

First state your initial preference, then consider the opposite perspective ("probing critique"),
and finally give your final verdict.

Output JSON:
{
    "initial_preference": "A" or "B",
    "probing_critique": "Why the other video might actually be better...",
    "final_preference": "A" or "B",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation"
}
"""

NORMAL_JUDGE_PROMPT = """You are a fair and balanced video quality judge.
Evaluate the video on the {dimension} dimension by comparing to the reference.
Provide an honest assessment of strengths and weaknesses.

Output JSON:
{{
    "score": 0.0-10.0,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "reasoning": "explanation"
}}
"""

ADVERSARIAL_JUDGE_PROMPT = """You are a critical, adversarial video quality judge.
Your role is to find flaws and issues that others might miss.
Evaluate the video on the {dimension} dimension compared to the reference.
Be deliberately critical and look for subtle problems.

Output JSON:
{{
    "score": 0.0-10.0,
    "critical_issues": ["list of problems found"],
    "subtle_flaws": ["things that seem fine but aren't"],
    "reasoning": "explanation"
}}
"""

META_JUDGE_PROMPT = """You are a meta-judge overseeing two other judges' assessments.
Judge A (Normal) gave: {normal_verdict}
Judge B (Adversarial) gave: {adversarial_verdict}

Synthesize both perspectives to provide a final balanced assessment on the {dimension} dimension.
Consider which judge's points are more valid and why.

Output JSON:
{{
    "final_score": 0.0-10.0,
    "normal_judge_validity": 0.0-1.0,
    "adversarial_judge_validity": 0.0-1.0,
    "synthesis": "Balanced assessment combining both perspectives",
    "key_issues": ["Most important issues to address"]
}}
"""

DTPA_PROMPT = """You are a deep-thinking prompt engineering agent. You must refine the video generation
prompt through careful introspective reasoning.

Follow these 6 steps strictly:

1. **Observe**: What specific visual differences exist between reference and generated video?
2. **Identify**: What are the root causes of these differences in the prompt?
3. **Hypothesize**: What prompt changes would most effectively address these differences?
4. **Evaluate**: For each hypothesis, what are potential side effects or risks?
5. **Synthesize**: Combine the best hypotheses into a coherent prompt revision.
6. **Verify**: Does the revised prompt maintain all existing good qualities?

Current prompt: {current_prompt}
MMAC Critiques Summary: {critiques_summary}
Previous iterations feedback: {history_summary}

Output JSON:
{{
    "step_1_observe": "observations about reference vs generated",
    "step_2_identify": "root causes in the prompt",
    "step_3_hypothesize": ["list of change hypotheses"],
    "step_4_evaluate": ["evaluation of each hypothesis"],
    "step_5_synthesize": "combined reasoning",
    "step_6_verify": "verification of preserved qualities",
    "refined_prompt": "The complete refined prompt",
    "confidence": 0.0-1.0,
    "key_changes": ["list of main changes made"]
}}
"""


# ============================================================================
# VISTA Optimizer Class
# ============================================================================

class VISTAOptimizer:
    """
    VISTA-Style Multi-Agent Test-Time Self-Improving Prompt Optimizer.

    Replaces P-Flow's simple single-VLM optimization with VISTA's
    multi-agent framework:

    1. SVPP: Structured prompt planning for scene decomposition
    2. Binary Tournament: Pairwise selection with probing critiques
    3. MMAC: Multi-dimensional multi-agent critiques (triadic court)
    4. DTPA: Deep thinking prompt refinement

    Interface-compatible with PromptOptimizer for drop-in replacement.
    """

    # Evaluation dimensions (adapted from VISTA's Visual/Audio/Context
    # to Visual Effects domain)
    DIMENSIONS = ["visual", "motion", "temporal"]

    def __init__(
        self,
        vlm_client: Optional[VLMClient] = None,
        max_iterations: int = 5,
        candidates_per_iteration: int = 3,
        output_dir: str = "outputs",
        use_mock: bool = False,
        enable_svpp: bool = True,
        enable_tournament: bool = True,
        enable_mmac: bool = True,
        enable_dtpa: bool = True,
        video_duration: float = 5.0,
    ):
        """
        Args:
            vlm_client: VLM client for multi-agent calls.
            max_iterations: Number of self-improvement iterations.
            candidates_per_iteration: Videos generated per iteration for tournament.
            output_dir: Output directory.
            use_mock: Use mock VLM for testing.
            enable_svpp: Enable Structured Video Prompt Planning.
            enable_tournament: Enable Binary Tournament Selection.
            enable_mmac: Enable MMAC multi-agent critiques.
            enable_dtpa: Enable Deep Thinking Prompt Agent.
            video_duration: Assumed video duration in seconds for scene planning.
        """
        if vlm_client is not None:
            self.vlm_client = vlm_client
        elif use_mock:
            self.vlm_client = MockVistaVLMClient()
        else:
            self.vlm_client = VLMClient()

        self.max_iterations = max_iterations
        self.candidates_per_iteration = candidates_per_iteration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature toggles for ablation
        self.enable_svpp = enable_svpp
        self.enable_tournament = enable_tournament
        self.enable_mmac = enable_mmac
        self.enable_dtpa = enable_dtpa
        self.video_duration = video_duration

        # State tracking
        self._iteration_count = 0
        self._history: List[Dict[str, Any]] = []

    # ========================================================================
    # Public Interface (compatible with PromptOptimizer)
    # ========================================================================

    def optimize_prompt(
        self,
        initial_prompt: str,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        iteration: int,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
        previous_video: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform one VISTA-style optimization iteration.

        Interface-compatible with PromptOptimizer.optimize_prompt().

        The VISTA approach for a single iteration:
        1. (SVPP) Decompose prompt into scene structure
        2. (MMAC) Multi-agent evaluation of the generated video
        3. (DTPA) Deep thinking refinement of the prompt

        Note: Binary Tournament requires multiple candidates, so it's used
        when called via optimize_with_candidates() instead.

        Args:
            initial_prompt: Current prompt P_i.
            reference_video: Reference video tensor.
            generated_video: Current generated video tensor.
            iteration: Current iteration number.
            history: Previous iteration results.
            user_description: Original user description.
            previous_video: Previous iteration's video.

        Returns:
            Dict with 'refined_prompt', 'analysis', 'improvements',
            'confidence', 'key_differences' (compatible with PromptOptimizer).
        """
        history = history or []
        self._iteration_count = iteration

        # Step 1: Structured Video Prompt Planning (first iteration only)
        scene_plan = None
        if self.enable_svpp and iteration == 1:
            scene_plan = self._structured_prompt_planning(
                user_description or initial_prompt,
                reference_video,
            )

        # Step 2: MMAC Multi-Agent Critique
        mmac_result = None
        if self.enable_mmac:
            mmac_result = self._mmac_evaluate(
                reference_video=reference_video,
                generated_video=generated_video,
                current_prompt=initial_prompt,
                iteration=iteration,
            )

        # Step 3: DTPA Deep Thinking Refinement
        if self.enable_dtpa:
            dtpa_result = self._dtpa_refine(
                current_prompt=initial_prompt,
                mmac_critiques=mmac_result,
                history=history,
                reference_video=reference_video,
                generated_video=generated_video,
                scene_plan=scene_plan,
            )
        else:
            # Fallback to simple VLM refinement
            dtpa_result = self._simple_refine(
                initial_prompt, reference_video, generated_video, history
            )

        # Build result compatible with PromptOptimizer output
        result = {
            "refined_prompt": dtpa_result.get("refined_prompt", initial_prompt),
            "analysis": dtpa_result.get("step_1_observe", ""),
            "improvements": dtpa_result.get("key_changes", []),
            "confidence": dtpa_result.get("confidence", 0.5),
            "key_differences": dtpa_result.get(
                "key_changes",
                mmac_result.get("key_issues", []) if mmac_result else []
            ),
            # VISTA-specific fields
            "vista_mmac": mmac_result,
            "vista_dtpa": dtpa_result,
            "vista_scene_plan": [s.__dict__ for s in scene_plan] if scene_plan else None,
        }

        # Save iteration results
        self._save_iteration_results(iteration, initial_prompt, result)

        return result

    def optimize_with_candidates(
        self,
        candidates: List[VideoCandidate],
        reference_video: torch.Tensor,
        iteration: int,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        VISTA-style optimization with multiple candidate videos.

        This is the full VISTA pipeline including Binary Tournament Selection.
        Use when the pipeline generates multiple candidates per iteration.

        Args:
            candidates: List of VideoCandidate objects with videos and prompts.
            reference_video: Reference video tensor.
            iteration: Current iteration number.
            history: Previous iteration results.

        Returns:
            Dict with best candidate info and refined prompt.
        """
        history = history or []

        # Step 1: Binary Tournament Selection
        if self.enable_tournament and len(candidates) >= 2:
            best_candidate = self._binary_tournament_select(
                candidates=candidates,
                reference_video=reference_video,
            )
        else:
            # Default to first candidate
            best_candidate = candidates[0] if candidates else None

        if best_candidate is None:
            return {
                "refined_prompt": "",
                "analysis": "No valid candidates",
                "improvements": [],
                "confidence": 0.0,
                "key_differences": [],
            }

        # Step 2: MMAC on best candidate
        mmac_result = None
        if self.enable_mmac and best_candidate.video is not None:
            mmac_result = self._mmac_evaluate(
                reference_video=reference_video,
                generated_video=best_candidate.video,
                current_prompt=best_candidate.prompt,
                iteration=iteration,
            )

        # Step 3: DTPA refinement
        if self.enable_dtpa:
            dtpa_result = self._dtpa_refine(
                current_prompt=best_candidate.prompt,
                mmac_critiques=mmac_result,
                history=history,
                reference_video=reference_video,
                generated_video=best_candidate.video,
                scene_plan=best_candidate.scene_plan,
            )
        else:
            dtpa_result = {"refined_prompt": best_candidate.prompt, "confidence": 0.5}

        return {
            "refined_prompt": dtpa_result.get("refined_prompt", best_candidate.prompt),
            "analysis": dtpa_result.get("step_1_observe", ""),
            "improvements": dtpa_result.get("key_changes", []),
            "confidence": dtpa_result.get("confidence", 0.5),
            "key_differences": mmac_result.get("key_issues", []) if mmac_result else [],
            "best_candidate_prompt": best_candidate.prompt,
            "best_candidate_score": best_candidate.overall_score,
            "vista_mmac": mmac_result,
            "vista_dtpa": dtpa_result,
        }

    def should_stop_early(
        self,
        history: List[Dict[str, Any]],
        min_iterations: int = 3,
    ) -> bool:
        """
        Check convergence. Compatible with PromptOptimizer interface.

        VISTA uses a combination of:
        - Confidence plateauing
        - No significant critique improvements
        - Prompt stability (edit distance)
        """
        if len(history) < min_iterations:
            return False

        # Check confidence trend
        recent_confidence = [h.get("confidence", 0) for h in history[-3:]]
        if all(c > 0.92 for c in recent_confidence):
            return True

        # Check if improvements are diminishing
        recent_improvements = [h.get("improvements", []) for h in history[-2:]]
        if all(len(imp) == 0 for imp in recent_improvements):
            return True

        # Check prompt stability (if prompts are barely changing)
        if len(history) >= 3:
            recent_prompts = [h.get("prompt", "") for h in history[-3:]]
            if len(set(recent_prompts)) == 1:
                return True

        return False

    # ========================================================================
    # Component 1: Structured Video Prompt Planning (SVPP)
    # ========================================================================

    def _structured_prompt_planning(
        self,
        user_prompt: str,
        reference_video: torch.Tensor,
    ) -> List[SceneSegment]:
        """
        Decompose user prompt into a structured scene plan.

        From VISTA Section 2.1:
        Given a user prompt u, the planning agent decomposes it into
        a sequence of timed scenes with 9 properties each.

        For visual effects (P-Flow context), scenes represent temporal phases
        of the effect (onset, peak, decay, etc.).

        Args:
            user_prompt: User's natural language effect description.
            reference_video: Reference video for analysis.

        Returns:
            List of SceneSegment objects.
        """
        # Build the planning request
        planning_prompt = f"""Decompose this visual effect description into temporal scene segments:

User Description: {user_prompt}
Video Duration: {self.video_duration} seconds

Create 2-4 scene segments that represent the temporal phases of this visual effect.
For visual effects, think about: onset/build-up phase, main effect phase, 
transitions, and decay/conclusion phase.

Output as JSON array:
[
  {{
    "scene_id": 1,
    "start_time": 0.0,
    "end_time": 2.0,
    "subject": "main subject",
    "action": "what's happening",
    "environment": "setting",
    "camera_movement": "static/pan/zoom",
    "lighting": "lighting description",
    "color_palette": "colors",
    "visual_effects": "specific VFX details",
    "mood": "emotional tone",
    "transition": "how it flows to next"
  }}
]
"""
        # Create composite for reference analysis
        composite_path = self._save_temp_video(reference_video, "svpp_reference")

        try:
            response = self.vlm_client.analyze_and_refine(
                composite_video_path=str(composite_path),
                current_prompt=planning_prompt,
                history=[],
                user_description=f"[SYSTEM: {SVPP_SYSTEM_PROMPT}]\n\n{user_prompt}",
            )

            # Parse scene plan from response
            scenes = self._parse_scene_plan(response, user_prompt)
            return scenes

        except Exception as e:
            # Fallback: single scene for the whole duration
            return [SceneSegment(
                scene_id=1,
                start_time=0.0,
                end_time=self.video_duration,
                subject="visual effect",
                action=user_prompt,
                environment="",
                camera_movement="static",
                lighting="natural",
                color_palette="",
                visual_effects=user_prompt,
                mood="dynamic",
                transition="none",
            )]

    def _parse_scene_plan(
        self, response: Dict[str, Any], fallback_prompt: str
    ) -> List[SceneSegment]:
        """Parse VLM response into SceneSegment objects."""
        # The response might contain the scene plan in different fields
        raw_text = response.get("analysis", "") or response.get("refined_prompt", "")

        # Try to parse JSON from the text
        try:
            # Look for JSON array in response
            import re
            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                scenes_data = json.loads(json_match.group(0))
            else:
                scenes_data = json.loads(raw_text)

            scenes = []
            for s in scenes_data:
                scenes.append(SceneSegment(
                    scene_id=s.get("scene_id", len(scenes) + 1),
                    start_time=float(s.get("start_time", 0)),
                    end_time=float(s.get("end_time", self.video_duration)),
                    subject=s.get("subject", ""),
                    action=s.get("action", ""),
                    environment=s.get("environment", ""),
                    camera_movement=s.get("camera_movement", "static"),
                    lighting=s.get("lighting", ""),
                    color_palette=s.get("color_palette", ""),
                    visual_effects=s.get("visual_effects", ""),
                    mood=s.get("mood", ""),
                    transition=s.get("transition", ""),
                ))
            return scenes if scenes else [self._default_scene(fallback_prompt)]

        except (json.JSONDecodeError, ValueError, KeyError):
            return [self._default_scene(fallback_prompt)]

    def _default_scene(self, prompt: str) -> SceneSegment:
        """Create a default single-scene plan."""
        return SceneSegment(
            scene_id=1,
            start_time=0.0,
            end_time=self.video_duration,
            subject="visual effect",
            action=prompt,
            environment="",
            camera_movement="static",
            lighting="natural",
            color_palette="",
            visual_effects=prompt,
            mood="dynamic",
            transition="none",
        )

    # ========================================================================
    # Component 2: Binary Tournament Selection with Probing Critiques
    # ========================================================================

    def _binary_tournament_select(
        self,
        candidates: List[VideoCandidate],
        reference_video: torch.Tensor,
    ) -> VideoCandidate:
        """
        Select the best candidate through binary tournament with probing critiques.

        From VISTA Algorithm 2 (PairwiseSelect):
        1. Randomly pair candidates
        2. For each pair, VLM judges with initial preference
        3. Apply probing critique (consider the opposite)
        4. Bidirectional swap (show in both orders A|B and B|A)
        5. Final verdict after self-reflection

        Args:
            candidates: List of video candidates to compare.
            reference_video: Reference video for comparison.

        Returns:
            The winning candidate.
        """
        if len(candidates) == 1:
            return candidates[0]

        # Shuffle candidates for fair pairing
        tournament_pool = list(candidates)
        random.shuffle(tournament_pool)

        # Single-elimination tournament
        while len(tournament_pool) > 1:
            next_round = []
            for i in range(0, len(tournament_pool) - 1, 2):
                winner = self._pairwise_compare(
                    candidate_a=tournament_pool[i],
                    candidate_b=tournament_pool[i + 1],
                    reference_video=reference_video,
                )
                next_round.append(winner)

            # Handle odd number
            if len(tournament_pool) % 2 == 1:
                next_round.append(tournament_pool[-1])

            tournament_pool = next_round

        return tournament_pool[0]

    def _pairwise_compare(
        self,
        candidate_a: VideoCandidate,
        candidate_b: VideoCandidate,
        reference_video: torch.Tensor,
    ) -> VideoCandidate:
        """
        Compare two candidates with probing critiques and bidirectional swap.

        From VISTA Section 2.2:
        - Show [Ref | A | B] to VLM, get initial preference
        - Apply probing: "Why might the other be better?"
        - Swap order [Ref | B | A] and compare again (mitigate position bias)
        - Final verdict based on both comparisons

        Args:
            candidate_a: First candidate.
            candidate_b: Second candidate.
            reference_video: Reference video.

        Returns:
            The winning candidate.
        """
        # Direction 1: A first, B second
        verdict_ab = self._judge_pair(
            reference_video, candidate_a, candidate_b, order="AB"
        )

        # Direction 2: B first, A second (bidirectional swap)
        verdict_ba = self._judge_pair(
            reference_video, candidate_b, candidate_a, order="BA"
        )

        # Aggregate verdicts
        # In AB order: if preferred "A" → candidate_a wins
        # In BA order: if preferred "A" → candidate_b wins (since B is shown first)
        score_a = 0
        score_b = 0

        if verdict_ab.get("final_preference") == "A":
            score_a += verdict_ab.get("confidence", 0.5)
        else:
            score_b += verdict_ab.get("confidence", 0.5)

        if verdict_ba.get("final_preference") == "B":
            score_a += verdict_ba.get("confidence", 0.5)
        else:
            score_b += verdict_ba.get("confidence", 0.5)

        return candidate_a if score_a >= score_b else candidate_b

    def _judge_pair(
        self,
        reference_video: torch.Tensor,
        first: VideoCandidate,
        second: VideoCandidate,
        order: str,
    ) -> Dict[str, Any]:
        """
        Single-direction pairwise judgment with probing critique.

        Args:
            reference_video: Reference video.
            first: Video shown in position A.
            second: Video shown in position B.
            order: "AB" or "BA" for logging.

        Returns:
            Judgment dict with preference, confidence, reasoning.
        """
        # Create 3-panel composite: [Reference | First | Second]
        videos_to_compare = [reference_video]
        labels = ["Reference"]

        if first.video is not None:
            videos_to_compare.append(first.video)
            labels.append("Video A")
        if second.video is not None:
            videos_to_compare.append(second.video)
            labels.append("Video B")

        if len(videos_to_compare) < 3:
            # Cannot compare without videos
            return {"final_preference": "A", "confidence": 0.5, "reasoning": "insufficient data"}

        composite = create_composite_video(videos=videos_to_compare, labels=labels)
        composite_path = self._save_temp_video(composite, f"tournament_{order}")

        # Call VLM with pairwise critique prompt
        prompt_text = PAIRWISE_CRITIQUE_PROMPT + f"""

Video A prompt: {first.prompt}
Video B prompt: {second.prompt}

The composite video shows: Reference (left), Video A (middle), Video B (right).
Which video better matches the reference visual effects?
"""
        try:
            response = self.vlm_client.analyze_and_refine(
                composite_video_path=str(composite_path),
                current_prompt=prompt_text,
                history=[],
                user_description=None,
            )

            # Parse preference from response
            analysis = response.get("analysis", "")
            refined = response.get("refined_prompt", "")
            combined_text = f"{analysis} {refined}".lower()

            # Determine preference
            final_preference = "A"  # default
            if "video b" in combined_text and "better" in combined_text:
                final_preference = "B"
            elif "b wins" in combined_text or "prefer b" in combined_text:
                final_preference = "B"

            return {
                "final_preference": final_preference,
                "confidence": response.get("confidence", 0.5),
                "reasoning": analysis,
            }

        except Exception as e:
            return {"final_preference": "A", "confidence": 0.5, "reasoning": str(e)}

    # ========================================================================
    # Component 3: MMAC (Multi-Dimensional Multi-Agent Critiques)
    # ========================================================================

    def _mmac_evaluate(
        self,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        current_prompt: str,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Multi-Dimensional Multi-Agent Critiques (MMAC).

        From VISTA Section 2.3:
        For each dimension d in {Visual, Audio, Context},
        a triadic court of judges evaluates the video:
        - Normal Judge: standard assessment
        - Adversarial Judge: deliberately contrarian
        - Meta Judge: reconciles the two

        The final critique aggregates across dimensions:
        C_final = (1/|D|) * sum_{d in D} MetaJudge_d(NJ_d, AJ_d)

        Args:
            reference_video: Reference video tensor.
            generated_video: Generated video tensor.
            current_prompt: The prompt that generated the video.
            iteration: Current iteration number.

        Returns:
            Dict with per-dimension critiques and aggregated assessment.
        """
        dimensions = ["Visual", "Motion", "Context"]
        # Adapted from VISTA: Audio → Motion for video effects domain

        dimension_critiques = {}

        for dim in dimensions:
            # Stage 1: Normal Judge
            normal_critique = self._call_judge(
                judge_role="normal",
                dimension=dim,
                reference_video=reference_video,
                generated_video=generated_video,
                current_prompt=current_prompt,
                iteration=iteration,
            )

            # Stage 2: Adversarial Judge (sees normal critique, must challenge it)
            adversarial_critique = self._call_judge(
                judge_role="adversarial",
                dimension=dim,
                reference_video=reference_video,
                generated_video=generated_video,
                current_prompt=current_prompt,
                iteration=iteration,
                previous_critique=normal_critique,
            )

            # Stage 3: Meta Judge (reconciles normal and adversarial)
            meta_critique = self._call_judge(
                judge_role="meta",
                dimension=dim,
                reference_video=reference_video,
                generated_video=generated_video,
                current_prompt=current_prompt,
                iteration=iteration,
                normal_critique=normal_critique,
                adversarial_critique=adversarial_critique,
            )

            dimension_critiques[dim] = {
                "normal": normal_critique,
                "adversarial": adversarial_critique,
                "meta": meta_critique,
            }

        # Aggregate across dimensions
        aggregated = self._aggregate_critiques(dimension_critiques)

        return {
            "dimension_critiques": dimension_critiques,
            "aggregated_assessment": aggregated["assessment"],
            "aggregated_score": aggregated["score"],
            "improvement_priorities": aggregated["priorities"],
        }

    def _call_judge(
        self,
        judge_role: str,
        dimension: str,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        current_prompt: str,
        iteration: int,
        previous_critique: Optional[str] = None,
        normal_critique: Optional[str] = None,
        adversarial_critique: Optional[str] = None,
    ) -> str:
        """
        Call a single judge agent for one dimension.

        Args:
            judge_role: One of 'normal', 'adversarial', 'meta'.
            dimension: The evaluation dimension.
            reference_video: Reference video.
            generated_video: Generated video.
            current_prompt: Current generation prompt.
            iteration: Current iteration.
            previous_critique: For adversarial judge - the normal critique to challenge.
            normal_critique: For meta judge - the normal assessment.
            adversarial_critique: For meta judge - the adversarial assessment.

        Returns:
            The judge's critique as a string.
        """
        # Build dimension-specific evaluation criteria
        dimension_criteria = {
            "Visual": (
                "Evaluate visual fidelity: color accuracy, texture quality, "
                "particle/effect appearance, lighting, contrast, and overall "
                "visual similarity to the reference."
            ),
            "Motion": (
                "Evaluate motion quality: movement speed, direction, acceleration, "
                "trajectory smoothness, temporal coherence, and dynamic patterns "
                "compared to the reference."
            ),
            "Context": (
                "Evaluate contextual coherence: spatial layout, effect placement, "
                "interaction with scene elements, physical plausibility, and "
                "overall scene consistency with the reference."
            ),
        }

        # Build role-specific instructions
        if judge_role == "normal":
            role_instruction = (
                f"You are a Normal Judge evaluating the '{dimension}' dimension. "
                f"Provide a fair and balanced assessment of how well the generated "
                f"video matches the reference in terms of {dimension.lower()} quality.\n"
                f"Criteria: {dimension_criteria[dimension]}\n"
                f"Current prompt: {current_prompt}\n"
                f"Provide: (1) score 1-10, (2) detailed assessment, (3) specific issues."
            )
        elif judge_role == "adversarial":
            role_instruction = (
                f"You are an Adversarial Judge for the '{dimension}' dimension. "
                f"Your role is to challenge the following assessment and find flaws "
                f"or overlooked issues:\n\n"
                f"Previous Assessment: {previous_critique}\n\n"
                f"Deliberately look for: missed defects, overly generous scoring, "
                f"subtle issues that were overlooked. Be critical but fair.\n"
                f"Criteria: {dimension_criteria[dimension]}\n"
                f"Provide: (1) counter-score 1-10, (2) challenges to the assessment, "
                f"(3) additional issues found."
            )
        else:  # meta
            role_instruction = (
                f"You are a Meta Judge for the '{dimension}' dimension. "
                f"Reconcile the following two assessments to produce a final verdict:\n\n"
                f"Normal Judge: {normal_critique}\n\n"
                f"Adversarial Judge: {adversarial_critique}\n\n"
                f"Weigh both perspectives fairly. The adversarial judge may have "
                f"valid points or may be overly critical. Produce a balanced final "
                f"assessment.\n"
                f"Provide: (1) final score 1-10, (2) reconciled assessment, "
                f"(3) key improvement areas for this dimension."
            )

        # In actual implementation, this would call the VLM with video frames
        # For the module structure, we delegate to the VLM client
        result = self.vlm_client.analyze_and_refine(
            composite_video_path=self._get_or_create_composite(
                reference_video, generated_video, iteration
            ),
            current_prompt=role_instruction,
            history=None,
            user_description=None,
        )

        return result.get("analysis", "")

    def _get_or_create_composite(
        self,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        iteration: int,
    ) -> str:
        """Get or create a composite video for judge evaluation."""
        composite_dir = self.output_dir / "vista_composites"
        composite_dir.mkdir(parents=True, exist_ok=True)
        composite_path = composite_dir / f"composite_iter_{iteration:03d}.mp4"

        if not composite_path.exists():
            composite = create_composite_video(
                videos=[reference_video, generated_video],
                labels=["Reference", "Generated"],
            )
            save_video_tensor(composite, str(composite_path))

        return str(composite_path)

    def _aggregate_critiques(
        self, dimension_critiques: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Aggregate critiques across all dimensions.

        Extracts scores and priorities from meta judge outputs.

        Returns:
            Dict with 'assessment', 'score', and 'priorities'.
        """
        assessments = []
        priorities = []

        for dim, critiques in dimension_critiques.items():
            meta = critiques["meta"]
            assessments.append(f"[{dim}] {meta}")
            priorities.append(dim)

        # Parse scores from meta critiques (best effort)
        scores = []
        for dim, critiques in dimension_critiques.items():
            meta_text = critiques["meta"]
            # Try to extract score
            import re
            score_match = re.search(r'(?:score|Score)[:\s]*(\d+(?:\.\d+)?)', meta_text)
            if score_match:
                scores.append(float(score_match.group(1)))
            else:
                scores.append(5.0)  # default mid score

        avg_score = sum(scores) / len(scores) if scores else 5.0

        return {
            "assessment": "\n\n".join(assessments),
            "score": avg_score,
            "priorities": priorities,
        }

    # ========================================================================
    # Component 4: DTPA (Deep Thinking Prompting Agent)
    # ========================================================================

    def _dtpa_refine(
        self,
        current_prompt: str,
        mmac_critiques: Optional[Dict[str, Any]],
        history: List[Dict[str, Any]],
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        scene_plan: Optional[List[SceneSegment]] = None,
    ) -> Dict[str, Any]:
        """
        Apply DTPA 6-step deep thinking to refine the prompt.

        Uses the DeepThinkingPromptAgent helper or inlines the logic
        if the helper is not needed.

        Args:
            current_prompt: Current generation prompt.
            mmac_critiques: Output from MMAC evaluation (may be None).
            history: Previous iteration history.
            reference_video: Reference video tensor.
            generated_video: Generated video tensor.
            scene_plan: Structured scene plan (if SVPP was applied).

        Returns:
            Dict with 'refined_prompt', 'confidence', reasoning steps, etc.
        """
        # Build critique summary for DTPA
        critiques_summary = ""
        if mmac_critiques:
            critiques_summary = mmac_critiques.get("aggregated_assessment", "")
            key_issues = mmac_critiques.get("key_issues", [])
            if key_issues:
                critiques_summary += f"\nKey issues: {', '.join(key_issues)}"

        # History summary
        history_summary = ""
        if history:
            history_summary = "\n".join([
                f"Iter {i+1}: confidence={h.get('confidence', 'N/A'):.2f}, "
                f"changes={h.get('improvements', [])}"
                for i, h in enumerate(history[-3:])
            ])

        # Scene plan context
        scene_context = ""
        if scene_plan:
            scene_context = "Structured scene plan:\n" + "\n".join(
                [f"  Scene {s.scene_id}: {s.to_prompt_text()}" for s in scene_plan]
            )

        # Build the DTPA 6-step instruction
        dtpa_instruction = DTPA_PROMPT.format(
            current_prompt=current_prompt,
            critiques_summary=critiques_summary or "No multi-agent critiques available.",
            history_summary=history_summary or "First iteration.",
        )

        if scene_context:
            dtpa_instruction += f"\n\nAdditional context:\n{scene_context}"

        # Create composite video for visual context
        composite_path = self._get_or_create_composite(
            reference_video, generated_video, self._iteration_count
        )

        # Call VLM with DTPA prompt
        try:
            response = self.vlm_client.analyze_and_refine(
                composite_video_path=composite_path,
                current_prompt=dtpa_instruction,
                history=history[-2:] if history else [],
                user_description=None,
            )

            # Extract structured output
            refined_prompt = (
                response.get("step5_refined_prompt", "")
                or response.get("refined_prompt", "")
                or current_prompt
            )

            return {
                "refined_prompt": refined_prompt,
                "confidence": response.get("confidence", 0.5),
                "step_1_observe": response.get("step_1_observe", response.get("analysis", "")),
                "step_2_identify": response.get("step_2_identify", ""),
                "step_3_hypothesize": response.get("step_3_hypothesize", []),
                "step_4_evaluate": response.get("step_4_evaluate", []),
                "step_5_synthesize": response.get("step_5_synthesize", ""),
                "step_6_verify": response.get("step_6_verify", ""),
                "key_changes": response.get("key_changes", response.get("improvements", [])),
            }

        except Exception as e:
            # Fallback on error
            return {
                "refined_prompt": current_prompt,
                "confidence": 0.3,
                "step_1_observe": f"DTPA error: {str(e)}",
                "key_changes": [],
            }

    def _simple_refine(
        self,
        current_prompt: str,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Simple VLM-based refinement (fallback when DTPA is disabled).

        This is equivalent to P-Flow's original single-call optimization.

        Args:
            current_prompt: Current prompt.
            reference_video: Reference video.
            generated_video: Generated video.
            history: Previous iterations.

        Returns:
            Dict with 'refined_prompt' and 'confidence'.
        """
        composite_path = self._get_or_create_composite(
            reference_video, generated_video, self._iteration_count
        )

        try:
            result = self.vlm_client.analyze_and_refine(
                composite_video_path=composite_path,
                current_prompt=current_prompt,
                history=history,
                user_description=None,
            )
            return {
                "refined_prompt": result.get("refined_prompt", current_prompt),
                "confidence": result.get("confidence", 0.5),
                "step_1_observe": result.get("analysis", ""),
                "key_changes": result.get("improvements", []),
            }
        except Exception as e:
            return {
                "refined_prompt": current_prompt,
                "confidence": 0.3,
                "step_1_observe": f"Error: {e}",
                "key_changes": [],
            }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _save_temp_video(
        self, video: torch.Tensor, name: str
    ) -> Path:
        """
        Save a video tensor to a temporary file.

        Args:
            video: Video tensor (C, F, H, W).
            name: Name prefix for the file.

        Returns:
            Path to saved video file.
        """
        temp_dir = self.output_dir / "vista_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        path = temp_dir / f"{name}.mp4"
        save_video_tensor(video, str(path))
        return path

    def _save_iteration_results(
        self,
        iteration: int,
        prompt: str,
        result: Dict[str, Any],
    ):
        """Save VISTA iteration results to disk for analysis."""
        results_dir = self.output_dir / "vista_optimization_log"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable fields
        serializable_result = {}
        for k, v in result.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_result[k] = v

        log_entry = {
            "iteration": iteration,
            "input_prompt": prompt,
            "output": serializable_result,
            "components_used": {
                "svpp": self.enable_svpp,
                "tournament": self.enable_tournament,
                "mmac": self.enable_mmac,
                "dtpa": self.enable_dtpa,
            },
        }

        log_path = results_dir / f"vista_iter_{iteration:03d}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)


class DeepThinkingPromptAgent:
    """
    Deep Thinking Prompting Agent (DTPA) from VISTA.

    Implements the 6-step introspective reasoning process:
    1. Critique Understanding - Comprehend what critiques say
    2. Root Cause Analysis - Identify WHY issues exist in the prompt
    3. Strategy Formulation - Develop specific modification strategies
    4. Conflict Resolution - Resolve competing improvement priorities
    5. Prompt Modification - Apply changes to produce new prompt
    6. Self-Verification - Verify modifications address all critiques

    From VISTA Eq. 2-3:
    R = f_think(C_final, P_current, H)  # reasoning chain
    P_new = f_modify(R, P_current)       # new prompt
    """

    def __init__(self, vlm_client):
        self.vlm_client = vlm_client

    def refine_prompt(
        self,
        current_prompt: str,
        mmac_critiques: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply 6-step deep thinking to refine the prompt.

        Args:
            current_prompt: Current generation prompt.
            mmac_critiques: Output from MMAC evaluation.
            history: Previous iteration history.
            user_description: Original user description.

        Returns:
            Dict with 'refined_prompt', 'reasoning_chain', 'confidence'.
        """
        history = history or []

        # Build the 6-step reasoning prompt
        dtpa_instruction = self._build_dtpa_prompt(
            current_prompt=current_prompt,
            critiques=mmac_critiques,
            history=history,
            user_description=user_description,
        )

        # Call VLM with the DTPA instruction
        result = self.vlm_client.analyze_and_refine(
            composite_video_path="",  # DTPA is text-only reasoning
            current_prompt=dtpa_instruction,
            history=history,
            user_description=user_description,
        )

        # Parse the structured output
        return self._parse_dtpa_output(result, current_prompt)

    def _build_dtpa_prompt(
        self,
        current_prompt: str,
        critiques: Dict[str, Any],
        history: List[Dict[str, Any]],
        user_description: Optional[str] = None,
    ) -> str:
        """
        Build the 6-step Deep Thinking prompt.

        This instructs the VLM to perform structured introspective reasoning
        following VISTA's DTPA methodology.
        """
        aggregated = critiques.get("aggregated_assessment", "")
        score = critiques.get("aggregated_score", 5.0)
        priorities = critiques.get("improvement_priorities", [])

        # Format dimension-level critiques
        dim_details = ""
        if "dimension_critiques" in critiques:
            for dim, c in critiques["dimension_critiques"].items():
                dim_details += f"\n  [{dim}] Meta-Judge: {c.get('meta', 'N/A')}"

        # History summary
        history_summary = ""
        if history:
            history_summary = "\n".join([
                f"  Iter {i+1}: score={h.get('confidence', 'N/A')}, "
                f"prompt_snippet='{h.get('prompt', '')[:60]}...'"
                for i, h in enumerate(history[-3:])  # last 3 iterations
            ])

        prompt = f"""You are a Deep Thinking Prompting Agent (DTPA). Your task is to refine a video generation prompt through structured 6-step reasoning.

## Current State
- Current Prompt: "{current_prompt}"
- User's Goal: "{user_description or 'Match reference video effects'}"
- MMAC Score: {score}/10
- Priority Dimensions: {', '.join(priorities)}

## Multi-Agent Critiques
{aggregated}
{dim_details}

## Recent History
{history_summary if history_summary else 'No previous iterations.'}

## Your Task: 6-Step Deep Thinking

Perform each step explicitly:

### Step 1: Critique Understanding
Summarize what the critiques are telling you. What are the main issues?

### Step 2: Root Cause Analysis
Why does the current prompt fail to address these issues? What's missing or wrong?

### Step 3: Strategy Formulation
What specific changes should be made? List concrete modification strategies.

### Step 4: Conflict Resolution
Are any improvement strategies contradictory? How do you prioritize?

### Step 5: Prompt Modification
Apply your strategies to produce the new, complete prompt. It must be self-contained.

### Step 6: Self-Verification
Verify: Does the new prompt address ALL critiques? Any remaining gaps?

## Output Format (JSON)
{{
    "step1_understanding": "...",
    "step2_root_cause": "...",
    "step3_strategies": ["strategy1", "strategy2", ...],
    "step4_conflicts": "...",
    "step5_refined_prompt": "The complete new prompt",
    "step6_verification": "...",
    "confidence": 0.0-1.0
}}"""

        return prompt

    def _parse_dtpa_output(
        self, result: Dict[str, Any], fallback_prompt: str
    ) -> Dict[str, Any]:
        """Parse DTPA output into structured format."""
        # The VLM client already parses JSON for us
        analysis = result.get("analysis", "")
        refined = result.get("refined_prompt", "")

        # Try to extract structured steps from analysis
        reasoning_chain = {
            "understanding": result.get("step1_understanding", ""),
            "root_cause": result.get("step2_root_cause", ""),
            "strategies": result.get("step3_strategies", []),
            "conflicts": result.get("step4_conflicts", ""),
            "verification": result.get("step6_verification", ""),
        }

        # Use step5 if available, otherwise fall back to refined_prompt
        final_prompt = (
            result.get("step5_refined_prompt", "")
            or refined
            or fallback_prompt
        )

        return {
            "refined_prompt": final_prompt,
            "reasoning_chain": reasoning_chain,
            "confidence": result.get("confidence", 0.5),
            "analysis": analysis,
            "improvements": result.get("step3_strategies", result.get("improvements", [])),
            "key_differences": result.get("key_differences", []),
        }


class BinaryTournamentSelector:
    """
    Binary Tournament Selection with Probing Critiques.

    From VISTA Section 2.2 (Algorithm 2 - PairwiseSelect):
    Given two candidate videos, performs bidirectional comparison:
    1. Compare (A vs B) - with A shown first
    2. Compare (B vs A) - with B shown first (to eliminate position bias)
    3. If both comparisons agree → select winner
    4. If disagreement → both are equivalent, keep either

    This eliminates positional bias in VLM evaluation.
    """

    def __init__(self, vlm_client):
        self.vlm_client = vlm_client

    def select_best(
        self,
        video_a: torch.Tensor,
        video_b: torch.Tensor,
        prompt_a: str,
        prompt_b: str,
        reference_video: torch.Tensor,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Perform binary tournament between two candidate videos.

        Args:
            video_a: First candidate video.
            video_b: Second candidate video.
            prompt_a: Prompt that produced video_a.
            prompt_b: Prompt that produced video_b.
            reference_video: Reference video for comparison.
            output_dir: Directory for temporary files.

        Returns:
            Dict with 'winner' ('A' or 'B'), 'prompt', 'video',
            'confidence', 'agreement' (bool).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Forward comparison: A vs B
        forward_result = self._compare(
            first_video=video_a,
            second_video=video_b,
            reference_video=reference_video,
            first_label="Candidate A",
            second_label="Candidate B",
            output_dir=output_dir,
            tag="forward",
        )

        # Backward comparison: B vs A (position swap)
        backward_result = self._compare(
            first_video=video_b,
            second_video=video_a,
            reference_video=reference_video,
            first_label="Candidate B",
            second_label="Candidate A",
            output_dir=output_dir,
            tag="backward",
        )

        # Check agreement
        forward_winner = forward_result.get("winner", "first")  # 'first' or 'second'
        backward_winner = backward_result.get("winner", "first")

        # Map back to A/B
        # Forward: first=A, second=B
        # Backward: first=B, second=A
        forward_picks_a = (forward_winner == "first")
        backward_picks_a = (backward_winner == "second")

        agreement = (forward_picks_a == backward_picks_a)

        if agreement:
            winner = "A" if forward_picks_a else "B"
        else:
            # Disagreement - default to A (first generated), or use score
            forward_conf = forward_result.get("confidence", 0.5)
            backward_conf = backward_result.get("confidence", 0.5)
            winner = "A" if forward_conf >= backward_conf else "B"

        selected_video = video_a if winner == "A" else video_b
        selected_prompt = prompt_a if winner == "A" else prompt_b

        return {
            "winner": winner,
            "prompt": selected_prompt,
            "video": selected_video,
            "agreement": agreement,
            "confidence": max(
                forward_result.get("confidence", 0.5),
                backward_result.get("confidence", 0.5),
            ),
            "forward_result": forward_result,
            "backward_result": backward_result,
        }

    def _compare(
        self,
        first_video: torch.Tensor,
        second_video: torch.Tensor,
        reference_video: torch.Tensor,
        first_label: str,
        second_label: str,
        output_dir: Path,
        tag: str,
    ) -> Dict[str, Any]:
        """
        Compare two videos with reference context.

        Returns:
            Dict with 'winner' ('first' or 'second'), 'reason', 'confidence'.
        """
        # Create 3-panel composite: [Reference | First | Second]
        composite = create_composite_video(
            videos=[reference_video, first_video, second_video],
            labels=["Reference", first_label, second_label],
        )

        composite_path = output_dir / f"tournament_{tag}.mp4"
        save_video_tensor(composite, str(composite_path))

        comparison_prompt = (
            f"Compare two candidate videos against the reference. "
            f"The video shows three panels: Reference (left), "
            f"{first_label} (middle), {second_label} (right).\n"
            f"Which candidate better reproduces the visual effects from the reference?\n"
            f"Consider: motion patterns, visual quality, temporal coherence, "
            f"and effect accuracy.\n"
            f"Respond with JSON: {{\"winner\": \"first\" or \"second\", "
            f"\"reason\": \"...\", \"confidence\": 0.0-1.0}}"
        )

        result = self.vlm_client.analyze_and_refine(
            composite_video_path=str(composite_path),
            current_prompt=comparison_prompt,
            history=None,
            user_description=None,
        )

        # Parse winner from result
        analysis = result.get("analysis", "").lower()
        refined = result.get("refined_prompt", "").lower()

        # Determine winner
        if "first" in analysis or "first" in refined:
            winner = "first"
        elif "second" in analysis or "second" in refined:
            winner = "second"
        else:
            winner = "first"  # default

        return {
            "winner": winner,
            "reason": result.get("analysis", ""),
            "confidence": result.get("confidence", 0.5),
        }


# =============================================================================
# Mock VISTA Client for testing without API
# =============================================================================

class MockVistaVLMClient:
    """
    Mock VLM client for testing VISTA optimizer without API access.
    Simulates multi-agent behavior with progressively improving results.
    """

    def __init__(self):
        self.call_count = 0

    def analyze_and_refine(
        self,
        composite_video_path: str,
        current_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.call_count += 1

        # Simulate different judge responses
        if "Normal Judge" in current_prompt:
            return {
                "analysis": f"Score: 6. The visual effects show reasonable quality but lack the precise motion dynamics of the reference. Color saturation is slightly lower.",
                "improvements": ["Increase motion detail", "Enhance color vibrancy"],
                "refined_prompt": "",
                "confidence": 0.6,
                "key_differences": ["Motion speed", "Color intensity"],
            }
        elif "Adversarial Judge" in current_prompt:
            return {
                "analysis": f"Score: 4. The previous assessment was too generous. The temporal coherence is significantly worse than claimed, and particle density is notably lower.",
                "improvements": ["Fix temporal coherence", "Increase particle density"],
                "refined_prompt": "",
                "confidence": 0.4,
                "key_differences": ["Temporal coherence", "Particle density"],
            }
        elif "Meta Judge" in current_prompt:
            return {
                "analysis": f"Score: 5. Balancing both perspectives: motion quality needs improvement (adversarial judge correctly identified temporal issues), but overall visual quality is acceptable (normal judge's assessment of color is fair).",
                "improvements": ["Prioritize temporal coherence", "Then motion detail"],
                "refined_prompt": "",
                "confidence": 0.5,
                "key_differences": ["Temporal coherence", "Motion detail"],
            }
        elif "Deep Thinking" in current_prompt or "deep-thinking" in current_prompt.lower():
            # Extract the actual user prompt from DTPA instruction
            import re
            prompt_match = re.search(r'Current [Pp]rompt:\s*["\']?(.+?)["\']?\n', current_prompt)
            user_prompt = prompt_match.group(1) if prompt_match else "visual effect"
            base_confidence = min(0.5 + self.call_count * 0.05, 0.92)
            return {
                "analysis": "Applied 6-step reasoning to refine prompt.",
                "improvements": ["Added temporal coherence descriptors", "Enhanced motion vocabulary"],
                "refined_prompt": f"{user_prompt}, with enhanced temporal coherence and fluid particle dynamics",
                "confidence": base_confidence,
                "key_differences": [],
                "step_1_observe": "Generated video lacks smooth temporal transitions seen in reference.",
                "step_2_identify": "Prompt lacks explicit temporal flow descriptors.",
                "step_3_hypothesize": ["Add temporal adjectives", "Specify motion patterns"],
                "step_4_evaluate": ["Temporal adjectives: low risk, high impact"],
                "step_5_synthesize": "Combining temporal and motion refinements.",
                "step5_refined_prompt": f"{user_prompt}, with smooth temporal flow and naturally accelerating particle dynamics",
                "step_6_verify": "New prompt addresses temporal and motion critiques.",
                "key_changes": ["Added temporal flow descriptor", "Enhanced particle motion"],
            }
        elif "Compare two candidate" in current_prompt:
            return {
                "analysis": "first candidate shows better temporal coherence",
                "improvements": [],
                "refined_prompt": "first",
                "confidence": 0.7,
                "key_differences": [],
            }
        else:
            # Generic response
            return {
                "analysis": "Generated video needs improvement in visual effects accuracy.",
                "improvements": ["Enhance detail"],
                "refined_prompt": current_prompt + ", with improved visual fidelity",
                "confidence": 0.55,
                "key_differences": ["Visual fidelity"],
            }

    def analyze_with_frames(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze_and_refine("", args[2] if len(args) > 2 else "")