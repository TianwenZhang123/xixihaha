"""
Test-Time Prompt Optimization Module for P-Flow.

Section 3.4: Iterative prompt refinement via VLM feedback.
1. Create VERTICAL composite videos for VLM comparison
2. Send to VLM with structured instruction (Listing 1)
3. Extract refined prompts from VLM output
4. NO confidence score, NO early stopping (fixed i_max=10)

Reference: Section 3.4, 3.5 and Algorithm 1.
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch

from .video_utils import create_vertical_composite, save_video_tensor

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Test-Time Prompt Optimization (Paper-Faithful).

    Algorithm:
        1. Generate video V_i using prompt P_i
        2. Create vertical composite [V_ref | V_{i-1} | V_i]
        3. Send composite + history to VLM
        4. VLM outputs analysis A_i and refined prompt P_{i+1}
        5. Repeat for fixed i_max iterations (NO early stopping)
    """

    def __init__(
        self,
        vlm_client,
        max_iterations: int = 10,
        output_dir: str = "outputs",
    ):
        """
        Args:
            vlm_client: VLM client instance (Gemini/DashScope/Mock).
            max_iterations: Fixed iteration count (i_max).
            output_dir: Directory for intermediate results.
        """
        self.vlm_client = vlm_client
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def optimize_prompt(
        self,
        current_prompt: str,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        iteration: int,
        desired_visual_effect: str = "",
        subject: str = "",
        environment: str = "",
        last_text_prompt: str = "",
        previous_video: Optional[torch.Tensor] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Perform one iteration of prompt optimization.

        Creates vertical composite → sends to VLM → returns refined prompt.

        Args:
            current_prompt: Current prompt P_i.
            reference_video: Reference video (C, F, H, W).
            generated_video: Current generated video (C, F, H, W).
            iteration: Current iteration (1-indexed).
            desired_visual_effect: Target effect description.
            subject: Main subject.
            environment: Scene/background.
            last_text_prompt: Previous prompt P_{i-1}.
            previous_video: Previous generated video V_{i-1}.
            history: Previous iteration results.

        Returns:
            {analysis: {...}, refined_prompt: "..."}
        """
        history = history or []

        # Create vertical composite for VLM
        composite_path = self._create_vertical_composite(
            reference_video=reference_video,
            generated_video=generated_video,
            previous_video=previous_video,
            iteration=iteration,
        )

        # Call VLM with error handling
        try:
            result = self.vlm_client.analyze_and_refine(
                composite_video_path=str(composite_path),
                current_prompt=current_prompt,
                iteration=iteration,
                i_max=self.max_iterations,
                desired_visual_effect=desired_visual_effect or current_prompt,
                subject=subject,
                environment=environment,
                last_text_prompt=last_text_prompt,
                history=history,
            )
        except Exception as e:
            error_msg = str(e)
            if "data_inspection_failed" in error_msg or "inappropriate" in error_msg.lower():
                logger.warning(f"VLM content moderation triggered (iter {iteration})")
                result = {
                    "analysis": {
                        "reference_description": "[SKIPPED: content moderation]",
                        "last_generated_description": "[SKIPPED]",
                        "new_generated_description": "[SKIPPED]",
                        "comparison": f"[Content filter at iteration {iteration}]",
                    },
                    "refined_prompt": current_prompt,
                }
            else:
                raise

        # Save iteration results
        self._save_iteration_results(iteration, current_prompt, result)
        return result

    def _create_vertical_composite(
        self,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        previous_video: Optional[torch.Tensor] = None,
        iteration: int = 0,
    ) -> Path:
        """
        Create vertical composite video (Section 3.5).

        3-panel: [V_ref (top) | V_{i-1} (middle) | V_i (bottom)]
        2-panel (iter 1): [V_ref (top) | V_i (bottom)]
        """
        composite_dir = self.output_dir / "composites"
        composite_dir.mkdir(parents=True, exist_ok=True)
        composite_path = composite_dir / f"composite_iter_{iteration:03d}.mp4"

        if previous_video is not None:
            composite = create_vertical_composite(
                videos=[reference_video, previous_video, generated_video],
                labels=["A: Reference", "B: Previous", "C: Current"],
            )
        else:
            composite = create_vertical_composite(
                videos=[reference_video, generated_video],
                labels=["A: Reference", "C: Current"],
            )

        save_video_tensor(composite, str(composite_path))
        return composite_path

    def _save_iteration_results(self, iteration: int, prompt: str, result: Dict[str, Any]):
        """Save iteration results to disk."""
        results_dir = self.output_dir / "optimization_log"
        results_dir.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "iteration": iteration,
            "input_prompt": prompt,
            "output": result,
        }

        log_path = results_dir / f"iter_{iteration:03d}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
