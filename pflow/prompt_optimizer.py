"""
Test-Time Prompt Optimization Module for P-Flow (Paper-Faithful).

Implements the iterative prompt refinement process from Section 3.4:
1. Creates VERTICAL composite videos (top/middle/bottom) for VLM comparison
2. Sends to VLM with structured instruction (Listing 1)
3. Extracts refined prompts from VLM output
4. NO confidence score, NO early stopping (paper limitation)

The paper uses fixed i_max=10 iterations without adaptive stopping.

Reference: Section 3.4, 3.5 and Algorithm 1 (Appendix A).
"""

import os
import json
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import torch

from .vlm_client import VLMClient, MockVLMClient
from .video_utils import (
    create_vertical_composite,
    save_video_tensor,
    extract_key_frames,
)


class PromptOptimizer:
    """
    Test-Time Prompt Optimization (Paper-Faithful).

    Algorithm (from paper):
        1. Generate video V_i using current prompt P_i
        2. Create VERTICAL composite video [V_ref (top) | V_{i-1} (middle) | V_i (bottom)]
        3. Send composite + history to VLM
        4. VLM outputs analysis A_i and refined prompt P_{i+1}
        5. Repeat for fixed i_max iterations (NO early stopping)

    Key differences from previous code:
        - Vertical (not horizontal) composite layout
        - NO confidence score
        - NO should_stop_early() — fixed iterations
        - Paper output format: {analysis: {4 sub-fields}, refined_prompt}
    """

    def __init__(
        self,
        vlm_client: Optional[VLMClient] = None,
        max_iterations: int = 10,
        output_dir: str = "outputs",
        use_mock: bool = False,
    ):
        """
        Args:
            vlm_client: VLM client instance.
            max_iterations: Fixed iteration count (i_max in paper, default 10).
            output_dir: Directory to save intermediate results.
            use_mock: Whether to use mock VLM (for testing).
        """
        if vlm_client is not None:
            self.vlm_client = vlm_client
        elif use_mock:
            self.vlm_client = MockVLMClient()
        else:
            self.vlm_client = VLMClient()

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

        The VLM receives:
        - A VERTICAL composite video: [V_ref (top) | V_{i-1} (middle) | V_i (bottom)]
        - Structured metadata (Listing 1 format)
        - Text history of previous iterations

        Args:
            current_prompt: Current prompt P_i.
            reference_video: Reference video tensor (C, F, H, W).
            generated_video: Current generated video tensor (C, F, H, W).
            iteration: Current iteration number (1-indexed).
            desired_visual_effect: User's description of target effect.
            subject: Main subject of the video.
            environment: Background/setting.
            last_text_prompt: Previous iteration's prompt P_{i-1}.
            previous_video: Previous iteration's generated video V_{i-1}.
            history: List of previous iteration results.

        Returns:
            Dict with paper-format output:
                - 'analysis': {reference_description, last_generated_description,
                              new_generated_description, comparison}
                - 'refined_prompt': improved T2V prompt
        """
        history = history or []

        # Create vertical composite video for VLM
        composite_path = self._create_vertical_composite(
            reference_video=reference_video,
            generated_video=generated_video,
            previous_video=previous_video,
            iteration=iteration,
        )

        # Call VLM for analysis and refinement
        # Wrap in try-except to handle DashScope content moderation errors gracefully
        try:
            result = self.vlm_client.analyze_and_refine(
                composite_video_path=str(composite_path),
                current_prompt=current_prompt,
                iteration=iteration,
                desired_visual_effect=desired_visual_effect or current_prompt,
                subject=subject,
                environment=environment,
                last_text_prompt=last_text_prompt,
                history=history,
            )
        except Exception as e:
            error_msg = str(e)
            if "data_inspection_failed" in error_msg or "inappropriate" in error_msg.lower():
                print(f"  WARNING: VLM content moderation triggered (iter {iteration}). "
                      f"Keeping current prompt and continuing.")
                result = {
                    "analysis": {
                        "reference_description": "[SKIPPED: content moderation]",
                        "last_generated_description": "[SKIPPED]",
                        "new_generated_description": "[SKIPPED]",
                        "comparison": f"[SKIPPED: DashScope content filter triggered at iteration {iteration}]",
                    },
                    "refined_prompt": current_prompt,  # Keep current prompt unchanged
                }
            else:
                # Re-raise non-moderation errors
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
        Create a VERTICAL composite video for VLM comparison.

        Paper Section 3.5 layout:
            Panel A (top): Reference video V_ref
            Panel B (middle): Previous generated V_{i-1}
            Panel C (bottom): Current generated V_i

        If previous_video is None (first iteration), creates 2-panel:
            Panel A (top): Reference
            Panel C (bottom): Current

        Args:
            reference_video: Reference video tensor (C, F, H, W).
            generated_video: Current generated video tensor.
            previous_video: Previous iteration's video (optional).
            iteration: Iteration number for file naming.

        Returns:
            Path to the saved composite video.
        """
        composite_dir = self.output_dir / "composites"
        composite_dir.mkdir(parents=True, exist_ok=True)
        composite_path = composite_dir / f"composite_iter_{iteration:03d}.mp4"

        if previous_video is not None:
            # 3-panel vertical: [ref (top) | prev (middle) | current (bottom)]
            composite = create_vertical_composite(
                videos=[reference_video, previous_video, generated_video],
                labels=["A: Reference", "B: Previous", "C: Current"],
            )
        else:
            # 2-panel vertical: [ref (top) | current (bottom)]
            composite = create_vertical_composite(
                videos=[reference_video, generated_video],
                labels=["A: Reference", "C: Current"],
            )

        save_video_tensor(composite, str(composite_path))

        return composite_path

    def _save_iteration_results(
        self,
        iteration: int,
        prompt: str,
        result: Dict[str, Any],
    ):
        """Save iteration results to disk for analysis."""
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
