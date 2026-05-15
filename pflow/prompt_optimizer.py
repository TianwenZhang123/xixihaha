"""
Test-Time Prompt Optimization Module for P-Flow.

This module implements the iterative prompt refinement process described
in Section 3.4 of the paper. It:
1. Creates composite (side-by-side) videos for VLM comparison
2. Sends to VLM for analysis
3. Extracts refined prompts from VLM output
4. Manages the optimization loop

Reference: Section 3.4 and Algorithm 1 (Appendix A) of the paper.
"""

import os
import json
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import torch

from .vlm_client import VLMClient, MockVLMClient
from .video_utils import (
    create_composite_video,
    save_video_tensor,
    extract_key_frames,
)


class PromptOptimizer:
    """
    Test-Time Prompt Optimization.
    
    Algorithm (from paper):
        1. Generate video V_i using current prompt P_i
        2. Create composite video [V_ref | V_i]
        3. Send composite + history to VLM
        4. VLM outputs analysis A_i and refined prompt P_{i+1}
        5. Repeat until convergence or max iterations
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
            vlm_client: VLM client instance. If None and use_mock=True, uses MockVLMClient.
            max_iterations: Maximum optimization iterations (i_max in paper).
            output_dir: Directory to save intermediate results.
            use_mock: Whether to use mock VLM (for testing without API).
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
        initial_prompt: str,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        iteration: int,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
        previous_video: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform one iteration of prompt optimization.
        
        The VLM receives:
        - A composite video showing V_ref and V_gen side by side
        - The current prompt
        - History of previous iterations (text only)
        
        From the paper (Section 3.5 - Historical Trajectory):
        Only 3 videos are sent to VLM: V_ref, V_{i-1}, V_i
        But ALL text history is preserved.
        
        Args:
            initial_prompt: Current prompt P_i.
            reference_video: Reference video tensor (B, C, F, H, W) or (C, F, H, W).
            generated_video: Current generated video tensor.
            iteration: Current iteration number.
            history: List of previous {prompt, analysis, improvements} dicts.
            user_description: Original user description.
            previous_video: Previous iteration's generated video (V_{i-1}).
            
        Returns:
            Dict with 'refined_prompt', 'analysis', 'improvements',
            'confidence', 'key_differences'.
        """
        history = history or []
        
        # Create composite video for VLM
        composite_path = self._create_comparison_video(
            reference_video=reference_video,
            generated_video=generated_video,
            previous_video=previous_video,
            iteration=iteration,
        )
        
        # Call VLM for analysis and refinement
        result = self.vlm_client.analyze_and_refine(
            composite_video_path=str(composite_path),
            current_prompt=initial_prompt,
            history=history,
            user_description=user_description,
        )
        
        # Save iteration results
        self._save_iteration_results(iteration, initial_prompt, result)
        
        return result
    
    def optimize_prompt_with_frames(
        self,
        initial_prompt: str,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        iteration: int,
        history: Optional[List[Dict[str, Any]]] = None,
        user_description: Optional[str] = None,
        num_frames: int = 8,
    ) -> Dict[str, Any]:
        """
        Alternative optimization using key frames instead of full video.
        Useful when video upload is slow or unsupported.
        
        Args:
            initial_prompt: Current prompt.
            reference_video: Reference video tensor.
            generated_video: Generated video tensor.
            iteration: Current iteration number.
            history: Previous iteration history.
            user_description: User's effect description.
            num_frames: Number of key frames to extract.
            
        Returns:
            Optimization result dictionary.
        """
        history = history or []
        
        # Extract key frames
        ref_frames = extract_key_frames(reference_video, num_frames)
        gen_frames = extract_key_frames(generated_video, num_frames)
        
        # Save frames temporarily
        ref_paths = []
        gen_paths = []
        
        frame_dir = self.output_dir / f"frames_iter_{iteration:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (ref_f, gen_f) in enumerate(zip(ref_frames, gen_frames)):
            ref_path = str(frame_dir / f"ref_frame_{i:02d}.png")
            gen_path = str(frame_dir / f"gen_frame_{i:02d}.png")
            
            from .video_utils import save_frame
            save_frame(ref_f, ref_path)
            save_frame(gen_f, gen_path)
            
            ref_paths.append(ref_path)
            gen_paths.append(gen_path)
        
        # Call VLM with frames
        result = self.vlm_client.analyze_with_frames(
            reference_frames=ref_paths,
            generated_frames=gen_paths,
            current_prompt=initial_prompt,
            history=history,
            user_description=user_description,
        )
        
        self._save_iteration_results(iteration, initial_prompt, result)
        
        return result
    
    def _create_comparison_video(
        self,
        reference_video: torch.Tensor,
        generated_video: torch.Tensor,
        previous_video: Optional[torch.Tensor] = None,
        iteration: int = 0,
    ) -> Path:
        """
        Create a side-by-side composite video for VLM comparison.
        
        From the paper (Section 3.5):
        The composite video contains V_ref (left) and V_i (right).
        If previous_video is provided, creates a 3-panel: [V_ref | V_{i-1} | V_i]
        
        Args:
            reference_video: Reference video tensor.
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
            # 3-panel comparison: [ref | prev | current]
            composite = create_composite_video(
                videos=[reference_video, previous_video, generated_video],
                labels=["Reference", "Previous", "Current"],
            )
        else:
            # 2-panel comparison: [ref | current]
            composite = create_composite_video(
                videos=[reference_video, generated_video],
                labels=["Reference", "Generated"],
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
    
    def should_stop_early(
        self,
        history: List[Dict[str, Any]],
        min_iterations: int = 3,
    ) -> bool:
        """
        Check if optimization should stop early based on convergence.
        
        Criteria:
        - Confidence is very high (> 0.95)
        - No significant improvements in last 2 iterations
        - Prompt hasn't changed meaningfully
        
        Args:
            history: Full optimization history.
            min_iterations: Minimum iterations before early stopping.
            
        Returns:
            True if optimization should stop.
        """
        if len(history) < min_iterations:
            return False
            
        # Check confidence trend
        recent_confidence = [h.get("confidence", 0) for h in history[-3:]]
        if all(c > 0.95 for c in recent_confidence):
            return True
            
        # Check if improvements are diminishing
        recent_improvements = [h.get("improvements", []) for h in history[-2:]]
        if all(len(imp) == 0 for imp in recent_improvements):
            return True
            
        return False
