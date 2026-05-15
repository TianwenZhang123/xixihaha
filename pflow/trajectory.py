"""
Historical Trajectory Maintenance Module for P-Flow.

This module manages the optimization history across iterations.
From Section 3.5 of the paper:

- Stores complete trajectory: {V_i, P_i, A_i} for each iteration
- For VLM input: only sends 3 videos (V_ref, V_{i-1}, V_i) to limit token usage
- But preserves ALL text history (prompts and analyses) for context

The trajectory provides the VLM with:
1. Visual comparison (limited to key videos)
2. Full textual context of optimization progress
"""

import json
import os
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch


@dataclass
class TrajectoryEntry:
    """Single entry in the optimization trajectory."""
    iteration: int
    prompt: str
    analysis: str = ""
    improvements: List[str] = field(default_factory=list)
    confidence: float = 0.0
    key_differences: List[str] = field(default_factory=list)
    video_path: Optional[str] = None  # Path to saved generated video
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrajectoryManager:
    """
    Manages the historical trajectory of the optimization process.
    
    Key design decisions (from paper):
    1. Video storage: Only keep recent videos in memory, save all to disk
    2. Text history: Keep ALL prompt/analysis text for VLM context
    3. VLM input: Provide at most 3 videos (ref, previous, current)
    4. Convergence tracking: Monitor confidence and improvement trends
    """
    
    def __init__(
        self,
        output_dir: str = "outputs",
        max_videos_in_memory: int = 3,
        save_all_videos: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save trajectory data.
            max_videos_in_memory: Max videos to keep in GPU memory.
            save_all_videos: Whether to save all generated videos to disk.
        """
        self.output_dir = Path(output_dir)
        self.trajectory_dir = self.output_dir / "trajectory"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_videos_in_memory = max_videos_in_memory
        self.save_all_videos = save_all_videos
        
        # Trajectory storage
        self.entries: List[TrajectoryEntry] = []
        self.videos_in_memory: Dict[int, torch.Tensor] = {}  # iteration -> video tensor
        
        # Reference video (always kept)
        self.reference_video: Optional[torch.Tensor] = None
        self.reference_video_path: Optional[str] = None
        
    def set_reference(self, video: torch.Tensor, save_path: Optional[str] = None):
        """
        Set the reference video.
        
        Args:
            video: Reference video tensor (C, F, H, W) or (B, C, F, H, W).
            save_path: Optional path to save the reference video.
        """
        self.reference_video = video
        if save_path:
            self.reference_video_path = save_path
            
    def add_entry(
        self,
        iteration: int,
        prompt: str,
        video: Optional[torch.Tensor] = None,
        analysis: str = "",
        improvements: Optional[List[str]] = None,
        confidence: float = 0.0,
        key_differences: Optional[List[str]] = None,
    ) -> TrajectoryEntry:
        """
        Add a new entry to the trajectory.
        
        Args:
            iteration: Iteration number.
            prompt: The prompt used for this iteration.
            video: Generated video tensor (optional, for memory management).
            analysis: VLM analysis text.
            improvements: List of improvements suggested.
            confidence: VLM confidence score.
            key_differences: Key differences identified.
            
        Returns:
            The created TrajectoryEntry.
        """
        # Save video to disk if requested
        video_path = None
        if video is not None and self.save_all_videos:
            video_path = str(self.trajectory_dir / f"video_iter_{iteration:03d}.mp4")
            from .video_utils import save_video_tensor
            save_video_tensor(video, video_path)
        
        entry = TrajectoryEntry(
            iteration=iteration,
            prompt=prompt,
            analysis=analysis,
            improvements=improvements or [],
            confidence=confidence,
            key_differences=key_differences or [],
            video_path=video_path,
        )
        
        self.entries.append(entry)
        
        # Manage video memory
        if video is not None:
            self.videos_in_memory[iteration] = video
            self._manage_video_memory()
        
        # Save trajectory to disk
        self._save_trajectory()
        
        return entry
    
    def get_history_for_vlm(self) -> List[Dict[str, Any]]:
        """
        Get the text history formatted for VLM input.
        
        From paper: ALL text is preserved, but only 3 videos are sent.
        This returns the text portion.
        
        Returns:
            List of dictionaries with prompt, analysis, and improvements
            for each past iteration.
        """
        history = []
        for entry in self.entries:
            history.append({
                "iteration": entry.iteration,
                "prompt": entry.prompt,
                "analysis": entry.analysis,
                "improvements": entry.improvements,
                "confidence": entry.confidence,
                "key_differences": entry.key_differences,
            })
        return history
    
    def get_videos_for_vlm(self) -> Tuple[
        Optional[torch.Tensor],  # reference
        Optional[torch.Tensor],  # previous (V_{i-1})
        Optional[torch.Tensor],  # current (V_i)
    ]:
        """
        Get the 3 videos to send to VLM.
        
        From paper (Section 3.5):
        Only V_ref, V_{i-1}, V_i are sent to limit token usage.
        
        Returns:
            Tuple of (reference_video, previous_video, current_video).
        """
        ref = self.reference_video
        
        if len(self.entries) == 0:
            return ref, None, None
        elif len(self.entries) == 1:
            current_iter = self.entries[-1].iteration
            current = self.videos_in_memory.get(current_iter)
            return ref, None, current
        else:
            prev_iter = self.entries[-2].iteration
            current_iter = self.entries[-1].iteration
            prev = self.videos_in_memory.get(prev_iter)
            current = self.videos_in_memory.get(current_iter)
            return ref, prev, current
    
    def get_latest_prompt(self) -> Optional[str]:
        """Get the most recent prompt in the trajectory."""
        if self.entries:
            return self.entries[-1].prompt
        return None
    
    def get_latest_analysis(self) -> Optional[str]:
        """Get the most recent VLM analysis."""
        if self.entries:
            return self.entries[-1].analysis
        return None
    
    def get_best_entry(self) -> Optional[TrajectoryEntry]:
        """Get the entry with highest confidence score."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.confidence)
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get convergence statistics for the optimization.
        
        Returns:
            Dictionary with convergence metrics.
        """
        if not self.entries:
            return {"converged": False, "iterations": 0}
        
        confidences = [e.confidence for e in self.entries]
        
        return {
            "iterations": len(self.entries),
            "confidences": confidences,
            "best_confidence": max(confidences),
            "latest_confidence": confidences[-1],
            "confidence_trend": (
                confidences[-1] - confidences[-2] if len(confidences) > 1 else 0
            ),
            "converged": (
                len(confidences) >= 3 and 
                all(c > 0.9 for c in confidences[-3:])
            ),
        }
    
    def _manage_video_memory(self):
        """
        Manage GPU memory by evicting old videos.
        Keep only the most recent videos in memory.
        """
        if len(self.videos_in_memory) > self.max_videos_in_memory:
            # Keep the most recent entries
            iterations = sorted(self.videos_in_memory.keys())
            to_remove = iterations[:-self.max_videos_in_memory]
            for it in to_remove:
                del self.videos_in_memory[it]
                # Force garbage collection for GPU memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _save_trajectory(self):
        """Save the full trajectory to disk as JSON."""
        trajectory_path = self.trajectory_dir / "trajectory.json"
        data = {
            "num_iterations": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
            "reference_video_path": self.reference_video_path,
        }
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_trajectory(self, path: Optional[str] = None):
        """
        Load a previously saved trajectory from disk.
        
        Args:
            path: Path to trajectory JSON. If None, uses default location.
        """
        if path is None:
            path = str(self.trajectory_dir / "trajectory.json")
            
        if not os.path.exists(path):
            return
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.entries = [
            TrajectoryEntry.from_dict(e) for e in data.get("entries", [])
        ]
        self.reference_video_path = data.get("reference_video_path")
    
    def reset(self):
        """Reset the trajectory for a new optimization run."""
        self.entries = []
        self.videos_in_memory = {}
        # Keep reference video
        
    def __len__(self) -> int:
        return len(self.entries)
    
    def __repr__(self) -> str:
        return (
            f"TrajectoryManager(iterations={len(self.entries)}, "
            f"videos_in_memory={len(self.videos_in_memory)})"
        )
