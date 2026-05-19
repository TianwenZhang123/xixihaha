"""
Historical Trajectory Maintenance Module for P-Flow (Paper-Faithful).

From Section 3.5 of the paper:
- Stores complete trajectory: {V_i, P_i, A_i} for each iteration
- For VLM input: only sends 3 videos (V_ref, V_{i-1}, V_i) to limit token usage
- Preserves ALL text history (prompts and analyses) for context
- NO confidence score

The trajectory provides the VLM with:
1. Visual comparison (limited to key videos via vertical composite)
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
    """Single entry in the optimization trajectory (paper format)."""
    iteration: int
    prompt: str
    video_path: Optional[str] = None
    analysis: Dict[str, str] = field(default_factory=dict)  # Paper format: 4 sub-fields
    refined_prompt: str = ""  # The VLM's suggested next prompt

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrajectoryManager:
    """
    Manages the historical trajectory of the optimization process.

    Paper design (Section 3.5):
    1. Video storage: Only keep V_{i-1} in memory for composite creation
    2. Text history: Keep ALL prompt/analysis text for VLM context
    3. VLM input: Provide at most 3 videos (ref, previous, current) via vertical composite
    4. NO convergence tracking (fixed iterations)
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        save_all_videos: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save trajectory data.
            save_all_videos: Whether to save all generated videos to disk.
        """
        self.output_dir = Path(output_dir)
        self.trajectory_dir = self.output_dir / "trajectory"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        self.save_all_videos = save_all_videos

        # Trajectory storage
        self.entries: List[TrajectoryEntry] = []

        # Only keep one previous video in memory (for composite creation)
        self._previous_video: Optional[torch.Tensor] = None

        # Reference video (always kept)
        self.reference_video: Optional[torch.Tensor] = None
        self.reference_video_path: Optional[str] = None

    def set_reference(self, video: torch.Tensor, save_path: Optional[str] = None):
        """Set the reference video."""
        self.reference_video = video
        if save_path:
            self.reference_video_path = save_path

    def add_entry(
        self,
        iteration: int,
        prompt: str,
        video: Optional[torch.Tensor] = None,
        video_path: Optional[str] = None,
        analysis: Optional[Dict[str, str]] = None,
        refined_prompt: str = "",
    ) -> TrajectoryEntry:
        """
        Add a new entry to the trajectory.

        Args:
            iteration: Iteration number.
            prompt: The prompt used for this iteration (P_i).
            video: Generated video tensor (for keeping previous in memory).
            video_path: Path where video was saved.
            analysis: VLM analysis dict (paper format with 4 sub-fields).
            refined_prompt: VLM's suggested refined prompt (P_{i+1}).

        Returns:
            The created TrajectoryEntry.
        """
        entry = TrajectoryEntry(
            iteration=iteration,
            prompt=prompt,
            video_path=video_path,
            analysis=analysis or {},
            refined_prompt=refined_prompt,
        )

        self.entries.append(entry)

        # Update previous video reference (only keep latest for memory)
        if video is not None:
            self._previous_video = video

        # Save trajectory to disk after each entry
        self._save_trajectory()

        return entry

    def get_previous_video(self) -> Optional[torch.Tensor]:
        """
        Get the previous iteration's video for composite creation.

        From paper: V_{i-1} is included in the composite sent to VLM.
        Returns None for first iteration.
        """
        return self._previous_video

    def get_text_history(self) -> List[Dict[str, Any]]:
        """
        Get the full text history formatted for VLM input.

        From paper: ALL text is preserved for VLM context.
        This returns prompt + analysis for each past iteration.

        Returns:
            List of dicts with iteration, prompt, analysis for each entry.
        """
        history = []
        for entry in self.entries:
            history.append({
                "iteration": entry.iteration,
                "prompt": entry.prompt,
                "analysis": entry.analysis,
            })
        return history

    def get_all_prompts(self) -> List[str]:
        """Get all prompts in order."""
        return [e.prompt for e in self.entries]

    def get_full_trajectory(self) -> Dict[str, Any]:
        """
        Get the complete trajectory for saving to disk.

        Returns:
            Full trajectory data as serializable dict.
        """
        return {
            "num_iterations": len(self.entries),
            "reference_video_path": self.reference_video_path,
            "entries": [e.to_dict() for e in self.entries],
        }

    def _save_trajectory(self):
        """Save the full trajectory to disk as JSON."""
        trajectory_path = self.trajectory_dir / "trajectory.json"
        data = self.get_full_trajectory()
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_trajectory(self, path: Optional[str] = None):
        """Load a previously saved trajectory from disk."""
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
        self._previous_video = None

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"TrajectoryManager(iterations={len(self.entries)})"
