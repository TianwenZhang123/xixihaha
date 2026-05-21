"""
Historical Trajectory Maintenance for P-Flow.

Section 3.5:
- Stores complete trajectory: {V_i, P_i, A_i} for each iteration
- VLM receives only 3 videos (V_ref, V_{i-1}, V_i) via vertical composite
- ALL text history preserved for context
- NO confidence score
"""

import json
import os
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch


@dataclass
class TrajectoryEntry:
    """Single entry in optimization trajectory."""
    iteration: int
    prompt: str
    video_path: Optional[str] = None
    analysis: Dict[str, str] = field(default_factory=dict)
    refined_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrajectoryManager:
    """
    Manages optimization trajectory (Section 3.5).

    Design:
    1. Only keep V_{i-1} in memory for composite creation
    2. Keep ALL text history for VLM context
    3. At most 3 videos in composite (ref, prev, current)
    4. NO convergence tracking (fixed iterations)
    """

    def __init__(self, output_dir: str = "outputs", save_all_videos: bool = True):
        self.output_dir = Path(output_dir)
        self.trajectory_dir = self.output_dir / "trajectory"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.save_all_videos = save_all_videos

        self.entries: List[TrajectoryEntry] = []
        self._previous_video: Optional[torch.Tensor] = None
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
        """Add new entry to trajectory."""
        entry = TrajectoryEntry(
            iteration=iteration,
            prompt=prompt,
            video_path=video_path,
            analysis=analysis or {},
            refined_prompt=refined_prompt,
        )
        self.entries.append(entry)

        if video is not None:
            self._previous_video = video

        self._save_trajectory()
        return entry

    def get_previous_video(self) -> Optional[torch.Tensor]:
        """Get V_{i-1} for composite creation."""
        return self._previous_video

    def get_text_history(self) -> List[Dict[str, Any]]:
        """Get full text history for VLM context."""
        return [
            {"iteration": e.iteration, "prompt": e.prompt, "analysis": e.analysis}
            for e in self.entries
        ]

    def get_all_prompts(self) -> List[str]:
        """Get all prompts in order."""
        return [e.prompt for e in self.entries]

    def get_full_trajectory(self) -> Dict[str, Any]:
        """Get complete trajectory for serialization."""
        return {
            "num_iterations": len(self.entries),
            "reference_video_path": self.reference_video_path,
            "entries": [e.to_dict() for e in self.entries],
        }

    def _save_trajectory(self):
        """Persist trajectory to disk."""
        path = self.trajectory_dir / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_full_trajectory(), f, indent=2, ensure_ascii=False)

    def load_trajectory(self, path: Optional[str] = None):
        """Load saved trajectory."""
        if path is None:
            path = str(self.trajectory_dir / "trajectory.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = [TrajectoryEntry.from_dict(e) for e in data.get("entries", [])]
        self.reference_video_path = data.get("reference_video_path")

    def reset(self):
        """Reset for new run."""
        self.entries = []
        self._previous_video = None

    def __len__(self):
        return len(self.entries)
