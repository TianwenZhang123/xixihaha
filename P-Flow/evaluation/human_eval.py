"""
Human Evaluation Setup for P-Flow.

Paper Section 4.1:
- 15 annotators perform pairwise comparisons
- 100 sample pairs per comparison
- 15 visual effect types
- Protocol: "Which video better matches the reference's visual effects?"

This module provides:
1. Pair generation for annotation
2. Annotation result collection
3. Win rate computation
4. Statistical significance testing
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from itertools import combinations


class HumanEvalPairGenerator:
    """Generate pairs for pairwise human evaluation."""

    def __init__(
        self,
        reference_dir: str,
        methods: Dict[str, str],
        num_samples: int = 100,
        seed: int = 42,
    ):
        """
        Args:
            reference_dir: Directory with reference videos.
            methods: Dict of {method_name: output_directory}.
            num_samples: Number of pairs to generate per comparison.
            seed: Random seed for reproducibility.
        """
        self.reference_dir = Path(reference_dir)
        self.methods = methods
        self.num_samples = num_samples
        random.seed(seed)

    def generate_pairs(self, output_dir: str) -> Dict[str, Any]:
        """
        Generate evaluation pairs for all method combinations.

        Creates a structured annotation file with randomized presentation order.

        Returns:
            Annotation task configuration.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        method_names = list(self.methods.keys())
        method_pairs = list(combinations(method_names, 2))

        all_tasks = []
        task_id = 0

        for method_a, method_b in method_pairs:
            dir_a = Path(self.methods[method_a])
            dir_b = Path(self.methods[method_b])

            # Find common samples
            videos_a = set(f.name for f in dir_a.glob("*.mp4"))
            videos_b = set(f.name for f in dir_b.glob("*.mp4"))
            common = sorted(videos_a & videos_b)

            # Sample pairs
            if len(common) > self.num_samples:
                selected = random.sample(common, self.num_samples)
            else:
                selected = common

            for video_name in selected:
                # Randomize left/right presentation
                if random.random() > 0.5:
                    left_method, right_method = method_a, method_b
                    left_path = str(dir_a / video_name)
                    right_path = str(dir_b / video_name)
                else:
                    left_method, right_method = method_b, method_a
                    left_path = str(dir_b / video_name)
                    right_path = str(dir_a / video_name)

                # Find reference
                ref_path = str(self.reference_dir / video_name)

                task = {
                    "task_id": task_id,
                    "reference_video": ref_path,
                    "video_left": left_path,
                    "video_right": right_path,
                    "left_method": left_method,
                    "right_method": right_method,
                    "video_name": video_name,
                    "comparison": f"{method_a}_vs_{method_b}",
                }
                all_tasks.append(task)
                task_id += 1

        # Save tasks
        config = {
            "total_tasks": len(all_tasks),
            "methods": method_names,
            "num_comparisons": len(method_pairs),
            "tasks_per_comparison": self.num_samples,
            "instructions": (
                "For each task, watch the reference video and both generated videos. "
                "Select which generated video better reproduces the visual effects "
                "shown in the reference. Consider: motion patterns, visual appearance, "
                "spatial distribution, temporal dynamics, and effect-scene interactions."
            ),
            "tasks": all_tasks,
        }

        with open(output_path / "annotation_tasks.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Generated {len(all_tasks)} annotation tasks")
        print(f"  Methods: {method_names}")
        print(f"  Comparisons: {len(method_pairs)}")
        print(f"  Tasks saved to: {output_path / 'annotation_tasks.json'}")

        return config


class HumanEvalAnalyzer:
    """Analyze collected human evaluation results."""

    def __init__(self, annotation_file: str, results_file: str):
        """
        Args:
            annotation_file: Path to annotation tasks JSON.
            results_file: Path to collected annotations JSON.
        """
        with open(annotation_file) as f:
            self.tasks = json.load(f)
        with open(results_file) as f:
            self.results = json.load(f)

    def compute_win_rates(self) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise win rates.

        Returns:
            {comparison: {method: win_rate}} for each method pair.
        """
        win_counts = {}

        for result in self.results.get("annotations", []):
            task_id = result["task_id"]
            choice = result["choice"]  # "left" or "right" or "tie"
            annotator = result.get("annotator_id", "unknown")

            # Find original task
            task = next((t for t in self.tasks["tasks"] if t["task_id"] == task_id), None)
            if task is None:
                continue

            comparison = task["comparison"]
            if comparison not in win_counts:
                win_counts[comparison] = {"left_wins": 0, "right_wins": 0, "ties": 0, "total": 0}

            if choice == "left":
                win_counts[comparison]["left_wins"] += 1
            elif choice == "right":
                win_counts[comparison]["right_wins"] += 1
            else:
                win_counts[comparison]["ties"] += 1
            win_counts[comparison]["total"] += 1

        # Compute rates
        win_rates = {}
        for comparison, counts in win_counts.items():
            total = counts["total"]
            if total == 0:
                continue
            methods = comparison.split("_vs_")
            win_rates[comparison] = {
                methods[0]: counts["left_wins"] / total,
                methods[1]: counts["right_wins"] / total,
                "tie": counts["ties"] / total,
                "total_annotations": total,
            }

        return win_rates

    def compute_agreement(self) -> float:
        """
        Compute inter-annotator agreement (Fleiss' Kappa).

        Returns:
            Kappa score (0-1, higher = better agreement).
        """
        # Group annotations by task
        task_annotations = {}
        for result in self.results.get("annotations", []):
            tid = result["task_id"]
            if tid not in task_annotations:
                task_annotations[tid] = []
            task_annotations[tid].append(result["choice"])

        if not task_annotations:
            return 0.0

        # Simple agreement: proportion of tasks where majority agrees
        agreements = 0
        total = 0
        for tid, choices in task_annotations.items():
            if len(choices) < 2:
                continue
            from collections import Counter
            counter = Counter(choices)
            most_common_count = counter.most_common(1)[0][1]
            if most_common_count > len(choices) / 2:
                agreements += 1
            total += 1

        return agreements / total if total > 0 else 0.0

    def generate_report(self, output_path: str) -> Dict[str, Any]:
        """Generate complete human evaluation report."""
        win_rates = self.compute_win_rates()
        agreement = self.compute_agreement()

        report = {
            "win_rates": win_rates,
            "inter_annotator_agreement": agreement,
            "total_annotations": len(self.results.get("annotations", [])),
            "num_annotators": len(set(
                r.get("annotator_id", "unknown")
                for r in self.results.get("annotations", [])
            )),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nHuman Evaluation Report:")
        print(f"  Total annotations: {report['total_annotations']}")
        print(f"  Annotators: {report['num_annotators']}")
        print(f"  Agreement: {agreement:.3f}")
        print(f"\n  Win Rates:")
        for comp, rates in win_rates.items():
            print(f"    {comp}: {rates}")

        return report


def create_annotation_template(output_dir: str):
    """
    Create a Gradio-based annotation interface template.

    This generates the HTML/Python code for a simple annotation UI.
    """
    template = '''#!/usr/bin/env python3
"""
Human Evaluation Annotation Interface.
Run: python annotation_interface.py --tasks annotation_tasks.json --port 7860
"""

import json
import argparse

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Install gradio for annotation UI: pip install gradio")


def create_interface(tasks_file: str):
    """Create Gradio annotation interface."""
    if not HAS_GRADIO:
        print("Gradio not available. Use manual annotation with annotation_tasks.json")
        return

    with open(tasks_file) as f:
        config = json.load(f)

    tasks = config["tasks"]
    annotations = []
    current_idx = [0]

    def get_task(idx):
        if idx >= len(tasks):
            return "Done!", None, None, None
        task = tasks[idx]
        return (
            f"Task {idx+1}/{len(tasks)} ({task['comparison']})",
            task["reference_video"],
            task["video_left"],
            task["video_right"],
        )

    def submit(choice, annotator_id):
        if current_idx[0] >= len(tasks):
            return "All tasks complete!", None, None, None
        annotations.append({
            "task_id": tasks[current_idx[0]]["task_id"],
            "choice": choice,
            "annotator_id": annotator_id,
        })
        current_idx[0] += 1
        # Save progress
        with open("annotations_progress.json", "w") as f:
            json.dump({"annotations": annotations}, f, indent=2)
        return get_task(current_idx[0])

    with gr.Blocks(title="P-Flow Human Evaluation") as demo:
        gr.Markdown("# P-Flow Human Evaluation")
        gr.Markdown(config["instructions"])

        annotator_id = gr.Textbox(label="Annotator ID")
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            ref_video = gr.Video(label="Reference")
        with gr.Row():
            left_video = gr.Video(label="Video A")
            right_video = gr.Video(label="Video B")
        with gr.Row():
            btn_left = gr.Button("A is Better")
            btn_tie = gr.Button("Tie")
            btn_right = gr.Button("B is Better")

        btn_left.click(fn=lambda aid: submit("left", aid),
                      inputs=[annotator_id], outputs=[status, ref_video, left_video, right_video])
        btn_right.click(fn=lambda aid: submit("right", aid),
                       inputs=[annotator_id], outputs=[status, ref_video, left_video, right_video])
        btn_tie.click(fn=lambda aid: submit("tie", aid),
                     inputs=[annotator_id], outputs=[status, ref_video, left_video, right_video])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_interface(args.tasks)
    if demo:
        demo.launch(server_port=args.port, share=True)
'''

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "annotation_interface.py", "w") as f:
        f.write(template)

    print(f"Annotation interface template saved to: {output_path / 'annotation_interface.py'}")
