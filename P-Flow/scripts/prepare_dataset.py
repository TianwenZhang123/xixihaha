#!/usr/bin/env python3
"""
Open-VFX Dataset Preparation Script.

Paper Section 4.1:
- Open-VFX: 15 visual effect types, 1003 samples total
- Each sample: reference video + text prompt + effect category
- Resolution: varies, resized to 480x832 for training/evaluation
- Also supports MovieGenBench for additional evaluation

Usage:
    python scripts/prepare_dataset.py --output_dir /data/datasets/Open-VFX
    python scripts/prepare_dataset.py --download --output_dir /data/datasets/Open-VFX
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Open-VFX dataset metadata (from paper Table 1)
OPEN_VFX_CATEGORIES = [
    "fire_effects",
    "water_effects",
    "lightning_effects",
    "smoke_effects",
    "particle_effects",
    "glow_effects",
    "ice_frost_effects",
    "wind_effects",
    "explosion_effects",
    "magic_effects",
    "dissolve_effects",
    "distortion_effects",
    "light_beam_effects",
    "energy_effects",
    "nature_effects",
]


def create_dataset_structure(output_dir: str):
    """Create the expected directory structure for Open-VFX."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Create category directories
    for category in OPEN_VFX_CATEGORIES:
        (base / "videos" / category).mkdir(parents=True, exist_ok=True)
        (base / "prompts" / category).mkdir(parents=True, exist_ok=True)

    # Create splits
    (base / "splits").mkdir(parents=True, exist_ok=True)

    # Create metadata template
    metadata = {
        "dataset_name": "Open-VFX",
        "version": "1.0",
        "total_samples": 1003,
        "num_categories": 15,
        "categories": OPEN_VFX_CATEGORIES,
        "video_format": "mp4",
        "target_resolution": {"height": 480, "width": 832},
        "target_fps": 16,
        "target_frames": 81,
        "description": "Open-source Visual Effects dataset for P-Flow evaluation",
        "source_paper": "arXiv:2603.22091",
    }

    with open(base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created dataset structure at: {base}")
    print(f"  Categories: {len(OPEN_VFX_CATEGORIES)}")
    print(f"  Directory structure:")
    print(f"    {base}/videos/<category>/  — reference videos")
    print(f"    {base}/prompts/<category>/ — text prompts (JSON)")
    print(f"    {base}/splits/             — train/test splits")
    print(f"    {base}/metadata.json       — dataset metadata")


def create_sample_prompts(output_dir: str):
    """Create sample prompt files for each category."""
    base = Path(output_dir)

    # Sample prompts for each category (representative examples from paper)
    sample_prompts = {
        "fire_effects": [
            "A campfire with dancing orange flames creating swirling patterns against a dark background",
            "Blue fire spreading across a surface with intense heat distortion",
            "A phoenix emerging from flames with trailing fire particles",
        ],
        "water_effects": [
            "Crystal clear water cascading over rocks with splashing droplets",
            "Underwater scene with light rays penetrating through bubbles",
            "Rain droplets creating ripples on a still pond surface",
        ],
        "lightning_effects": [
            "Bright electrical arcs jumping between two metal conductors",
            "A thunderstorm with forking lightning illuminating dark clouds",
            "Energy crackling along a wire with blue-white sparks",
        ],
        "smoke_effects": [
            "Thick black smoke billowing upward with turbulent swirling motion",
            "Ethereal wisps of smoke in slow motion, catching colored light",
            "Smoke rings expanding and dissipating in still air",
        ],
        "particle_effects": [
            "Golden sparkles floating upward like fireflies in a dark forest",
            "Confetti explosion with multicolored particles falling slowly",
            "Dust motes dancing in a beam of sunlight",
        ],
        "glow_effects": [
            "A neon sign flickering to life with warm orange glow spreading outward",
            "Bioluminescent organisms pulsing with blue-green light underwater",
            "A magical orb emitting pulsating soft white light",
        ],
        "ice_frost_effects": [
            "Frost crystals rapidly spreading across a window pane in intricate patterns",
            "Ice forming on a surface in time-lapse with branching crystal growth",
            "A freezing wave turning to ice mid-crash",
        ],
        "wind_effects": [
            "Strong gusts making tall grass bend and wave in complex patterns",
            "A tornado forming with debris spiraling upward",
            "Cherry blossom petals caught in a gentle breeze swirling",
        ],
        "explosion_effects": [
            "A firework bursting into a chrysanthemum pattern of gold sparks",
            "A controlled demolition with expanding dust cloud",
            "A supernova-like burst of light and debris expanding outward",
        ],
        "magic_effects": [
            "A wizard casting a spell with purple energy spiraling from their hands",
            "A portal opening with swirling blue energy and lightning at edges",
            "Runes glowing on the ground in a circular pattern",
        ],
        "dissolve_effects": [
            "A person dissolving into golden particles floating away",
            "An object shattering into tiny fragments that drift apart",
            "Sand sculpture being blown away grain by grain in the wind",
        ],
        "distortion_effects": [
            "Heat haze causing wavering distortion above a hot surface",
            "A lens warping effect expanding from center outward",
            "Reality bending and folding like a glitch in the matrix",
        ],
        "light_beam_effects": [
            "Volumetric god rays streaming through gaps in clouds",
            "A laser beam cutting through fog with visible light scattering",
            "Spotlight beams sweeping through a dark concert venue",
        ],
        "energy_effects": [
            "Plasma ball with tendrils of energy reaching toward a finger",
            "An energy shield absorbing impacts with ripple effects",
            "Chi energy flowing along meridian lines of a martial artist",
        ],
        "nature_effects": [
            "Aurora borealis with green and purple curtains of light undulating",
            "A time-lapse of clouds forming and dissipating rapidly",
            "Volcanic lava flowing downhill with glowing orange surface",
        ],
    }

    for category, prompts in sample_prompts.items():
        prompts_dir = base / "prompts" / category
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompts_data = []
        for i, prompt in enumerate(prompts):
            prompts_data.append({
                "id": f"{category}_{i:04d}",
                "category": category,
                "prompt": prompt,
                "video_file": f"{category}_{i:04d}.mp4",
            })

        with open(prompts_dir / "prompts.json", "w") as f:
            json.dump(prompts_data, f, indent=2)

    print(f"Created sample prompts for {len(sample_prompts)} categories")


def create_splits(output_dir: str, train_ratio: float = 0.8):
    """Create train/test splits."""
    base = Path(output_dir)
    splits_dir = base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load all prompt files to create split
    all_samples = []
    for category in OPEN_VFX_CATEGORIES:
        prompts_file = base / "prompts" / category / "prompts.json"
        if prompts_file.exists():
            with open(prompts_file) as f:
                samples = json.load(f)
            all_samples.extend(samples)

    if not all_samples:
        print("No samples found, creating placeholder splits")
        # Create placeholder
        train_split = {"samples": [], "num_samples": 0}
        test_split = {"samples": [], "num_samples": 0}
    else:
        import random
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]

        train_split = {"samples": train_samples, "num_samples": len(train_samples)}
        test_split = {"samples": test_samples, "num_samples": len(test_samples)}

    with open(splits_dir / "train.json", "w") as f:
        json.dump(train_split, f, indent=2)
    with open(splits_dir / "test.json", "w") as f:
        json.dump(test_split, f, indent=2)

    print(f"Created splits: train={train_split['num_samples']}, test={test_split['num_samples']}")


def preprocess_videos(input_dir: str, output_dir: str, target_h: int = 480,
                      target_w: int = 832, target_frames: int = 81, target_fps: int = 16):
    """
    Preprocess videos to target resolution and frame count.
    Uses ffmpeg for efficient video processing.
    """
    import subprocess

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    video_files = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi")) + list(input_path.rglob("*.mov"))

    print(f"Found {len(video_files)} videos to preprocess")

    for video_file in video_files:
        rel_path = video_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".mp4")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-i", str(video_file),
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(target_fps),
            "-frames:v", str(target_frames),
            "-c:v", "libx264", "-crf", "18",
            "-an",  # No audio
            str(out_file),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"  Processed: {rel_path}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {rel_path} - {e.stderr.decode()[:100]}")


def download_moviegen_bench(output_dir: str):
    """
    Download MovieGenBench prompts for additional evaluation.
    MovieGenBench: 1003 diverse prompts from Meta's MovieGen paper.
    """
    base = Path(output_dir)
    moviegen_dir = base / "moviegen_bench"
    moviegen_dir.mkdir(parents=True, exist_ok=True)

    print("MovieGenBench download:")
    print("  The MovieGenBench prompts can be obtained from:")
    print("  https://github.com/facebookresearch/MovieGenBench")
    print(f"  Place MovieGenVideoBench.txt in: {moviegen_dir}/")
    print()
    print("  Expected format: one prompt per line, 1003 total prompts")

    # Create placeholder info
    info = {
        "name": "MovieGenVideoBench",
        "source": "https://github.com/facebookresearch/MovieGenBench",
        "num_prompts": 1003,
        "description": "Diverse video generation prompts from Meta MovieGen",
        "usage": "Additional evaluation benchmark for P-Flow",
    }
    with open(moviegen_dir / "README.json", "w") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare Open-VFX dataset for P-Flow")
    parser.add_argument("--output_dir", type=str, default="/data/datasets/Open-VFX",
                       help="Output directory for dataset")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset (if available)")
    parser.add_argument("--preprocess", action="store_true",
                       help="Preprocess videos to target resolution")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Input directory for raw videos (with --preprocess)")
    parser.add_argument("--moviegen", action="store_true",
                       help="Setup MovieGenBench evaluation prompts")

    args = parser.parse_args()

    print("=" * 60)
    print("Open-VFX Dataset Preparation")
    print("=" * 60)

    # Always create structure
    create_dataset_structure(args.output_dir)
    create_sample_prompts(args.output_dir)
    create_splits(args.output_dir)

    if args.preprocess and args.input_dir:
        print("\n[Preprocessing videos...]")
        preprocess_videos(args.input_dir, os.path.join(args.output_dir, "videos"))

    if args.moviegen:
        print("\n[Setting up MovieGenBench...]")
        download_moviegen_bench(args.output_dir)

    if args.download:
        print("\n[Download]")
        print("  Open-VFX dataset download instructions:")
        print("  1. Check paper supplementary materials for dataset link")
        print("  2. Or contact authors at the paper's repository")
        print(f"  3. Place videos in: {args.output_dir}/videos/<category>/")
        print(f"  4. Run: python prepare_dataset.py --preprocess --input_dir <raw_dir>")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"  Location: {args.output_dir}")
    print(f"  Next: Place reference videos and run preprocessing")
    print("=" * 60)


if __name__ == "__main__":
    main()
