#!/usr/bin/env python3
"""
Download MovieGenVideoBench videos and filter to our 200 selected samples.

MovieGenVideoBench is hosted on HuggingFace:
    https://huggingface.co/datasets/meta-ai-for-media-research/movie_gen_video_bench

This script:
    1. Downloads the full dataset from HuggingFace (videos + metadata)
    2. Filters to the 200 selected IDs (from data/selected_ids.txt)
    3. Copies matching videos to data/video-200/water_mark_out/
    4. Reports any missing videos

Prerequisites:
    pip install huggingface_hub datasets

Usage:
    # Download and filter (default)
    python scripts/download_dataset.py

    # Specify output directory
    python scripts/download_dataset.py --output-dir /root/autodl-tmp/data/video-200/water_mark_out

    # If you already have the videos somewhere, just filter/symlink
    python scripts/download_dataset.py --source-dir /path/to/existing/videos

    # If your collaborator's videos are on the same server
    python scripts/download_dataset.py --source-dir /path/to/Video2Prompt/video-200/water_mark_out

Notes:
    - MovieGenVideoBench contains 1003 prompts. The dataset on HuggingFace
      contains Meta's GENERATED videos (not original reference videos).
    - If you need the ORIGINAL reference videos (water_mark_out), you likely
      need to get them from your collaborator's server, as these are not
      publicly available in the benchmark.
    - This script supports both scenarios: downloading from HF or copying
      from a local source directory.
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/prepare MovieGenVideoBench videos for P-Flow experiments"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/root/autodl-tmp/data/video-200/water_mark_out"),
        help="Where to put the filtered 200 videos"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=None,
        help="If you already have the videos locally (e.g., collaborator's dir), "
             "copy/symlink from here instead of downloading"
    )
    parser.add_argument(
        "--ids-file", type=Path,
        default=Path(__file__).parent.parent / "data" / "selected_ids.txt",
        help="File with selected video IDs (one per line)"
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Create symlinks instead of copying (saves disk space)"
    )
    parser.add_argument(
        "--download-hf", action="store_true",
        help="Download from HuggingFace dataset (Meta's generated videos)"
    )
    parser.add_argument(
        "--caption-dir", type=Path, default=None,
        help="Also copy captions to this directory"
    )
    parser.add_argument(
        "--caption-source", type=Path, default=None,
        help="Source directory for captions (if different from default)"
    )
    return parser.parse_args()


def load_selected_ids(ids_file: Path) -> list[int]:
    """Load the 200 selected video IDs."""
    if not ids_file.exists():
        raise FileNotFoundError(f"IDs file not found: {ids_file}")
    ids = []
    for line in ids_file.read_text().strip().split("\n"):
        line = line.strip()
        if line and line.isdigit():
            ids.append(int(line))
    return ids


def copy_from_source(source_dir: Path, output_dir: Path, selected_ids: list[int],
                     use_symlink: bool = False) -> tuple[int, list[int]]:
    """Copy or symlink videos from a local source directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    missing = []

    for vid_id in selected_ids:
        src = source_dir / f"{vid_id}.mp4"
        dst = output_dir / f"{vid_id}.mp4"

        if not src.exists():
            missing.append(vid_id)
            continue

        if dst.exists() or dst.is_symlink():
            found += 1
            continue

        if use_symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        found += 1

    return found, missing


def download_from_huggingface(output_dir: Path, selected_ids: list[int]):
    """
    Download videos from HuggingFace dataset.

    Note: This downloads Meta's GENERATED videos, not original reference videos.
    For original reference videos, use --source-dir with your collaborator's data.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print("Downloading from HuggingFace: meta-ai-for-media-research/movie_gen_video_bench")
    print("WARNING: This downloads Meta's GENERATED videos, not original reference videos.")
    print("         For original videos, use --source-dir with your collaborator's data.")
    print()

    # Download the dataset
    cache_dir = output_dir.parent / ".hf_cache"
    local_dir = snapshot_download(
        repo_id="meta-ai-for-media-research/movie_gen_video_bench",
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir / "movie_gen_video_bench"),
    )

    print(f"Downloaded to: {local_dir}")
    print("Please check the downloaded structure and map videos to IDs manually.")
    print(f"Selected IDs: {len(selected_ids)} videos needed")

    # List what was downloaded
    local_path = Path(local_dir)
    video_files = list(local_path.rglob("*.mp4"))
    print(f"Found {len(video_files)} video files in download")

    return local_path


def main():
    args = parse_args()

    # Load selected IDs
    selected_ids = load_selected_ids(args.ids_file)
    print(f"Loaded {len(selected_ids)} selected video IDs")

    if args.source_dir:
        # Copy from local source
        print(f"Source: {args.source_dir}")
        print(f"Output: {args.output_dir}")
        print(f"Mode: {'symlink' if args.symlink else 'copy'}")
        print()

        if not args.source_dir.exists():
            print(f"ERROR: Source directory does not exist: {args.source_dir}")
            sys.exit(1)

        found, missing = copy_from_source(
            args.source_dir, args.output_dir, selected_ids, args.symlink
        )

        print(f"Done! {found}/{len(selected_ids)} videos prepared")
        if missing:
            print(f"WARNING: {len(missing)} videos not found in source:")
            for vid_id in missing[:20]:
                print(f"  - {vid_id}.mp4")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")

    elif args.download_hf:
        # Download from HuggingFace
        download_from_huggingface(args.output_dir, selected_ids)

    else:
        # Print instructions
        print()
        print("=" * 70)
        print("HOW TO PREPARE THE 200 REFERENCE VIDEOS")
        print("=" * 70)
        print()
        print("Option 1: Copy from collaborator's server (RECOMMENDED)")
        print("  If your collaborator's data is on the same server:")
        print()
        print("    python scripts/download_dataset.py \\")
        print("        --source-dir /path/to/Video2Prompt/video-200/water_mark_out \\")
        print("        --output-dir /root/autodl-tmp/data/video-200/water_mark_out")
        print()
        print("  Or use symlinks to save space:")
        print()
        print("    python scripts/download_dataset.py \\")
        print("        --source-dir /path/to/Video2Prompt/video-200/water_mark_out \\")
        print("        --output-dir /root/autodl-tmp/data/video-200/water_mark_out \\")
        print("        --symlink")
        print()
        print("Option 2: Download from HuggingFace (Meta's generated videos)")
        print("  NOTE: These are Meta's generated videos, NOT original references!")
        print()
        print("    python scripts/download_dataset.py --download-hf")
        print()
        print("Option 3: Manual setup")
        print("  Place 200 videos named {id}.mp4 in:")
        print(f"    {args.output_dir}")
        print()
        print(f"  Required IDs are listed in: {args.ids_file}")
        print("=" * 70)

    # Also handle captions if requested
    if args.caption_dir and args.caption_source:
        print(f"\nCopying captions from {args.caption_source} to {args.caption_dir}")
        args.caption_dir.mkdir(parents=True, exist_ok=True)
        caption_found = 0
        for vid_id in selected_ids:
            src = args.caption_source / f"{vid_id}.txt"
            dst = args.caption_dir / f"{vid_id}.txt"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                caption_found += 1
        print(f"  Copied {caption_found} caption files")


if __name__ == "__main__":
    main()
