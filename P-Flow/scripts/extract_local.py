#!/usr/bin/env python3
"""
在本地 Mac 上下载 MovieGenVideoBench 并提取 200 个视频。

用法:
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/extract_local.py
"""

import os
import gc
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "videos_200"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 prompt → ID 映射
    prompts = (base_dir / "data" / "MovieGenVideoBench.txt").read_text().strip().split("\n")
    prompt_to_id = {p.strip(): i + 1 for i, p in enumerate(prompts)}
    print(f"Prompt list: {len(prompt_to_id)} entries")

    # 加载需要的 ID
    selected_ids = set(
        int(x.strip())
        for x in (base_dir / "data" / "selected_ids.txt").read_text().strip().split("\n")
        if x.strip()
    )
    already = set(int(f.stem) for f in output_dir.glob("*.mp4"))
    need = selected_ids - already
    print(f"Have: {len(already)}, need: {len(need)}")

    if not need:
        print("All 200 done!")
        return

    # 下载 parquet 文件
    print("\nDownloading parquet files from HuggingFace mirror...")
    local_dir = snapshot_download(
        repo_id="meta-ai-for-media-research/movie_gen_video_bench",
        repo_type="dataset",
        local_dir="/tmp/movie_gen_bench",
        allow_patterns="data/test_with_generations-*.parquet",
    )
    print(f"Downloaded to: {local_dir}")

    # 逐文件提取
    parquet_dir = Path(local_dir) / "data"
    files = sorted(parquet_dir.glob("test_with_generations-*.parquet"))
    print(f"\nFound {len(files)} parquet files, extracting...")

    found = len(already)
    for fi, fp in enumerate(files):
        if not need:
            break
        pf = pq.ParquetFile(str(fp))
        for batch in pf.iter_batches(batch_size=1):
            p = batch.column("prompt")[0].as_py()
            vid_id = prompt_to_id.get(p.strip())
            if vid_id is None or vid_id not in need:
                del batch
                continue
            video_bytes = batch.column("video")[0].as_py()
            del batch
            (output_dir / f"{vid_id}.mp4").write_bytes(video_bytes)
            found += 1
            need.discard(vid_id)
            print(f"  [{found}/200] {vid_id}.mp4 ({len(video_bytes) // 1024 // 1024}MB)")
            del video_bytes
            gc.collect()
        print(f"-- file {fi + 1}/{len(files)}, got {found}/200")
        gc.collect()

    print(f"\nDone! {found}/200 videos in {output_dir}")


if __name__ == "__main__":
    main()
