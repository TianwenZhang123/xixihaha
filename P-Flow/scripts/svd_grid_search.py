#!/usr/bin/env python3
"""
α (噪声混合权重) 网格搜索脚本。

对不同 alpha 值跑 run.py --noise_prior，汇总评测结果。

用法:
    python scripts/svd_grid_search.py \
        --data_dir /root/autodl-tmp/data/video-200/water_mark_out \
        --caption_dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
        --output_dir /root/autodl-tmp/outputs/alpha_search

等价于依次执行:
    python run.py --data_dir ... --caption_dir ... --output_dir .../alpha_0.001 --noise_prior --alpha 0.001 ...
    python run.py --data_dir ... --caption_dir ... --output_dir .../alpha_0.002 --noise_prior --alpha 0.002 ...
    ...
"""

import subprocess
import sys
from pathlib import Path

# ─── 搜索空间 ───
ALPHA_VALUES = [0.001, 0.002, 0.003, 0.004, 0.005]
SAMPLE_IDS = "7 17 21 31 32 33 34 43 46 47"


def main():
    import argparse
    p = argparse.ArgumentParser(description="α 网格搜索")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--caption_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/outputs/alpha_search")
    p.add_argument("--alpha_values", type=float, nargs="+", default=ALPHA_VALUES)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_py = str(Path(__file__).parent.parent / "run.py")
    output_root = Path(args.output_dir)

    for alpha in args.alpha_values:
        out_dir = output_root / f"alpha_{alpha}"
        print(f"\n{'═' * 50}")
        print(f"  α = {alpha}")
        print(f"{'═' * 50}")

        cmd = [
            sys.executable, run_py,
            "--data_dir", args.data_dir,
            "--caption_dir", args.caption_dir,
            "--output_dir", str(out_dir),
            "--sample_ids", *SAMPLE_IDS.split(),
            "--noise_prior",
            "--alpha", str(alpha),
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
            "--seed", str(args.seed),
            "--resume",
        ]

        print(f"  cmd: python run.py ... --alpha {alpha} --output_dir {out_dir}")
        subprocess.run(cmd)

    print(f"\n完成! 各 α 结果在: {output_root}/alpha_*")
    print(f"评测命令:")
    for alpha in args.alpha_values:
        out_dir = output_root / f"alpha_{alpha}"
        print(f"  python evaluation/run_clip_xclip_eval.py --orig-dir {args.data_dir} --gen-dir {out_dir} --caption-dir {args.caption_dir}")


if __name__ == "__main__":
    main()
