"""
批量测试 TransNetV2 切分镜效果。

对 collected_15s_videos 中的所有视频进行镜头检测，
输出每个视频的切分结果统计。

用法:
    python scripts/batch_shot_detect.py
"""

import sys
import time
import subprocess
import json
import numpy as np
from pathlib import Path

# ─── 配置 ────────────────────────────────────────────────────
VIDEO_DIR = Path("data/collected_15s_videos/collected_15s_videos")
THRESHOLD = 0.5
# ─────────────────────────────────────────────────────────────


def get_video_info(video_path: str) -> dict:
    """用 ffprobe 获取视频的真实 fps 和时长"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    info = json.loads(result.stdout)

    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        return None

    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den

    duration = None
    if "duration" in video_stream:
        duration = float(video_stream["duration"])
    elif "duration" in info.get("format", {}):
        duration = float(info["format"]["duration"])

    nb_frames = int(video_stream.get("nb_frames", 0))

    return {
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
    }


def main():
    print("=" * 70)
    print("  TransNetV2 批量切分镜测试")
    print("=" * 70)

    # 收集视频文件
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        print(f"ERROR: 未找到视频文件: {VIDEO_DIR}")
        sys.exit(1)

    print(f"\n共找到 {len(video_files)} 个视频文件")
    print(f"阈值: {THRESHOLD}")

    # 加载模型
    print("\n[加载模型]...")
    try:
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        print("ERROR: transnetv2-pytorch 未安装! pip install transnetv2-pytorch")
        sys.exit(1)

    model = TransNetV2()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 批量处理
    print(f"\n{'='*70}")
    print(f"{'视频':<12} {'时长(s)':<9} {'帧数':<7} {'FPS':<7} "
          f"{'镜头数':<7} {'切分点帧号':<20} {'各段时长'}")
    print(f"{'-'*70}")

    total_time = 0
    results = []

    for video_path in video_files:
        # 获取视频信息
        info = get_video_info(video_path)
        if info is None:
            print(f"{video_path.name:<12} [ERROR: 无法读取视频信息]")
            continue

        fps = info["fps"]
        duration = info["duration"]

        # 预测
        t0 = time.time()
        try:
            video_frames, single_frame_preds, all_frame_preds = model.predict_video(str(video_path))
        except Exception as e:
            print(f"{video_path.name:<12} [ERROR: {e}]")
            continue

        # 转换为 numpy
        if hasattr(single_frame_preds, 'numpy'):
            single_frame_preds = single_frame_preds.numpy()
        single_frame_preds = np.asarray(single_frame_preds).flatten()

        elapsed = time.time() - t0
        total_time += elapsed

        # 检测镜头
        scenes = model.predictions_to_scenes(single_frame_preds, threshold=THRESHOLD)
        num_shots = len(scenes)

        # 切分点（边界帧）
        cut_frames = [str(scenes[i][0]) for i in range(1, len(scenes))]
        cut_str = ",".join(cut_frames) if cut_frames else "无切分"

        # 各段时长
        shot_durations = []
        for i, (start, end) in enumerate(scenes):
            s_time = start / fps
            if i == len(scenes) - 1 and duration:
                e_time = min((end + 1) / fps, duration)
            else:
                e_time = (end + 1) / fps
            shot_durations.append(f"{e_time - s_time:.2f}s")
        dur_str = " | ".join(shot_durations)

        print(f"{video_path.name:<12} {duration:<9.2f} {len(video_frames):<7} {fps:<7.1f} "
              f"{num_shots:<7} {cut_str:<20} {dur_str}")

        results.append({
            "name": video_path.name,
            "duration": duration,
            "fps": fps,
            "frames": len(video_frames),
            "num_shots": num_shots,
            "scenes": scenes,
            "cut_frames": cut_frames,
        })

    # 汇总统计
    print(f"\n{'='*70}")
    print(f"  汇总统计")
    print(f"{'='*70}")

    if results:
        shot_counts = [r["num_shots"] for r in results]
        durations = [r["duration"] for r in results]

        print(f"  视频总数: {len(results)}")
        print(f"  总处理时间: {total_time:.1f}s")
        print(f"  平均处理时间: {total_time/len(results):.2f}s/视频")
        print(f"\n  视频时长: min={min(durations):.2f}s, max={max(durations):.2f}s, "
              f"mean={np.mean(durations):.2f}s")
        print(f"  镜头数分布: min={min(shot_counts)}, max={max(shot_counts)}, "
              f"mean={np.mean(shot_counts):.1f}")

        # 镜头数分布
        from collections import Counter
        dist = Counter(shot_counts)
        print(f"\n  镜头数分布详情:")
        for k in sorted(dist.keys()):
            bar = "█" * dist[k]
            print(f"    {k:>2} 个镜头: {dist[k]:>3} 个视频  {bar}")

        # 单镜头视频（无切分）
        single_shot = [r["name"] for r in results if r["num_shots"] == 1]
        if single_shot:
            print(f"\n  单镜头视频（无切分点）: {len(single_shot)} 个")
            for name in single_shot[:10]:
                print(f"    {name}")
            if len(single_shot) > 10:
                print(f"    ...共 {len(single_shot)} 个")

    print(f"\n{'='*70}")
    print("  批量测试完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
