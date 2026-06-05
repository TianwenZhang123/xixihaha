"""
测试 TransNetV2 切分镜效果。

用法:
    pip install transnetv2-pytorch
    python scripts/test_shot_detect.py
"""

import sys
import time
import subprocess
import json
import numpy as np

# ─── 配置 ────────────────────────────────────────────────────
VIDEO_PATH = "data/generated_videos/human_action/地铁里的疲惫上班族/full_video.mp4"
THRESHOLD = 0.5  # 边界判定阈值，可调
# ─────────────────────────────────────────────────────────────


def get_video_info(video_path: str) -> dict:
    """用 ffprobe 获取视频的真实 fps 和时长"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr}")
    
    info = json.loads(result.stdout)
    
    # 从 video stream 中获取 fps
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    
    if video_stream is None:
        raise RuntimeError("未找到视频流")
    
    # 获取精确 fps (r_frame_rate 格式如 "16/1" 或 "30000/1001")
    r_frame_rate = video_stream.get("r_frame_rate", "16/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den
    
    # 获取精确时长
    # 优先用 stream duration，其次用 format duration
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
        "r_frame_rate": r_frame_rate,
    }


print("=" * 60)
print("  TransNetV2 切分镜测试")
print("=" * 60)

# 0. 获取视频真实信息
print(f"\n[0] 获取视频元信息: {VIDEO_PATH}")
video_info = get_video_info(VIDEO_PATH)
fps = video_info["fps"]
video_duration = video_info["duration"]
print(f"    真实 FPS: {fps:.4f}")
print(f"    真实时长: {video_duration:.4f}s")
print(f"    帧率字符串: {video_info['r_frame_rate']}")
if video_info["nb_frames"] > 0:
    print(f"    文件记录帧数: {video_info['nb_frames']}")

# 1. 导入模型
print("\n[1] 加载 TransNetV2 模型...")
try:
    from transnetv2_pytorch import TransNetV2
except ImportError:
    print("ERROR: transnetv2-pytorch 未安装!")
    print("请执行: pip install transnetv2-pytorch")
    sys.exit(1)

model = TransNetV2()
print(f"    模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 2. 预测
print(f"\n[2] 对视频进行预测: {VIDEO_PATH}")
start_time = time.time()

video_frames, single_frame_preds, all_frame_preds = model.predict_video(VIDEO_PATH)

# 确保是 numpy array（某些版本返回 Tensor）
if hasattr(single_frame_preds, 'numpy'):
    single_frame_preds = single_frame_preds.numpy()
single_frame_preds = np.asarray(single_frame_preds).flatten()

elapsed = time.time() - start_time
total_frames = len(video_frames)
print(f"    视频帧数: {total_frames}")
print(f"    预测耗时: {elapsed:.2f}s")
print(f"    处理速度: {total_frames / elapsed:.0f} fps")

# 校验帧数与时长的一致性
calculated_duration = total_frames / fps
print(f"\n    帧数换算时长: {total_frames} / {fps:.2f} = {calculated_duration:.4f}s")
if video_duration:
    diff = abs(calculated_duration - video_duration)
    print(f"    与视频真实时长差异: {diff:.4f}s {'✓' if diff < 0.1 else '⚠️ 偏差较大'}")

# 3. 查看逐帧概率
print(f"\n[3] 逐帧边界概率统计:")
print(f"    min={single_frame_preds.min():.4f}, max={single_frame_preds.max():.4f}, "
      f"mean={single_frame_preds.mean():.4f}")
print(f"    超过阈值 {THRESHOLD} 的帧数: {(single_frame_preds > THRESHOLD).sum()}")

# 打印概率较高的帧（可能是边界）
high_prob_frames = np.where(single_frame_preds > 0.2)[0]
if len(high_prob_frames) > 0:
    print(f"\n    概率 > 0.2 的帧:")
    for f in high_prob_frames:
        print(f"      帧 {f}: probability = {single_frame_preds[f]:.4f}")

# 4. 获取镜头列表（精确时间）
print(f"\n[4] 检测到的镜头 (threshold={THRESHOLD}):")
scenes = model.predictions_to_scenes(single_frame_preds, threshold=THRESHOLD)
print(f"    共 {len(scenes)} 个镜头:")
print(f"    {'镜头':<6} {'起始帧':<8} {'结束帧':<8} {'帧数':<6} "
      f"{'起始时间':<10} {'结束时间':<10} {'时长(s)':<8}")
print(f"    {'-'*65}")

for i, (start, end) in enumerate(scenes):
    num_frames = end - start + 1
    start_time_s = start / fps
    # 最后一个镜头的结束时间 clamp 到视频真实时长
    if i == len(scenes) - 1 and video_duration:
        end_time_s = min((end + 1) / fps, video_duration)
    else:
        end_time_s = (end + 1) / fps  # end+1 因为 end 是包含的最后一帧
    duration_s = end_time_s - start_time_s
    print(f"    Shot {i}: 帧[{start:>4} ~ {end:>4}], "
          f"{num_frames:>4}帧, "
          f"{start_time_s:>7.3f}s ~ {end_time_s:>7.3f}s, "
          f"{duration_s:.3f}s")

# 显示总时长验证
total_detected = scenes[-1][1] / fps if len(scenes) > 0 else 0
print(f"\n    所有镜头覆盖: 0.000s ~ {min((scenes[-1][1]+1)/fps, video_duration):.3f}s")
print(f"    视频真实时长:           {video_duration:.3f}s")

# 5. 对比已有的 scene_1/2/3
print(f"\n[5] 对比:")
print(f"    已有文件: scene_1.mp4, scene_2.mp4, scene_3.mp4 (人工/脚本切分)")
print(f"    TransNetV2 自动检测到 {len(scenes)} 个镜头")
if len(scenes) == 3:
    print(f"    ✓ 数量一致！与已有的 3 段吻合")
elif len(scenes) < 3:
    print(f"    △ 检测到更少的镜头，可能阈值太高或转场是渐变的")
    print(f"    可尝试降低阈值: THRESHOLD = 0.3")
else:
    print(f"    △ 检测到更多镜头，可能视频内有额外的快速切换")

# 6. 输出可直接用于 ffmpeg 切分的命令
print(f"\n[6] ffmpeg 精确切分命令 (可直接复制使用):")
for i, (start, end) in enumerate(scenes):
    start_time_s = start / fps
    if i == len(scenes) - 1 and video_duration:
        end_time_s = min((end + 1) / fps, video_duration)
    else:
        end_time_s = (end + 1) / fps
    duration_s = end_time_s - start_time_s
    print(f"    ffmpeg -ss {start_time_s:.3f} -i {VIDEO_PATH} "
          f"-t {duration_s:.3f} -c copy shot_{i}.mp4")

print("\n" + "=" * 60)
print("  测试完成!")
print("=" * 60)
