"""
测试 TransNetV2 切分镜效果。

用法:
    pip install transnetv2-pytorch
    python scripts/test_shot_detect.py
"""

import sys
import time
import numpy as np

# ─── 配置 ────────────────────────────────────────────────────
VIDEO_PATH = "data/generated_videos/human_action/地铁里的疲惫上班族/full_video.mp4"
THRESHOLD = 0.5  # 边界判定阈值，可调
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  TransNetV2 切分镜测试")
print("=" * 60)

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

elapsed = time.time() - start_time
print(f"    视频帧数: {len(video_frames)}")
print(f"    预测耗时: {elapsed:.2f}s")
print(f"    处理速度: {len(video_frames) / elapsed:.0f} fps")

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

# 4. 获取镜头列表
print(f"\n[4] 检测到的镜头 (threshold={THRESHOLD}):")
scenes = model.predictions_to_scenes(single_frame_preds, threshold=THRESHOLD)
print(f"    共 {len(scenes)} 个镜头:")
print(f"    {'镜头':<6} {'起始帧':<8} {'结束帧':<8} {'帧数':<6} {'时长(s)':<8}")
print(f"    {'-'*40}")

# 假设 fps（从帧数和常见 fps 推断）
total_frames = len(video_frames)
# 尝试几种常见 fps
for assumed_fps in [16, 24, 30]:
    duration = total_frames / assumed_fps
    if 14 <= duration <= 16:  # 15s 左右
        fps = assumed_fps
        break
else:
    fps = 16  # 默认

print(f"    (推断 FPS={fps}, 总时长≈{total_frames/fps:.1f}s)")
print()

for i, (start, end) in enumerate(scenes):
    num_frames = end - start + 1
    duration = num_frames / fps
    start_time_s = start / fps
    end_time_s = end / fps
    print(f"    Shot {i}: 帧[{start:>4} ~ {end:>4}], "
          f"{num_frames:>4}帧, "
          f"{duration:.2f}s  "
          f"({start_time_s:.2f}s ~ {end_time_s:.2f}s)")

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

print("\n" + "=" * 60)
print("  测试完成!")
print("=" * 60)
