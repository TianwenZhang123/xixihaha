"""
VMAD Evaluation Suite.

评测指标体系 (对应论文 Section 4):

运动保真度 (Motion Fidelity):
    - Optical Flow EPE: 光流端点误差 (越低越好)
    - X-CLIP Motion Sim: 时序感知的运动相似度 (越高越好)
    - CLIP Frame Sim: 帧级视觉相似度 (越高越好)

内容一致性 (Content Consistency):
    - CLIP Content Score: 生成视频与 content prompt 的匹配度
    - Cross-Content Variance: 同一 asset 不同 content 的运动一致性 (越低越好)

内容解耦度 (Content Disentanglement):
    - Content Leakage Score: 生成视频中源视频内容残留程度 (越低越好)

分布质量 (Distribution Quality):
    - STREAM-T/F/D: 视频分布统计距离

消融实验支持:
    - 每个指标支持按 flag 组合分组统计
    - 支持 baseline 对比 (text-only, noise-only, full)
"""

__all__ = [
    "run_motion_fidelity_eval",
    "run_content_consistency_eval",
    "run_full_eval",
]
