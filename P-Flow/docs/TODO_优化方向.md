# P-Flow 优化方向 TODO

> 创建时间: 2025-06  
> 目标: 在不破坏现有功能的前提下，移植 Position-Aware Gradient Scaling + RF-Solver (2nd-order Taylor) 到 P-Flow

---

## 背景分析

### 当前 P-Flow v6.0 状态

- VelocityMatcher: 30步轻量版，无 position-aware，||Δe||≈8.5
- FlowMatchingInverter: 支持 Euler + Midpoint，无 RF-Solver
- embed_strength = 0.005 (验证最优，但可能需要 grid search 确认)

### VMAD 参考数据

- PositionAwareVelocityMatcher: 100步，有 position-aware，||Δe||≈36
- Position-aware 使 Δe 集中在 position 0 (attention sink)，减少浪费
- RF-Solver 2nd-order Taylor: 减少反演离散化误差，使 η_inv 更精确

### 预期收益

- Position-Aware: 让 30 步 velocity matching 的梯度更有效利用 → 可能用 30 步达到之前 50+ 步的效果
- RF-Solver: 反演精度提升 → η_inv 更好 → v* = z₀ - η_inv 更准确 → velocity matching 起点更优

---

## TODO 清单

### 1. [待定] embed_strength Grid Search

- [ ] 实验配置: es ∈ [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
- [ ] 运行评估，确认 P-Flow 的最优 embed_strength
- [ ] 根据 ||Δe||≈8.5 推算: es=0.02~0.035 → ||injection|| ≈ 0.17~0.30 (对齐 VMAD 有效区间)
- **状态**: 等待 GPU 资源

### 2. [进行中] Position-Aware Gradient Scaling — 实现

**文件**: `src/velocity_matching.py`

- [x] 分析 VMAD 的 `PositionAwareVelocityMatcher` 实现
- [ ] 在 `VelocityMatcher` 添加 `position_aware: bool = False` 参数
- [ ] 实现 `_initialize_position_weights()` (从 VMAD 移植 U-shape profile)
- [ ] 在优化循环 backward 后添加 gradient scaling (有 flag 控制)
- [ ] 可选: 添加 `_compute_position_regularization()` (lambda_pos 控制)

**关键代码** (from VMAD lines 421-430):
```python
if self.position_aware and delta_e.grad is not None:
    grad_scale = 1.0 / (position_weights + 0.1)
    grad_scale = grad_scale / grad_scale.mean()
    if delta_e.grad.dim() == 3:
        delta_e.grad.data *= grad_scale.unsqueeze(0).unsqueeze(-1)
```

**设计约束**:
- `position_aware=False` 时行为完全不变 (default off)
- 不增加额外的 model forward pass (零额外计算开销)

### 3. [进行中] RF-Solver (2nd-order Taylor) — 实现

**文件**: `src/flow_matching.py`

- [x] 研究 RF-Solver 公式
- [ ] 新增 `invert_rfsolver()` 方法
- [ ] 实现 2nd-order Taylor 展开: `x_{t-dt} = x_t - dt * v - (dt²/2) * dv/dt`
- [ ] `dv/dt` 使用前一步的速度差分估计

**RF-Solver-2 公式**:
```
# Standard Euler: x_{t-dt} = x_t - dt * v_θ(x_t, t)
# RF-Solver-2: x_{t-dt} = x_t - dt * v_θ(x_t, t) - (dt²/2) * dv/dt
# where dv/dt ≈ (v_θ(x_t, t) - v_θ(x_{t-dt_prev}, t-dt_prev)) / dt_prev
#
# 第一步退化为 Euler (因为没有 previous velocity)
# 后续步使用 2nd-order 校正
```

**设计约束**:
- 新方法独立于 `invert()` 和 `invert_midpoint()`，不修改已有代码
- 首步退化为 Euler（无历史速度），后续步用 2nd-order Taylor 校正

### 4. [待做] Pipeline 集成

**文件**: `src/pipeline.py` + `run.py`

- [ ] `PFlowConfig` 添加:
  - `use_position_aware: bool = False`
  - `use_rfsolver: bool = False`
- [ ] `pipeline.py` 的 `_compute_noise_prior()` 中添加 RF-Solver 分支
- [ ] `pipeline.py` 的 `_compute_delta_e()` 中传递 position_aware flag
- [ ] `run.py` 添加 `--position_aware` 和 `--rfsolver` CLI flags
- [ ] 更新快捷组合文档

### 5. [待做] 文档更新

**文件**: `docs/P-Flow故事与技术演进.md`

- [ ] 新增 v7.0 章节: Position-Aware + RF-Solver
- [ ] 更新 Flag 体系表格
- [ ] 更新计算开销分析
- [ ] 更新文件结构说明

### 6. [待做] 验证

- [ ] 所有现有 flag 组合仍然正常工作 (regression test)
- [ ] `--position_aware` 单独使用时 ||Δe|| 应更集中 (position 0 占比提升)
- [ ] `--rfsolver` 的 η_inv 与 midpoint 对比: 理论上精度更高、步数可减少
- [ ] Grid search: position_aware + 不同 embed_strength 的最优组合

---

## 实现原则

1. **Additive Only**: 所有新功能通过新 flag 控制，默认关闭
2. **Zero Regression**: 不修改任何已有方法的签名和行为
3. **Minimal Complexity**: 移植 VMAD 的核心逻辑，去除不必要的复杂性
4. **Independent Testing**: 每个新功能可独立测试和消融

---

## 参考资源

- VMAD position-aware 实现: `VMAD/src/velocity_matching.py` (lines 184-243, 421-430)
- RF-Solver 论文: "RF-Solver: A Second-Order ODE Solver for Rectified Flow" (2024)
- P-Flow 当前文档: `docs/P-Flow故事与技术演进.md`
