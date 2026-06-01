# Layer 3 (Noise Prior Blending) 修复方案

> 状态：设计完成，待实施
> 依赖：P0 多样本验证之后再做

---

## 问题根因

Layer 3 在当前实现中导致 CLIP 暴跌 -0.10~-0.15，根因是 **latents 生成路径差异**：

```
当前代码（有害）：
1. apply_noise_prior() 在外部用 torch.randn(generator=gen) 生成 eta_random
2. 混合：latents = √α·η_motion + √(1-α)·eta_random  
3. 把 latents 传给 pipeline(latents=latents)
4. pipeline 检测到 latents 已传入，跳过内部 prepare_latents

正常路径（不开 blend 时）：
1. pipeline 内部调用 prepare_latents()
2. prepare_latents 用 randn_tensor(shape, generator=gen) 生成噪声
3. 直接用这个噪声开始去噪

问题：torch.randn 和 randn_tensor 即使给同一个 generator，产生的随机数完全不同！
原因：randn_tensor 可能用不同的内存布局、dtype 处理路径或分块策略。
```

---

## 修复方案

**核心思路**：不再在外部生成噪声。让 pipeline 正常生成初始 latents（走 `prepare_latents` 路径），然后在**生成之后、去噪之前**将 eta_motion blend 进去。

### 方案选择

**方案 A（推荐）**：手动调用 `pipe.prepare_latents` 获得正确路径的噪声，再 blend。

优点：不修改 diffusers 源码，只改 VMAD 代码。
缺点：需要知道 prepare_latents 的参数签名。

---

## 具体代码修改

### 改动 1：`src/pipeline.py` — 新增 `_prepare_blended_latents` 方法

在 `_generate` 方法之前（约 L779 前）添加新方法：

```python
def _prepare_blended_latents(
    self,
    eta_motion: torch.Tensor,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Layer 3 修复版：用 pipeline 内部路径生成基础噪声，再 blend eta_motion。
    
    这保证了基础噪声与 "不开 blend" 时的噪声路径完全一致，
    eta_motion 只做方向性引导，不破坏随机数路径。
    
    公式：latents = √α·η_motion + √(1-α)·η_baseline
    其中 η_baseline 由 pipeline 的 prepare_latents 生成（randn_tensor 路径）
    """
    cfg = self.config
    
    # 计算 latent shape（与 WanPipeline.__call__ 内部逻辑一致）
    vae_scale_factor_temporal = self.pipe.vae_scale_factor_temporal
    vae_scale_factor_spatial = self.pipe.vae_scale_factor_spatial
    
    num_channels_latents = self.pipe.transformer.config.in_channels
    height = cfg.height // vae_scale_factor_spatial
    width = cfg.width // vae_scale_factor_spatial
    num_frames = (cfg.num_frames - 1) // vae_scale_factor_temporal + 1
    
    shape = (1, num_channels_latents, num_frames, height, width)
    
    # 用 pipeline 内部的 prepare_latents 生成基础噪声
    # 这与不传 latents 时 pipeline 内部走的路径完全一致
    eta_baseline = self.pipe.prepare_latents(
        shape=shape,
        dtype=torch.bfloat16,
        device=self.device,
        generator=generator,
    )
    
    # Blend eta_motion into baseline noise
    alpha_scaled = min(alpha, 1.0)
    eta_motion = eta_motion.to(device=self.device, dtype=eta_baseline.dtype)
    
    # 确保 shape 一致
    if eta_motion.shape != eta_baseline.shape:
        logger.warning(
            f"Shape mismatch: eta_motion={eta_motion.shape}, "
            f"eta_baseline={eta_baseline.shape}. Attempting reshape."
        )
        if eta_motion.dim() == 4:
            eta_motion = eta_motion.unsqueeze(0)
    
    latents = (alpha_scaled ** 0.5) * eta_motion + ((1 - alpha_scaled) ** 0.5) * eta_baseline
    
    logger.info(
        f"    Blended latents: α={alpha_scaled:.4f}, "
        f"||η_motion||={eta_motion.norm():.1f}, ||η_baseline||={eta_baseline.norm():.1f}, "
        f"||latents||={latents.norm():.1f}"
    )
    
    return latents
```

### 改动 2：`src/pipeline.py` L657-667 — 修改 Layer 3 调用逻辑

**原代码**（L657-667）：
```python
        # -- Layer 3: Noise prior blending (structural guidance) --
        # Following P-Flow: alpha=0.001 provides minimal but sufficient structural guidance
        if cfg.use_blend:
            latents = self._asset_manager.apply_noise_prior(
                asset, alpha=cfg.alpha, strength=strength, generator=generator
            )
            if latents is not None:
                logger.info(f"  [Layer 3] Noise prior: alpha={cfg.alpha}")
        else:
            latents = None
            logger.info("  [Layer 3] Skipped (use_blend=False)")
```

**新代码**：
```python
        # -- Layer 3: Noise prior blending (structural guidance) --
        # FIX: Use pipeline's own prepare_latents to generate baseline noise,
        # then blend eta_motion externally. This preserves the random number path.
        if cfg.use_blend and asset.eta_motion is not None:
            blend_alpha = cfg.blend_alpha if hasattr(cfg, 'blend_alpha') else cfg.alpha
            latents = self._prepare_blended_latents(
                eta_motion=asset.eta_motion,
                alpha=blend_alpha * strength,
                generator=generator,
            )
            logger.info(f"  [Layer 3] Noise prior (fixed): blend_alpha={blend_alpha}")
        else:
            latents = None
            if not cfg.use_blend:
                logger.info("  [Layer 3] Skipped (use_blend=False)")
            else:
                logger.info("  [Layer 3] Skipped (no eta_motion in asset)")
```

### 改动 3：`src/pipeline.py` VMADConfig — 新增 `blend_alpha` 参数（解耦）

在 VMADConfig 的定义中添加独立参数，解耦 Layer 2 和 Layer 3：

```python
@dataclass
class VMADConfig:
    # ... 现有参数 ...
    alpha: float = 0.005          # Layer 2: Δe 注入强度
    blend_alpha: float = 0.001    # Layer 3: noise prior 混合比例（独立于 alpha）
    # ...
```

同时在 `run_batch_extract.py` 的 argparse 中添加：
```python
parser.add_argument('--blend-alpha', type=float, default=0.001,
                    help='Layer 3 noise blending alpha (independent of --alpha)')
```

### 改动 4：删除 `src/motion_asset.py` 中 `apply_noise_prior` 的 generator 参数使用

修复后 `apply_noise_prior` 不再需要直接调用了（被 `_prepare_blended_latents` 取代），但如果要保留向后兼容，可以标记为 deprecated。

---

## 验证步骤

修复后，运行以下验证确认修复正确：

```bash
cd /root/autodl-tmp/videofake/VMAD

# 验证 1：修复后 blend=True + alpha 极小（如 0.0001）应该≈不开 blend
python run_batch_extract.py --apply-only \
    --output-dir /tmp/vmad_l3_fix_verify \
    --sample-ids 7 --alpha 0.005 --blend-alpha 0.0001 \
    --no-token_decode --content SELF --seed 42 -v
# 期望：CLIP ≈ 0.9446（与不开 blend 时几乎一致）

# 验证 2：修复后 blend=True + alpha=0.001
python run_batch_extract.py --apply-only \
    --output-dir /tmp/vmad_l3_fix_a001 \
    --sample-ids 7 --alpha 0.005 --blend-alpha 0.001 \
    --no-token_decode --content SELF --seed 42 -v
# 期望：CLIP ≥ 0.94（不再暴跌），XCLIP 可能有小幅提升

# 验证 3：sweep blend_alpha in {0.001, 0.005, 0.01, 0.05}
# 寻找 Layer 3 的最优混合比例
```

**成功标准**：
- blend_alpha=0.0001 时结果与不开 blend 差异 < 0.002 CLIP
- 存在某个 blend_alpha 使得 CLIP 不降（或微升）且 XCLIP 提升

---

## Layer 1 (Token Decode) 修复方向

Token Decode 的修复相对简单但优先级低（P2）：

**当前问题**：`token_decoder.py` 中搜索用的是输入 embedding 矩阵。

**修复**：在 `VelocityPreservingTokenDecoder` 中，把 `_build_codebook()` 改为调用 `_build_output_codebook()`：

```python
# 当前（错误）：
self.codebook = self.tokenizer.get_input_embeddings().weight  # 输入空间

# 修复：
# 把所有 token 过一遍 T5 encoder，用输出 hidden states 作为 codebook
self.codebook = self._build_output_codebook()  # 输出空间
```

不过这需要对每个 token 做一次 forward pass 来建 codebook（vocab_size=32000），计算量较大（但只需做一次可缓存）。鉴于 Layer 1 的理论价值有限（Δe 的信息无法用离散 token 完整表达），建议优先级放在 Layer 3 修复之后。

---

## 总结：需要改的文件清单

| 文件 | 改动 | 行数 |
|------|------|------|
| `src/pipeline.py` | 新增 `_prepare_blended_latents()` 方法 | +50行 |
| `src/pipeline.py` | 修改 L657-667 Layer 3 调用逻辑 | 改10行 |
| `src/pipeline.py` | VMADConfig 新增 `blend_alpha` 参数 | +1行 |
| `run_batch_extract.py` | argparse 新增 `--blend-alpha` | +2行 |
| `src/motion_asset.py` | `apply_noise_prior` 标记 deprecated（可选） | 不改也行 |
