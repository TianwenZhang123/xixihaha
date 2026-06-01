# VMAD 实验 TODO

> 最后更新：2025-06-01

---

## P0 — 下一步立刻做

### 多样本验证（10 样本）
- [ ] V4 caption + L2 α=0.005 → 10 样本 extract + apply + eval
- [ ] V4 caption + L2 α=0.008 → 10 样本 apply + eval（复用 assets）
- [ ] 对照组：V4 caption 三层全关 → 10 样本（确认 Δe 增量在多样本上稳定）

**命令**：
```bash
cd /root/autodl-tmp/videofake/VMAD

# 需要重新 extract（V4 caption 的 Δe 需要重新优化）
python run_batch_extract.py \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir /root/autodl-tmp/outputs/vmad_v4_10samples \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.005 --num_opt_steps 200 \
    --no-svd --no-disentangle --no-blend --no-token_decode \
    --content SELF --resume -v

# Apply α=0.008（复用 assets，只改 alpha）
mkdir -p /root/autodl-tmp/outputs/vmad_v4_10samples_a008
ln -s /root/autodl-tmp/outputs/vmad_v4_10samples/assets \
      /root/autodl-tmp/outputs/vmad_v4_10samples_a008/assets

python run_batch_extract.py --apply-only \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir /root/autodl-tmp/outputs/vmad_v4_10samples_a008 \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.008 --no-blend --no-token_decode \
    --content SELF --seed 42 -v

# 对照组（三层全关）
python run_batch_extract.py --apply-only \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir /root/autodl-tmp/outputs/vmad_v4_10samples_ctrl \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.005 --no-blend --no-token_decode --no-velocity \
    --content SELF --seed 42 -v
```

**判断门控**：如果 10 样本均值 CLIP > P-Flow V4 iter1 (0.8842) 且 XCLIP > 0.7430，则 Layer 2 验证通过。

---

## P1 — 验证通过后做

### Cross-Content 迁移测试
- [ ] 用 T_m=0.5 提取 Δe（只捕获运动，不含内容结构）
- [ ] 应用到不同主体（"a white cat", "a robot", "a teddy bear"）
- [ ] 评测 motion fidelity（XCLIP）vs 内容多样性（CLIP with target prompt）

### 修复 Layer 3（独立 generator）
- [ ] 方案：让 pipeline 正常 `prepare_latents`，然后外部 blend eta_motion
- [ ] 验证修复后 Layer 3 是否能提供正向增量
- [ ] 如果有效，解耦 cfg.alpha（L2 和 L3 用独立参数）

---

## P2 — 可选优化

- [ ] 步数扫描（50/100/200/500），确认 200 步是否足够
- [ ] Position-Aware vs Uniform 消融
- [ ] T_m 时间步上界扫描（0.3/0.5/0.7/1.0）
- [ ] Token Decoder 修复（改用 output codebook）
- [ ] α ∈ [0.003, 0.006] 更精细扫描
- [ ] 扩大到 50+ 样本

---

## 通用评测命令

```bash
python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir <OUTPUT_DIR>/generated \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir <OUTPUT_DIR>/eval
```

---

## 通用参数速查

```
--no-velocity       关闭 Layer 2 (Δe)
--no-blend          关闭 Layer 3 (noise prior)
--no-token_decode   关闭 Layer 1 (motion text)
--alpha             Δe 注入强度（推荐 0.005~0.008）
--num_opt_steps     VFM 优化步数（默认 200）
--apply-only        只做 apply，复用已有 assets
--content SELF      用原始 caption 生成（复现模式）
```
