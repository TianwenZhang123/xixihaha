#!/usr/bin/env python3
"""
Dump Wan2.1 DiT Cross-Attention Weights.

分析 cross-attention 中各 text token position 的真实注意力分布，
验证 "position 0 是 attention sink" 假设是否成立。

原理：
    Wan2.1 默认使用 F.scaled_dot_product_attention (Flash Attention)，
    不暴露 attention weights。本脚本替换 cross-attention processor，
    用手动 softmax(QK^T/sqrt(d)) 计算并记录 attention weights。

输出：
    1. attention_pattern.pt  — 原始数据 (可后续分析)
    2. attention_analysis.json — 统计摘要
    3. attention_heatmap.png — 可视化热力图

用法：
    python scripts/dump_cross_attention.py \
        --video /root/autodl-tmp/data/video-200/water_mark_out/7.mp4 \
        --caption_file /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0/7.txt \
        --output_dir /root/autodl-tmp/outputs/attention_analysis \
        --model_path /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers \
        --num_samples 3

    # 多样本统计 (推荐)
    python scripts/dump_cross_attention.py \
        --data_dir /root/autodl-tmp/data/video-200/water_mark_out \
        --caption_dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
        --sample_ids 7 17 21 31 32 \
        --output_dir /root/autodl-tmp/outputs/attention_analysis \
        --model_path /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers

耗时约 3-5 分钟 (单样本，只跑 inversion 的几步来采集 attention)。
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 自定义 Attention Processor —— 记录 cross-attention weights
# ═══════════════════════════════════════════════════════════════

class AttentionRecorderProcessor:
    """
    替换 WanAttnProcessor，在 cross-attention 时手动计算 attention weights 并记录。
    Self-attention 仍用 SDPA (不记录，因为 token 数太大会 OOM)。
    """

    def __init__(self, record_store: dict, layer_name: str):
        self.record_store = record_store
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Handle image embeddings (I2V case)
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # QKV projections
        if hasattr(attn, 'fused_projections') and attn.fused_projections:
            if not attn.is_cross_attention and not is_cross:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        # RMSNorm on Q, K
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape to multi-head: (B, S, H*D) -> (B, S, H, D)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply rotary embeddings (self-attention only)
        if rotary_emb is not None and not is_cross:
            def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # ─── Cross-attention: 手动计算 attention weights ───
        if is_cross:
            # query: (B, Sv, H, D), key: (B, St, H, D)
            # 转为 (B, H, Sv, D) 和 (B, H, St, D)
            q = query.permute(0, 2, 1, 3)  # (B, H, Sv, D)
            k = key.permute(0, 2, 1, 3)    # (B, H, St, D)
            v = value.permute(0, 2, 1, 3)  # (B, H, St, D)

            scale = 1.0 / (q.shape[-1] ** 0.5)

            # 对大序列分块计算 attention weights 的统计量
            # 视频 token 数可能很大 (81*60*104=505440 for 1.3B)
            # 我们只采样部分 visual tokens 来计算统计
            sv = q.shape[2]  # visual token count
            st = k.shape[2]  # text token count

            # 采样策略：均匀采样 max_vis_tokens 个 visual tokens
            max_vis_tokens = min(sv, 2048)
            if sv > max_vis_tokens:
                indices = torch.linspace(0, sv - 1, max_vis_tokens).long().to(q.device)
                q_sampled = q[:, :, indices, :]
            else:
                q_sampled = q

            # 计算 attention scores: (B, H, sampled_Sv, St)
            attn_scores = torch.matmul(q_sampled, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, sampled_Sv, St)

            # 记录: 对 visual tokens 取平均 → (H, St) per-position attention
            # 即：每个 text position 平均被多少 visual tokens 关注
            mean_attn = attn_weights.mean(dim=2).mean(dim=0)  # (H, St) — 跨batch、visual tokens平均

            self.record_store.setdefault(self.layer_name, []).append(
                mean_attn.detach().cpu().float()  # (H, St)
            )

            # 用 SDPA 做实际计算 (保证数值一致性和效率)
            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.permute(0, 2, 1, 3)  # (B, Sv, H, D)

        else:
            # Self-attention: 直接用 SDPA，不记录 (太大)
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.permute(0, 2, 1, 3)

        # I2V image attention (if present)
        hidden_states_img = None
        if encoder_hidden_states_img is not None and attn.add_k_proj is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            if hasattr(attn, 'norm_added_k'):
                key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            q_for_img = query.permute(0, 2, 1, 3)
            k_img = key_img.permute(0, 2, 1, 3)
            v_img = value_img.permute(0, 2, 1, 3)
            hidden_states_img = F.scaled_dot_product_attention(
                q_for_img, k_img, v_img, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.permute(0, 2, 1, 3)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def install_recorders(transformer, record_store: dict):
    """替换所有 cross-attention 的 processor 为 recorder 版本。"""
    count = 0
    for name, module in transformer.named_modules():
        # WanTransformerBlock 中 attn2 = cross-attention
        if hasattr(module, 'processor') and hasattr(module, 'is_cross_attention'):
            if module.is_cross_attention:
                layer_name = name
                module.set_processor(AttentionRecorderProcessor(record_store, layer_name))
                count += 1
    logger.info(f"Installed attention recorders on {count} cross-attention layers")
    return count


def restore_processors(transformer):
    """恢复默认 processor。"""
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    for module in transformer.modules():
        if hasattr(module, 'processor') and hasattr(module, 'is_cross_attention'):
            if module.is_cross_attention:
                module.set_processor(WanAttnProcessor())


def run_forward_passes(pipe, video_path, caption, num_timesteps=5, device="cuda"):
    """
    执行若干次 forward pass 来采集 attention weights。
    模拟 inversion 过程的不同时间步。
    """
    from src.video_utils import load_video, normalize_video
    from src.flow_matching import encode_video_to_latents

    # 加载视频
    ref_video = load_video(video_path, num_frames=81, height=480, width=832, device=device)
    ref_norm = normalize_video(ref_video).unsqueeze(0)

    # 编码到 latent
    z0 = encode_video_to_latents(pipe, ref_norm, device)
    logger.info(f"Video latent shape: {z0.shape}")

    # 编码 caption
    import inspect
    if hasattr(pipe, "encode_prompt"):
        sig = inspect.signature(pipe.encode_prompt)
        params = sig.parameters
        kwargs = {"prompt": caption}
        if "device" in params:
            kwargs["device"] = device
        if "num_videos_per_prompt" in params:
            kwargs["num_videos_per_prompt"] = 1
        if "do_classifier_free_guidance" in params:
            kwargs["do_classifier_free_guidance"] = False
        if "max_sequence_length" in params:
            kwargs["max_sequence_length"] = 512
        result = pipe.encode_prompt(**kwargs)
        prompt_embeds = result[0] if isinstance(result, tuple) else result
    else:
        raise RuntimeError("Cannot encode prompt")

    logger.info(f"Prompt embedding shape: {prompt_embeds.shape}")  # 应为 (1, 512, dim)

    # 在不同时间步做 forward pass
    timesteps = torch.linspace(0.0, 1.0, num_timesteps, device=device)

    for i, t in enumerate(timesteps):
        logger.info(f"  Forward pass {i+1}/{num_timesteps}, t={t.item():.3f}")

        # 构造 x_t = (1-t)*noise + t*z0
        noise = torch.randn_like(z0)
        x_t = (1 - t) * noise + t * z0

        # 模型 forward
        t_tensor = t.unsqueeze(0) * 1000.0  # Wan 用 [0, 1000]
        with torch.no_grad():
            _ = pipe.transformer(
                hidden_states=x_t,
                timestep=t_tensor.expand(x_t.shape[0]),
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )

    return prompt_embeds.shape[1]  # text seq_len


def analyze_results(record_store: dict, text_seq_len: int, output_dir: Path):
    """分析并输出结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总所有层、所有时间步的 attention
    all_attentions = []  # list of (H, St) tensors
    per_layer_stats = {}

    for layer_name, attn_list in record_store.items():
        # attn_list: list of (H, St) tensors (每个时间步一个)
        stacked = torch.stack(attn_list, dim=0)  # (T, H, St)
        mean_per_layer = stacked.mean(dim=0)  # (H, St) — 该层跨时间步平均
        all_attentions.append(mean_per_layer)

        # 每层的 per-position 平均 (跨 head)
        pos_mean = mean_per_layer.mean(dim=0)  # (St,)
        per_layer_stats[layer_name] = {
            "pos_0_weight": pos_mean[0].item(),
            "pos_last_weight": pos_mean[-1].item() if len(pos_mean) > 1 else 0,
            "interior_mean": pos_mean[1:-1].mean().item() if len(pos_mean) > 2 else 0,
            "pos_0_ratio_vs_interior": (pos_mean[0] / pos_mean[1:-1].mean()).item() if len(pos_mean) > 2 else 0,
        }

    # 全局统计: (num_layers, H, St) → 平均
    global_attn = torch.stack(all_attentions, dim=0)  # (L, H, St)
    global_mean = global_attn.mean(dim=(0, 1))  # (St,) — 全局 per-position

    # 归一化为相对权重
    relative_weights = global_mean / global_mean.mean()

    # ─── 输出 1: 详细分析 JSON ───
    analysis = {
        "text_seq_len": text_seq_len,
        "num_layers": len(record_store),
        "num_heads": all_attentions[0].shape[0] if all_attentions else 0,
        "global_position_weights": {
            "pos_0": global_mean[0].item(),
            "pos_1": global_mean[1].item() if text_seq_len > 1 else 0,
            "pos_2": global_mean[2].item() if text_seq_len > 2 else 0,
            "pos_last": global_mean[-1].item(),
            "pos_second_last": global_mean[-2].item() if text_seq_len > 1 else 0,
            "interior_mean": global_mean[2:-2].mean().item() if text_seq_len > 4 else 0,
            "interior_std": global_mean[2:-2].std().item() if text_seq_len > 4 else 0,
        },
        "relative_weights (vs mean=1.0)": {
            "pos_0": relative_weights[0].item(),
            "pos_1": relative_weights[1].item() if text_seq_len > 1 else 0,
            "pos_last": relative_weights[-1].item(),
            "interior_mean": relative_weights[2:-2].mean().item() if text_seq_len > 4 else 0,
        },
        "attention_sink_test": {
            "is_pos0_sink": bool(relative_weights[0].item() > 3.0),
            "pos0_relative_weight": relative_weights[0].item(),
            "threshold_for_sink": 3.0,
            "verdict": (
                "YES - Position 0 IS an attention sink (>3x mean)"
                if relative_weights[0].item() > 3.0
                else "NO - Position 0 is NOT a significant attention sink"
            ),
        },
        "per_layer_stats": per_layer_stats,
    }

    # 找出真正的 "high attention" positions (> 2x mean)
    high_attn_positions = []
    for pos_idx in range(text_seq_len):
        if relative_weights[pos_idx].item() > 2.0:
            high_attn_positions.append({
                "position": pos_idx,
                "relative_weight": relative_weights[pos_idx].item(),
            })
    analysis["high_attention_positions"] = high_attn_positions

    # 建议的 position weights (基于真实数据)
    suggested_weights = (1.0 / (relative_weights + 0.1)).tolist()
    # 截断显示前20和后5
    analysis["suggested_position_weights_for_velocity_matching"] = {
        "first_20": suggested_weights[:20],
        "last_5": suggested_weights[-5:],
        "description": "Inverse of relative attention weight — high-attention positions get low regularization",
    }

    with open(output_dir / "attention_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"Analysis saved to {output_dir / 'attention_analysis.json'}")

    # ─── 输出 2: 原始数据 ───
    torch.save({
        "global_attn": global_attn,  # (L, H, St)
        "global_mean": global_mean,  # (St,)
        "relative_weights": relative_weights,  # (St,)
        "per_layer": {k: torch.stack(v) for k, v in record_store.items()},
    }, output_dir / "attention_pattern.pt")
    logger.info(f"Raw data saved to {output_dir / 'attention_pattern.pt'}")

    # ─── 输出 3: 可视化 ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Plot 1: Global per-position attention (前 100 个 position)
        show_len = min(text_seq_len, 100)
        ax = axes[0]
        ax.bar(range(show_len), global_mean[:show_len].numpy(), color="steelblue", alpha=0.7)
        ax.axhline(y=global_mean.mean().item(), color="red", linestyle="--", label=f"mean={global_mean.mean():.4f}")
        ax.set_xlabel("Text Token Position")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_title("Wan2.1 DiT Cross-Attention: Per-Position Mean Weight (first 100 positions)")
        ax.legend()

        # Plot 2: Relative weights (log scale)
        ax = axes[1]
        ax.semilogy(range(show_len), relative_weights[:show_len].numpy(), "b-", linewidth=1.5)
        ax.axhline(y=1.0, color="red", linestyle="--", label="mean=1.0")
        ax.axhline(y=3.0, color="orange", linestyle="--", label="sink threshold (3x)")
        ax.set_xlabel("Text Token Position")
        ax.set_ylabel("Relative Weight (log scale)")
        ax.set_title("Relative Attention Weight vs Mean (is pos 0 a sink?)")
        ax.legend()

        # Plot 3: Per-layer heatmap (layers × positions, first 50 pos)
        ax = axes[2]
        show_pos = min(text_seq_len, 50)
        heatmap_data = global_attn[:, :, :show_pos].mean(dim=1).numpy()  # (L, show_pos)
        im = ax.imshow(heatmap_data, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_xlabel("Text Token Position")
        ax.set_ylabel("Transformer Layer")
        ax.set_title(f"Cross-Attention Heatmap: Layer × Position (first {show_pos} positions)")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output_dir / "attention_heatmap.png", dpi=150, bbox_inches="tight")
        logger.info(f"Heatmap saved to {output_dir / 'attention_heatmap.png'}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")

    # ─── 打印关键结论 ───
    print("\n" + "=" * 70)
    print("  ATTENTION PATTERN ANALYSIS RESULTS")
    print("=" * 70)
    print(f"  Text sequence length: {text_seq_len}")
    print(f"  Number of layers: {len(record_store)}")
    print(f"  Position 0 relative weight: {relative_weights[0].item():.4f}x mean")
    print(f"  Position 1 relative weight: {relative_weights[1].item():.4f}x mean")
    print(f"  Last position relative weight: {relative_weights[-1].item():.4f}x mean")
    print(f"  Interior mean relative weight: {relative_weights[2:-2].mean().item():.4f}x mean")
    print()
    if relative_weights[0].item() > 3.0:
        print("  ✅ CONFIRMED: Position 0 IS an attention sink")
        print(f"     (weight = {relative_weights[0].item():.1f}x mean, threshold = 3x)")
    else:
        print("  ❌ REJECTED: Position 0 is NOT an attention sink in Wan2.1 DiT")
        print(f"     (weight = {relative_weights[0].item():.2f}x mean, need >3x)")
        print("     → The position-aware gradient scaling assumption is INVALID")
    print()
    if high_attn_positions:
        print("  High-attention positions (>2x mean):")
        for p in high_attn_positions[:10]:
            print(f"    pos {p['position']}: {p['relative_weight']:.2f}x")
    else:
        print("  No positions exceed 2x mean — attention is relatively UNIFORM")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dump Wan2.1 cross-attention weights")
    parser.add_argument("--video", type=str, help="单个视频路径")
    parser.add_argument("--caption_file", type=str, help="单个 caption 文件")
    parser.add_argument("--data_dir", type=str, help="批量: 视频目录")
    parser.add_argument("--caption_dir", type=str, help="批量: caption 目录")
    parser.add_argument("--sample_ids", type=int, nargs="+", default=[7], help="样本 ID")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/outputs/attention_analysis")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--num_timesteps", type=int, default=5, help="每个样本采集的时间步数")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    logger.info(f"Loading model from {args.model_path}...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.distributed import load_model_single_gpu
    pipe = load_model_single_gpu(args.model_path, dtype=torch.bfloat16, model_type="t2v")
    device = "cuda"

    # 安装 attention recorder
    record_store = {}
    num_layers = install_recorders(pipe.transformer, record_store)

    if num_layers == 0:
        logger.error("No cross-attention layers found! Trying alternative detection...")
        # Fallback: 遍历所有 WanAttention 模块
        from diffusers.models.transformers.transformer_wan import WanAttention
        count = 0
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, WanAttention) and "attn2" in name:
                module.set_processor(AttentionRecorderProcessor(record_store, name))
                count += 1
        logger.info(f"Fallback: installed on {count} attn2 modules")

    # 处理样本
    text_seq_len = 0

    if args.video and args.caption_file:
        # 单样本模式
        caption = Path(args.caption_file).read_text(encoding="utf-8").strip()
        logger.info(f"Processing single video: {args.video}")
        text_seq_len = run_forward_passes(pipe, args.video, caption, args.num_timesteps, device)

    elif args.data_dir and args.caption_dir:
        # 批量模式
        data_path = Path(args.data_dir)
        caption_path = Path(args.caption_dir)
        for sid in args.sample_ids:
            video_file = data_path / f"{sid}.mp4"
            caption_file = caption_path / f"{sid}.txt"
            if not video_file.exists():
                logger.warning(f"Video not found: {video_file}, skipping")
                continue
            if not caption_file.exists():
                logger.warning(f"Caption not found: {caption_file}, skipping")
                continue
            caption = caption_file.read_text(encoding="utf-8").strip()
            logger.info(f"Processing sample {sid}...")
            text_seq_len = run_forward_passes(pipe, str(video_file), caption, args.num_timesteps, device)
    else:
        logger.error("Please provide --video/--caption_file or --data_dir/--caption_dir")
        sys.exit(1)

    # 分析结果
    if not record_store:
        logger.error("No attention data recorded! Check model architecture.")
        sys.exit(1)

    analyze_results(record_store, text_seq_len, output_dir)

    # 恢复原始 processor
    restore_processors(pipe.transformer)
    logger.info("Done!")


if __name__ == "__main__":
    main()
