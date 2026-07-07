"""
Flow Matching Inversion for P-Flow (Wan2.1-1.3B).

Implements the flow matching inversion process:
    dx/dt = v_θ(x_t, t, c)
Integrating from t=1 (data) to t=0 (noise) via Euler method.

Adapted for Wan 2.1-1.3B single-GPU inference.

Reference: Section 3.2-3.3, Algorithm 1 line 2.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FlowMatchingInverter:
    """
    Flow matching inversion: x_1 (video latents) → x_0 (noise η_inv).

    The flow matching model defines:
        x_t = (1 - t) * ε + t * x_1     (Eq. 1)
        v_θ(x_t, t) ≈ x_1 - ε           (Eq. 2)

    Inversion integrates backward from t=1 to t=0 via Euler ODE solver.
    """

    def __init__(
        self,
        pipe,
        num_inversion_steps: int = 50,
        guidance_scale: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            pipe: Wan 2.1-1.3B pipeline.
            num_inversion_steps: Number of ODE steps.
            guidance_scale: Guidance during inversion (1.0 = no guidance).
            device: Primary device for operations.
        """
        self.pipe = pipe
        self.num_inversion_steps = num_inversion_steps
        self.guidance_scale = guidance_scale
        self.device = device

    @torch.no_grad()
    def invert(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform flow matching inversion: x_1 → x_0.

        Euler integration from t=1 to t=0:
            x_{t-dt} = x_t - dt * v_θ(x_t, t, c)

        Args:
            video_latents: Encoded video latents (B, C, F, H, W).
            prompt_embeds: Text embeddings (P_0) for conditioning.
            negative_prompt_embeds: Negative embeddings (for CFG if scale > 1).

        Returns:
            Inverted noise η_inv (B, C, F, H, W).
        """
        # Timestep schedule: t=1 → t=0 (linear)
        timesteps = torch.linspace(1.0, 0.0, self.num_inversion_steps + 1, device=self.device)
        dt = -1.0 / self.num_inversion_steps

        x_t = video_latents.clone()

        for i in tqdm(range(self.num_inversion_steps), desc="Flow Matching Inversion", leave=False):
            t = timesteps[i]
            t_tensor = torch.full(
                (x_t.shape[0],), t.item(), device=self.device, dtype=x_t.dtype
            )

            # Predict velocity v_θ(x_t, t, c)
            velocity = self._predict_velocity(
                x_t, t_tensor, prompt_embeds, negative_prompt_embeds
            )

            # Euler step: x_{t+dt} = x_t + dt * v_θ
            x_t = x_t + dt * velocity

        return x_t

    @torch.no_grad()
    def invert_with_trajectory(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cache_every_n: int = 1,
        fi_config: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Perform flow matching inversion with trajectory caching: x_1 → x_0.

        与 invert() 相同的 Euler ODE，但额外缓存中间状态，
        用于灰盒 Latent Trajectory Soft Anchor。

        Args:
            video_latents: Encoded video latents (B, C, F, H, W).
            prompt_embeds: Text embeddings for conditioning.
            negative_prompt_embeds: Negative embeddings (for CFG if scale > 1).
            cache_every_n: 每隔 n 步缓存一次 (1=全部缓存, 用于显存优化).
            fi_config: Feature Injection 配置 dict，包含:
                - target_layers: List[int], 注入的 DiT 层号
                - cache_mode: str, "attention"/"hidden"/"mlp"
                - gen_num_steps: int, 生成步数 (用于 t 映射)
                如果提供，反演时同时通过 hook 缓存 DiT 中间特征，
                省去事后重新前向的开销 (~77s → 0s)。

        Returns:
            (x_0, trajectory, fi_ref_features):
                x_0: Inverted noise η_inv (B, C, F, H, W).
                trajectory: dict {t_value(float): x_t_tensor(cpu)},
                    包含从 t=1.0 到 t=0.0 沿途的中间状态。
                fi_ref_features: dict {step_index: {layer_idx: tensor(cpu)}} 或 None,
                    Feature Injection 参考特征缓存。
        """
        fi_ref_features = None
        fi_hooks = []
        fi_captured = {}  # {layer_idx: tensor}

        if fi_config is not None:
            target_layers = fi_config["target_layers"]
            cache_mode = fi_config.get("cache_mode", "attention")
            gen_num_steps = fi_config["gen_num_steps"]

            # 注册 FI hook
            transformer = self.pipe.transformer
            blocks = transformer.blocks if hasattr(transformer, 'blocks') else []

            def make_fi_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        fi_captured[layer_idx] = output[0].detach().cpu()
                    else:
                        fi_captured[layer_idx] = output.detach().cpu()
                return hook_fn

            for layer_idx in target_layers:
                if layer_idx < len(blocks):
                    block = blocks[layer_idx]
                    if cache_mode == "attention" and hasattr(block, 'cross_attn'):
                        h = block.cross_attn.register_forward_hook(make_fi_hook(layer_idx))
                    elif cache_mode == "mlp" and hasattr(block, 'ffn'):
                        h = block.ffn.register_forward_hook(make_fi_hook(layer_idx))
                    else:
                        h = block.register_forward_hook(make_fi_hook(layer_idx))
                    fi_hooks.append(h)

            # 预计算 t 映射: 反演 step → 生成 step_index
            # 反演: step i 结束后 t_next = 1 - (i+1)/N_inv
            # 生成: step_index 完成后 t = 1 - (step_index+1)/N_gen
            # 映射: 对每个反演步，找最近的生成 step_index
            fi_ref_features = {}  # {gen_step_idx: {layer_idx: tensor}}
            # 构建反演 t 值 → 生成 step_index 的反向映射
            inv_t_to_gen_step = {}  # {t_inv_value: gen_step_index}
            for step_idx in range(gen_num_steps):
                t_gen_progress = 1.0 - (step_idx + 1) / gen_num_steps
                t_gen_traj = 1.0 - t_gen_progress  # = (step_idx + 1) / gen_num_steps
                inv_t_to_gen_step[round(t_gen_traj, 6)] = step_idx

            logger.info(
                f"  [FI Inline] 反演时同时缓存 FI 特征: "
                f"{len(target_layers)} layers, gen_steps={gen_num_steps}"
            )

        logger.info(
            f"  [Inversion+Trajectory] Starting: steps={self.num_inversion_steps}, "
            f"cache_every_n={cache_every_n}, "
            f"latent_shape={list(video_latents.shape)}, "
            f"estimated_cache_size={self.num_inversion_steps // cache_every_n + 1} points"
        )

        timesteps = torch.linspace(1.0, 0.0, self.num_inversion_steps + 1, device=self.device)
        dt = -1.0 / self.num_inversion_steps

        x_t = video_latents.clone()
        trajectory = {}

        # 缓存起点 t=1.0 (原始数据 latent)
        trajectory[1.0] = x_t.cpu().clone()

        try:
            for i in tqdm(range(self.num_inversion_steps), desc="Inversion + Trajectory", leave=False):
                t = timesteps[i]
                t_tensor = torch.full(
                    (x_t.shape[0],), t.item(), device=self.device, dtype=x_t.dtype
                )

                velocity = self._predict_velocity(
                    x_t, t_tensor, prompt_embeds, negative_prompt_embeds
                )
                x_t = x_t + dt * velocity

                # 按 cache_every_n 间隔缓存（存 CPU 节省显存）
                t_next = timesteps[i + 1].item()
                if (i + 1) % cache_every_n == 0 or i == self.num_inversion_steps - 1:
                    trajectory[t_next] = x_t.cpu().clone()

                # FI inline: 捕获当前步的 DiT 特征
                # 注意: _predict_velocity 内部的 _model_forward 已经触发了 hook
                # fi_captured 此时应已填充
                # 关键: forward 用的是 t=timesteps[i] (当前步的 x_t),
                #        不是 t_next (下一步的 x_t+dt)
                # 所以映射时也必须用 t (不是 t_next)
                if fi_config is not None and fi_captured:
                    t_current = t.item()  # 当前 forward 的 timestep
                    # 找最近的生成 step_index
                    # 生成 step_idx 对应的 t_gen_traj = (step_idx + 1) / gen_num_steps
                    # 反演 t_current 对应同一进度
                    nearest_gen_step = min(
                        inv_t_to_gen_step.keys(),
                        key=lambda t_key: abs(t_key - round(t_current, 6))
                    )
                    gen_step_idx = inv_t_to_gen_step[nearest_gen_step]

                    # 避免重复覆盖（如果多个反演步映射到同一生成步，取最近的）
                    if gen_step_idx not in fi_ref_features:
                        fi_ref_features[gen_step_idx] = {}
                        for layer_idx, feat in fi_captured.items():
                            fi_ref_features[gen_step_idx][layer_idx] = feat.clone().cpu()  # 存CPU省显存

                    fi_captured.clear()

                # 每 10 步打印一次进度
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"    [Inversion+Traj step {i+1}/{self.num_inversion_steps}] "
                        f"t={t_next:.3f}, x_t: mean={x_t.mean().item():.4f}, "
                        f"std={x_t.std().item():.4f}, cached={len(trajectory)} points"
                    )

        finally:
            # 移除 FI hook
            for h in fi_hooks:
                h.remove()

        logger.info(
            f"  [Inversion+Trajectory] Done: {len(trajectory)} points cached, "
            f"final x_0: mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}"
        )

        if fi_config is not None and fi_ref_features:
            # 保存元信息
            fi_ref_features["_meta"] = {
                "target_layers": fi_config["target_layers"],
                "num_layers": fi_config.get("num_layers", 30),
                "cache_mode": fi_config.get("cache_mode", "attention"),
                "num_steps": fi_config["gen_num_steps"],
            }
            total_cached = sum(
                len(v) for k, v in fi_ref_features.items() if k != "_meta"
            )
            logger.info(
                f"  [FI Inline] 特征缓存完成: "
                f"{len([k for k in fi_ref_features if k != '_meta'])} steps × "
                f"{len(fi_config['target_layers'])} layers = {total_cached} tensors"
            )

        return x_t, trajectory, fi_ref_features


    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity v_θ(x_t, t, c) with optional CFG.

        For inversion, guidance_scale=1.0 (no guidance) is standard.
        """
        if self.guidance_scale > 1.0 and negative_prompt_embeds is not None:
            latent_input = torch.cat([x_t, x_t], dim=0)
            t_input = torch.cat([t, t], dim=0)
            embed_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            velocity_pred = self._model_forward(latent_input, t_input, embed_input)
            v_uncond, v_cond = velocity_pred.chunk(2, dim=0)
            velocity = v_uncond + self.guidance_scale * (v_cond - v_uncond)
        else:
            velocity = self._model_forward(x_t, t, prompt_embeds)

        return velocity

    def _model_forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Wan 2.1-14B transformer.

        The model is distributed across multiple GPUs via device_map.
        Input tensors are automatically moved to the correct device.
        """
        # The transformer handles device placement internally when using device_map
        model_output = self.pipe.transformer(
            hidden_states=x_t,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        return model_output


def encode_video_to_latents(
    pipe,
    video_tensor: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode video tensor to latent space using VAE encoder.

    Args:
        pipe: Pipeline with VAE.
        video_tensor: Video (B, C, F, H, W) in [-1, 1].
        device: Target device.

    Returns:
        Latent tensor (B, C_latent, F_latent, H_latent, W_latent).
    """
    with torch.no_grad():
        video_tensor = video_tensor.to(device=device, dtype=pipe.vae.dtype)

        # Wan 2.1 VAE may process frames in chunks for memory
        latents = pipe.vae.encode(video_tensor).latent_dist.sample()

        # Apply VAE scaling factor
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", None)
        if scaling_factor is None:
            scaling_factor = getattr(pipe, "vae_scaling_factor", 0.18215)
        latents = latents * scaling_factor

    return latents



