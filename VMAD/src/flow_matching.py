"""
Flow Matching Inversion for VMAD.

将视频 latent z0 (t=0) 通过 ODE 正向积分到噪声 eta (t=1)。
支持一阶 Euler 和二阶 Midpoint 两种求解器。

参考:
    - RF-Inversion (ICLR 2025): Rectified Flow 反演的最优控制理论
    - RF-Solver (arXiv 2024): 高阶 Taylor 展开减少 ODE 求解误差
    - Flow Matching for Generative Modeling (Lipman et al., 2023)

数学:
    Rectified Flow ODE: dx_t/dt = v_theta(x_t, t, c)
    正向积分 (inversion): x_0 -> x_1, 即 z0 -> eta_inv
    反向采样 (generation): x_1 -> x_0, 即 eta -> z0

    Euler: x_{t+dt} = x_t + v_theta(x_t, t, c) * dt
    Midpoint: x_{t+dt} = x_t + v_theta(x_t + 0.5*dt*v_theta(x_t,t,c), t+0.5*dt, c) * dt
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def encode_video_to_latents(pipe, video_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    使用 VAE 将视频像素编码为 latent 表示。

    Args:
        pipe: diffusers pipeline (含 pipe.vae)
        video_tensor: (B, C, F, H, W) in [-1, 1]
        device: target device

    Returns:
        latents: (B, C_latent, F_latent, H_latent, W_latent)
    """
    vae = pipe.vae
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype)

    with torch.no_grad():
        if hasattr(vae, "encode"):
            # 标准 diffusers VAE
            # Wan2.1 VAE 期望 (B, C, F, H, W)
            latent_dist = vae.encode(video_tensor)
            if hasattr(latent_dist, "latent_dist"):
                latents = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, "sample"):
                latents = latent_dist.sample()
            else:
                latents = latent_dist
        else:
            raise ValueError("VAE does not have encode method")

    # 应用 scaling factor
    if hasattr(vae.config, "scaling_factor"):
        latents = latents * vae.config.scaling_factor

    return latents


class FlowMatchingInverter:
    """
    Flow Matching Inversion: 从 latent z0 反演到噪声 eta。

    通过正向积分 ODE (t: 0 -> 1)，使用模型预测的速度场 v_theta
    将干净 latent 推到噪声空间。

    参考 RF-Inversion 的实现:
        - 使用 null prompt (无条件) 进行反演，避免条件信息泄露
        - guidance_scale=1.0 (无 CFG)
        - 支持 Euler (1阶) 和 Midpoint (2阶) 求解器
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
            pipe: diffusers pipeline (含 pipe.transformer 或 pipe.unet)
            num_inversion_steps: ODE 积分步数 (越多越精确，越慢)
            guidance_scale: CFG scale (反演时通常为 1.0)
            device: 计算设备
        """
        self.pipe = pipe
        self.num_steps = num_inversion_steps
        self.guidance_scale = guidance_scale
        self.device = device

    def _get_model(self):
        """获取去噪模型 (transformer 或 unet)。"""
        if hasattr(self.pipe, "transformer"):
            return self.pipe.transformer
        elif hasattr(self.pipe, "unet"):
            return self.pipe.unet
        else:
            raise ValueError("Pipeline has neither transformer nor unet")

    @torch.no_grad()
    def _model_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        模型前向传播，预测速度场 v_theta(x_t, t, c)。

        适配 Wan2.1 DiT 的接口:
            - transformer(hidden_states, timestep, encoder_hidden_states)
        """
        model = self._get_model()

        # 确保 timestep 格式正确
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=latents.device, dtype=latents.dtype)

        # Wan2.1 DiT forward
        if hasattr(model, "config") and hasattr(model.config, "in_channels"):
            # 标准 diffusers transformer 接口
            model_output = model(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0]),
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            if isinstance(model_output, tuple):
                return model_output[0]
            return model_output
        else:
            # Fallback: 尝试直接调用
            return model(latents, timestep, prompt_embeds)

    def invert(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Euler 一阶反演: z0 (t=0) -> eta (t=1)

        ODE: dx_t/dt = v_theta(x_t, t, c_null)
        积分: x_0 = z0, x_1 = eta_inv

        Args:
            latents: z0, shape (B, C, F, H, W)
            prompt_embeds: null prompt embedding (用于无条件反演)
            negative_prompt_embeds: unused (保持接口兼容)

        Returns:
            eta_inv: 反演得到的噪声, shape (B, C, F, H, W)
        """
        dt = 1.0 / self.num_steps
        x = latents.clone()

        logger.info(f"  [Inversion/Euler] {self.num_steps} steps, dt={dt:.4f}")

        for i in range(self.num_steps):
            # t_norm in [0, 1] for interpolation, t_model in [0, 1000] for DiT
            t_norm = i * dt
            t_model = torch.tensor(t_norm * 1000.0, device=self.device)

            # 预测速度场 (无条件)
            v = self._model_forward(x, t_model, prompt_embeds)

            # Euler 步进: x_{t+dt} = x_t + v * dt
            x = x + v * dt

        return x

    def invert_midpoint(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Midpoint 二阶反演: z0 (t=0) -> eta (t=1)

        二阶精度，减少截断误差:
            k1 = v_theta(x_t, t, c)
            x_mid = x_t + 0.5 * dt * k1
            k2 = v_theta(x_mid, t + 0.5*dt, c)
            x_{t+dt} = x_t + dt * k2

        Args:
            latents: z0, shape (B, C, F, H, W)
            prompt_embeds: null prompt embedding
            negative_prompt_embeds: unused

        Returns:
            eta_inv: 反演得到的噪声
        """
        dt = 1.0 / self.num_steps
        x = latents.clone()

        logger.info(f"  [Inversion/Midpoint] {self.num_steps} steps, dt={dt:.4f}")

        for i in range(self.num_steps):
            # t_norm in [0, 1] for interpolation, t_model in [0, 1000] for DiT
            t_model = torch.tensor(i * dt * 1000.0, device=self.device)
            t_mid_model = torch.tensor((i + 0.5) * dt * 1000.0, device=self.device)

            # Stage 1: 计算中点
            k1 = self._model_forward(x, t_model, prompt_embeds)
            x_mid = x + 0.5 * dt * k1

            # Stage 2: 用中点速度更新
            k2 = self._model_forward(x_mid, t_mid_model, prompt_embeds)
            x = x + dt * k2

        return x
