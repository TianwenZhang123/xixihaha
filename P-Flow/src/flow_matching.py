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
from typing import Optional
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
    def invert_midpoint(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Midpoint method inversion for higher accuracy.

            k1 = v_θ(x_t, t, c)
            x_mid = x_t + (dt/2) * k1
            k2 = v_θ(x_mid, t + dt/2, c)
            x_{t+dt} = x_t + dt * k2
        """
        timesteps = torch.linspace(1.0, 0.0, self.num_inversion_steps + 1, device=self.device)
        dt = -1.0 / self.num_inversion_steps

        x_t = video_latents.clone()

        for i in tqdm(range(self.num_inversion_steps), desc="Inversion (Midpoint)", leave=False):
            t = timesteps[i]
            t_tensor = torch.full(
                (x_t.shape[0],), t.item(), device=self.device, dtype=x_t.dtype
            )
            t_mid_tensor = torch.full(
                (x_t.shape[0],), (t + dt / 2).item(), device=self.device, dtype=x_t.dtype
            )

            k1 = self._predict_velocity(x_t, t_tensor, prompt_embeds, negative_prompt_embeds)
            x_mid = x_t + (dt / 2) * k1
            k2 = self._predict_velocity(x_mid, t_mid_tensor, prompt_embeds, negative_prompt_embeds)
            x_t = x_t + dt * k2

        return x_t

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


def decode_latents_to_video(
    pipe,
    latents: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Decode latents to video space using VAE decoder.

    Args:
        pipe: Pipeline with VAE.
        latents: Latent tensor.
        device: Target device.

    Returns:
        Video tensor (B, C, F, H, W) in [0, 1].
    """
    with torch.no_grad():
        latents = latents.to(device=device, dtype=pipe.vae.dtype)

        scaling_factor = getattr(pipe.vae.config, "scaling_factor", None)
        if scaling_factor is None:
            scaling_factor = getattr(pipe, "vae_scaling_factor", 0.18215)
        latents = latents / scaling_factor

        video = pipe.vae.decode(latents).sample
        video = (video + 1.0) / 2.0
        video = video.clamp(0, 1)

    return video
