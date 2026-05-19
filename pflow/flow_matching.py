"""
Flow Matching Inversion for P-Flow.

This module implements the flow matching inversion process to obtain
the initial noise from a reference video. The inversion follows the ODE:
    dx/dt = v_θ(x_t, t, c)
by integrating backward from t=1 (data) to t=0 (noise).

Reference: Section 3.2 and 3.3 of the paper.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from tqdm import tqdm


class FlowMatchingInverter:
    """
    Performs flow matching inversion on a reference video to obtain
    the noise prior η_inv.
    
    The flow matching model defines:
        x_t = (1 - t) * ε + t * x_1     (Eq. 1)
        v_θ(x_t, t) ≈ x_1 - ε           (Eq. 2)
    
    Inversion: given x_1 (the video), integrate backward to get x_0 (noise).
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
            pipe: The video generation pipeline (Wan 2.1) with a transformer/UNet.
            num_inversion_steps: Number of ODE steps for inversion.
            guidance_scale: Guidance scale during inversion (typically 1.0 for inversion).
            device: Computation device.
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
        Perform flow matching inversion: x_1 (video latents) → x_0 (noise).
        
        The ODE for flow matching is:
            dx/dt = v_θ(x_t, t, c)
        
        For inversion, we integrate from t=1 to t=0:
            x_{t-dt} = x_t - dt * v_θ(x_t, t, c)
        
        Args:
            video_latents: Encoded video latents of shape (B, C, F, H, W).
            prompt_embeds: Text embeddings for conditioning.
            negative_prompt_embeds: Negative text embeddings (optional).
            
        Returns:
            Inverted noise tensor η_inv of shape (B, C, F, H, W).
        """
        # Create timestep schedule from t=1 to t=0 (inversion direction)
        timesteps = torch.linspace(1.0, 0.0, self.num_inversion_steps + 1, device=self.device)
        dt = -1.0 / self.num_inversion_steps  # Negative because going from 1→0
        
        # Start from the video latents (x_1)
        x_t = video_latents.clone()
        
        for i in tqdm(range(self.num_inversion_steps), desc="Flow Matching Inversion"):
            t = timesteps[i]
            t_tensor = torch.full((x_t.shape[0],), t.item(), device=self.device, dtype=x_t.dtype)
            
            # Predict velocity v_θ(x_t, t, c)
            velocity = self._predict_velocity(
                x_t, t_tensor, prompt_embeds, negative_prompt_embeds
            )
            
            # Euler step: x_{t+dt} = x_t + dt * v_θ(x_t, t, c)
            # Since dt is negative (going from 1 to 0), this moves toward noise
            x_t = x_t + dt * velocity
            
        return x_t  # This is η_inv (the inverted noise)
    
    @torch.no_grad()
    def invert_midpoint(
        self,
        video_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform flow matching inversion using midpoint method for better accuracy.
        
        Midpoint method:
            k1 = v_θ(x_t, t, c)
            x_mid = x_t + (dt/2) * k1
            k2 = v_θ(x_mid, t + dt/2, c)
            x_{t+dt} = x_t + dt * k2
            
        Args:
            video_latents: Encoded video latents (B, C, F, H, W).
            prompt_embeds: Text embeddings.
            negative_prompt_embeds: Negative text embeddings.
            
        Returns:
            Inverted noise tensor η_inv.
        """
        timesteps = torch.linspace(1.0, 0.0, self.num_inversion_steps + 1, device=self.device)
        dt = -1.0 / self.num_inversion_steps
        
        x_t = video_latents.clone()
        
        for i in tqdm(range(self.num_inversion_steps), desc="Flow Matching Inversion (Midpoint)"):
            t = timesteps[i]
            t_tensor = torch.full((x_t.shape[0],), t.item(), device=self.device, dtype=x_t.dtype)
            t_mid_tensor = torch.full((x_t.shape[0],), (t + dt / 2).item(), device=self.device, dtype=x_t.dtype)
            
            # First velocity estimate
            k1 = self._predict_velocity(x_t, t_tensor, prompt_embeds, negative_prompt_embeds)
            
            # Midpoint
            x_mid = x_t + (dt / 2) * k1
            
            # Second velocity estimate at midpoint
            k2 = self._predict_velocity(x_mid, t_mid_tensor, prompt_embeds, negative_prompt_embeds)
            
            # Update with midpoint velocity
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
        Predict velocity v_θ(x_t, t, c) using the model.
        
        If guidance_scale > 1, applies classifier-free guidance:
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            
        Args:
            x_t: Current noisy latents (B, C, F, H, W).
            t: Current timestep tensor (B,).
            prompt_embeds: Conditional text embeddings.
            negative_prompt_embeds: Unconditional text embeddings.
            
        Returns:
            Predicted velocity tensor.
        """
        # For Wan 2.1 model, we need to adapt the forward pass
        # The model expects specific input format
        
        if self.guidance_scale > 1.0 and negative_prompt_embeds is not None:
            # Classifier-free guidance: concatenate conditional and unconditional
            latent_input = torch.cat([x_t, x_t], dim=0)
            t_input = torch.cat([t, t], dim=0)
            embed_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            
            # Forward pass through the model
            velocity_pred = self._model_forward(latent_input, t_input, embed_input)
            
            # Split and apply guidance
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
        Forward pass through the video generation model.
        
        This wraps the Wan 2.1 transformer model's forward pass.
        The exact interface depends on the diffusers implementation.
        
        Args:
            x_t: Noisy latent input (B, C, F, H, W).
            t: Timestep (B,).
            encoder_hidden_states: Text condition embeddings.
            
        Returns:
            Model velocity prediction.
        """
        # Convert timestep to the format expected by the model
        # Wan 2.1 uses a specific timestep embedding format
        
        # The model's forward pass - adapt based on actual Wan 2.1 API
        # In diffusers, this is typically:
        #   model_output = self.pipe.transformer(
        #       hidden_states=x_t,
        #       timestep=t,
        #       encoder_hidden_states=encoder_hidden_states,
        #   ).sample
        
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
    Encode a video tensor to latent space using the VAE encoder.
    
    Args:
        pipe: The pipeline containing the VAE.
        video_tensor: Video tensor of shape (B, C, F, H, W) in [-1, 1].
        device: Computation device.
        
    Returns:
        Latent tensor of shape (B, C_latent, F_latent, H_latent, W_latent).
    """
    with torch.no_grad():
        # Wan 2.1 VAE expects (B, C, F, H, W)
        video_tensor = video_tensor.to(device=device, dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(video_tensor).latent_dist.sample()
        # Get scaling factor (Wan 2.1 VAE may not have it in config)
        scaling_factor = getattr(pipe.vae.config, 'scaling_factor', None)
        if scaling_factor is None:
            scaling_factor = getattr(pipe, 'vae_scaling_factor', 0.18215)
        latents = latents * scaling_factor
    return latents


def decode_latents_to_video(
    pipe,
    latents: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Decode latents back to video space using the VAE decoder.
    
    Args:
        pipe: The pipeline containing the VAE.
        latents: Latent tensor.
        device: Computation device.
        
    Returns:
        Video tensor of shape (B, C, F, H, W) in [0, 1].
    """
    with torch.no_grad():
        latents = latents.to(device=device, dtype=pipe.vae.dtype)
        # Get scaling factor (Wan 2.1 VAE may not have it in config)
        scaling_factor = getattr(pipe.vae.config, 'scaling_factor', None)
        if scaling_factor is None:
            scaling_factor = getattr(pipe, 'vae_scaling_factor', 0.18215)
        latents = latents / scaling_factor
        video = pipe.vae.decode(latents).sample
        video = (video + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        video = video.clamp(0, 1)
    return video
