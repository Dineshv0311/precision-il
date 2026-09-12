"""
src/models/policies.py
Implements three policy architectures conditioned on VisuoForceFusionEncoder:
1. BCPolicy: MLP baseline predicting 1-step action.
2. ACTPolicy: CVAE + Transformer Decoder predicting action chunks (K=16).
3. DiffusionPolicy: 1D Temporal Denoiser with DDPM noise scheduler.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vision_encoder import VisuoForceFusionEncoder


# ==============================================================================
# 1. Behavioral Cloning (BC) MLP Baseline
# ==============================================================================
class BCPolicy(nn.Module):
    def __init__(self, action_dim=7, fused_feature_dim=128):
        super().__init__()
        self.encoder = VisuoForceFusionEncoder(fused_feature_dim=fused_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bounds actions strictly to [-1, 1]
        )

    def forward(self, image, proprio):
        z = self.encoder(image, proprio)
        return self.head(z)

    def compute_loss(self, image, proprio, actions):
        pred_actions = self.forward(image, proprio)
        loss = F.mse_loss(pred_actions, actions)
        return loss, {"mse_loss": loss.item()}


# ==============================================================================
# 2. Action Chunking Transformer (ACT)
# ==============================================================================
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ACTPolicy(nn.Module):
    def __init__(self, action_dim=7, chunk_size=16, fused_feature_dim=128, latent_dim=32, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = VisuoForceFusionEncoder(fused_feature_dim=fused_feature_dim)

        # CVAE Encoder (Used during training to encode action trajectory)
        self.action_encoder = nn.Linear(action_dim, d_model)
        self.cvae_pos_embed = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.cvae_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        # Decoder & Query Generation
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cond_proj = nn.Linear(fused_feature_dim, d_model)
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.dec_pos_embed = SinusoidalPositionEncoding(d_model, max_len=chunk_size)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, action_dim),
            nn.Tanh()
        )

    def forward(self, image, proprio, actions=None):
        B = image.shape[0]
        z_fused = self.encoder(image, proprio)  # (B, 128)

        if self.training and actions is not None:
            # Training: Encode condition + action sequence into CVAE latent space
            act_tokens = self.action_encoder(actions)  # (B, chunk_size, d_model)
            cond_token = self.cond_proj(z_fused).unsqueeze(1)  # (B, 1, d_model)
            seq = torch.cat([cond_token, act_tokens], dim=1)
            seq = self.cvae_pos_embed(seq)
            latent_tokens = self.cvae_transformer(seq)
            cls_token = latent_tokens[:, 0]
            mu = self.mu_head(cls_token)
            logvar = self.logvar_head(cls_token)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_latent = mu + eps * std
        else:
            # Inference: Sample prior z ~ N(0, I)
            z_latent = torch.zeros(B, self.latent_dim, device=image.device)
            mu, logvar = None, None

        # Decode action chunk
        latent_info = self.latent_proj(z_latent).unsqueeze(1)
        cond_info = self.cond_proj(z_fused).unsqueeze(1)
        memory = torch.cat([cond_info, latent_info], dim=1)  # (B, 2, d_model)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        queries = self.dec_pos_embed(queries)
        dec_out = self.transformer_decoder(tgt=queries, memory=memory)  # (B, chunk_size, d_model)
        pred_actions = self.action_head(dec_out)  # (B, chunk_size, action_dim)

        return pred_actions, mu, logvar

    def compute_loss(self, image, proprio, actions, kl_weight=10.0):
        pred_actions, mu, logvar = self.forward(image, proprio, actions)
        l1_loss = F.l1_loss(pred_actions, actions)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        total_loss = l1_loss + kl_weight * kl_loss
        return total_loss, {"l1_loss": l1_loss.item(), "kl_loss": kl_loss.item(), "total_loss": total_loss.item()}


# ==============================================================================
# 3. Diffusion Policy (DDPM with 1D Temporal ResNet Denoiser)
# ==============================================================================
class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=7, chunk_size=16, fused_feature_dim=128, num_diffusion_steps=20):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_steps = num_diffusion_steps
        self.encoder = VisuoForceFusionEncoder(fused_feature_dim=fused_feature_dim)

        # Sinusoidal time embedding
        self.time_embed_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128)
        )

        # 1D Temporal Denoising Network
        in_dim = action_dim + (128 + 128) // chunk_size  # Distributed condition representation
        self.net = nn.Sequential(
            nn.Conv1d(action_dim, 128, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(128, action_dim, kernel_size=3, padding=1)
        )
        self.cond_proj = nn.Linear(fused_feature_dim + 128, 128)
        self.cond_to_traj = nn.Linear(128, action_dim * chunk_size)

        # DDPM Linear Beta Schedule
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _get_time_embedding(self, timesteps):
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(embedding)

    def forward(self, noisy_actions, timesteps, condition):
        # noisy_actions: (B, chunk_size, action_dim) -> transpose to (B, action_dim, chunk_size)
        x = noisy_actions.transpose(1, 2)
        t_embed = self._get_time_embedding(timesteps)  # (B, 128)
        cond_full = torch.cat([condition, t_embed], dim=-1)
        cond_feature = self.cond_proj(cond_full)  # (B, 128)
        cond_delta = self.cond_to_traj(cond_feature).reshape(x.shape[0], self.action_dim, self.chunk_size)
        x = x + cond_delta
        noise_pred = self.net(x).transpose(1, 2)
        return noise_pred

    def compute_loss(self, image, proprio, actions):
        B = actions.shape[0]
        z_fused = self.encoder(image, proprio)
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=actions.device).long()
        noise = torch.randn_like(actions)
        alpha_hat = self.alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        noisy_actions = torch.sqrt(alpha_hat) * actions + torch.sqrt(1.0 - alpha_hat) * noise

        pred_noise = self.forward(noisy_actions, t, z_fused)
        loss = F.mse_loss(pred_noise, noise)
        return loss, {"diffusion_mse": loss.item()}

    @torch.no_grad()
    def sample(self, image, proprio):
        B = image.shape[0]
        z_fused = self.encoder(image, proprio)
        actions = torch.randn(B, self.chunk_size, self.action_dim, device=image.device)

        for step in reversed(range(self.num_steps)):
            t = torch.full((B,), step, device=image.device, dtype=torch.long)
            pred_noise = self.forward(actions, t, z_fused)
            alpha = self.alphas[step]
            alpha_hat = self.alphas_cumprod[step]
            beta = self.betas[step]
            if step > 0:
                noise = torch.randn_like(actions)
            else:
                noise = torch.zeros_like(actions)
            actions = (1.0 / torch.sqrt(alpha)) * (actions - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * pred_noise) + torch.sqrt(beta) * noise

        return torch.clamp(actions, -1.0, 1.0)