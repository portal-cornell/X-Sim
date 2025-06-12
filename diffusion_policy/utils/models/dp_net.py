from typing import Optional
from dataclasses import dataclass, field
import torch
from torch import nn

from .multiview_encoder import  ResNetEncoderConfig, MultiViewEncoder
from .cond_unet1d import ConditionalUnet1D


@dataclass
class MultiviewCondUnetConfig:
    resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
    feat_dim: int = 512
    use_prop: int = 1
    base_down_dims: int = 256
    kernel_size: int = 5
    diffusion_step_embed_dim: int = 128
    proto_dim: int = 0


class MultiviewCondUnet(nn.Module):
    def __init__(
        self,
        obs_shape,
        obs_horizon,
        prop_dim,
        cameras,
        action_dim,
        cfg: MultiviewCondUnetConfig,
    ):
        super().__init__()

        self.encoder = MultiViewEncoder(
            obs_shape,
            obs_horizon,
            cameras,
            prop_dim,
            cfg.use_prop,
            cfg.feat_dim,
            cfg.resnet,
        )
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.encoder.repr_dim + cfg.proto_dim,
            kernel_size=cfg.kernel_size,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=[cfg.base_down_dims, cfg.base_down_dims * 2, cfg.base_down_dims * 4],
        )

    def predict_noise(
        self,
        obs: dict[str, torch.Tensor],
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        cached_image_emb: Optional[torch.Tensor] = None,
        extra_conditioning = None,
    ):
        if cached_image_emb is None:
            image_emb = self.encoder(obs)
        else:
            image_emb = cached_image_emb

        global_cond = image_emb
        if extra_conditioning is not None:
            global_cond = torch.cat((global_cond, extra_conditioning), dim=1)

        noise_pred = self.noise_pred_net(noisy_action, timestep, global_cond=global_cond)
        return noise_pred, image_emb
