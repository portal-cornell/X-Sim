from typing import Optional
from dataclasses import dataclass
import torch
from torch import nn

from .resnet import ResNet


@dataclass
class ResNetEncoderConfig:
    stem: str = "default"
    downsample: str = "default"
    norm_layer: str = "gnn"


class ResNetEncoder(nn.Module):
    def __init__(self, obs_shape, cfg: ResNetEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        layers = [2, 2, 2, 2]
        self.nets = ResNet(
            stem=self.cfg.stem,
            downsample=self.cfg.downsample,
            norm_layer=self.cfg.norm_layer,
            layers=layers,
            in_channels=obs_shape[0],
        )
        self.repr_dim, self.num_patch, self.patch_repr_dim = self._get_repr_dim(obs_shape)

    def _get_repr_dim(self, obs_shape: list[int]):
        x = torch.rand(1, *obs_shape)
        y = self.nets.forward(x).flatten(2, 3)
        repr_dim = y.flatten().size(0)
        _, patch_repr_dim, num_patch = y.size()
        return repr_dim, num_patch, patch_repr_dim

    def forward(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        h: torch.Tensor = self.nets(obs)
        if flatten:
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


class LinearCompress(nn.Module):
    def __init__(self, in_dim, out_dim, num_net, prop_dim, use_prop):
        super().__init__()
        nets = [
            nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
            )
            for _ in range(num_net)
        ]
        self.streams = nn.ModuleList(nets)
        self.prop_dim = prop_dim
        self.use_prop = use_prop
        self.out_dim = out_dim * num_net + int(use_prop) * prop_dim

    def forward(self, view_feats: list[torch.Tensor], prop: Optional[torch.Tensor]):
        outs = []
        assert len(view_feats) == len(self.streams)
        for i, feat in enumerate(view_feats):
            outs.append(self.streams[i](feat))

        if self.use_prop:
            assert prop is not None
            outs.append(prop)
        out_feat = torch.concat(outs, dim=1)  # dim 0 is the batch dim
        return out_feat


class MultiViewEncoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        obs_horizon,
        cameras,
        prop_dim,
        use_prop,
        feat_dim,
        resnet_cfg: ResNetEncoderConfig,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_horizon = obs_horizon
        self.cameras = cameras
        self.use_prop = use_prop

        self.encoders = nn.ModuleList([ResNetEncoder(obs_shape, resnet_cfg) for _ in cameras])
        self.compress_streams = LinearCompress(
            self.encoders[0].repr_dim, feat_dim, len(self.cameras), prop_dim, use_prop
        )
        self.repr_dim = self.compress_streams.out_dim

    def forward(self, obs: dict[str, torch.Tensor]):
        hs = []
        assert self.obs_horizon == 1
        for i, camera in enumerate(self.cameras):
            x = obs[camera]
            h = self.encoders[i](x, flatten=True)
            hs.append(h)

        prop = obs["prop"] if self.use_prop else None
        out = self.compress_streams(hs, prop)
        return out
