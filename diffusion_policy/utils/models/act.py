import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataclasses import dataclass, field
from typing import Optional

import torch
from ..common_utils import *

from ...detr.models.detr_vae import build, ACTModelConfig
from ..dataset_utils.dataset import Batch
from .action_normalizer import ActionNormalizer


@dataclass
class ACTPolicyConfig:
    model: ACTModelConfig = field(default_factory=ACTModelConfig)
    kl_weight: int = 10
    action_horizon: int = 8


class ACTPolicy(nn.Module):
    def __init__(
        self,
        obs_shape,
        prop_shape,
        action_dim,
        camera_views,
        cfg: ACTPolicyConfig,
        stat
    ):
        super().__init__()
        model = build(obs_shape, prop_shape, action_dim, camera_views, cfg.model)
        self.cfg = cfg
        self.action_min = nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
        self.action_max = nn.Parameter(torch.zeros(action_dim) + 1, requires_grad=False)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)
        self.camera_views = camera_views
        self.model = model  # CVAE decoder
        self.stat = stat
        print(f"KL Weight {self.cfg.kl_weight}")

    def init_action_normalizer(
        self, action_min: torch.Tensor, action_max: torch.Tensor
    ):
        # for action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.action_min.data.copy_(action_min)
        self.action_max.data.copy_(action_max)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)
        self.action_normalizer.to(self.action_min.device)
        print("creating action normalizer with")
        print("  scale:", self.action_normalizer.scale.squeeze())
        print("  offset:", self.action_normalizer.offset.squeeze())

    def to(self, device):
        self.action_normalizer.to(device)
        return super().to(device)

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)
        print("creating action normalizer with")
        print("  scale:", self.action_normalizer.scale.squeeze())
        print("  offset:", self.action_normalizer.offset.squeeze())

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=True, cpu=True, ret_all=False):
        assert eval_mode
        assert not self.training

        unsqueezed = False
        if obs[self.camera_views[0]].dim() == 3:
            # add batch dim
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
            unsqueezed = True

        prop = obs["prop"]
        images = [obs[view] for view in self.camera_views]
        image = torch.stack(images, dim=1)
        # change to float torch and divide by 255
        image = image.float() / 255

        action = self(prop, image)
        assert isinstance(action, torch.Tensor)
        if not ret_all:
            action = action[:, : self.cfg.action_horizon]

        action = self.action_normalizer.denormalize(action)

        if unsqueezed:
            action = action.squeeze()
        if cpu:
            action = action.cpu()
        return action

    def loss(self, batch: Batch):
        obs = batch.obs
        actions = batch.action["action"]
        actions = self.action_normalizer.normalize(actions)
        is_pad = 1 - obs["valid_action"]
        prop = obs["prop"]
        images = [obs[view] for view in self.camera_views]
        image = torch.stack(images, dim=1)
        # change to float torch and divide by 255
        image = image.float() / 255

        loss_dict = self(prop, image, actions, is_pad)
        return loss_dict["loss"]

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            assert is_pad is not None, "is_pad must be provided"
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")  # [batch, num_queries, action_dim]
            # [batch, num_queries, action_dim] -> [batch]
            l1 = (all_l1 * (1 - is_pad).unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.cfg.kl_weight
            if self.stat is not None:
                self.stat["train/l1"].append(l1.item())
                self.stat["train/kl"].append(total_kld.item())
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def test():
    obs_shape = (3, 96, 96)
    prop_shape = (9,)
    action_dim = 7
    camera_views = ["agentview", "robot0_eye_in_hand"]

    cfg = ACTPolicyConfig()
    model = ACTPolicy(obs_shape, prop_shape, action_dim, camera_views, cfg, None).cuda()
    # make sure valid_action is boolean
    batch = Batch(
        obs={
            "agentview": torch.rand(1, 3, 96, 96).cuda(),
            "robot0_eye_in_hand": torch.rand(1, 3, 96, 96).cuda(),
            "prop": torch.rand(1, 9).cuda(),
            "valid_action": torch.ones(1, 100).cuda(),
        },
        action=torch.rand(1, 100, 7).cuda(),
    )
    loss = model.loss(batch)
    print(loss)

    obs = {
        "agentview": torch.rand(3, 96, 96).cuda(),
        "robot0_eye_in_hand": torch.rand(3, 96, 96).cuda(),
        "prop": torch.rand(9).cuda(),
    }
    model.eval()
    action = model.act(obs, eval_mode=True, cpu=True)
    print(action)


if __name__ == "__main__":
    test()
