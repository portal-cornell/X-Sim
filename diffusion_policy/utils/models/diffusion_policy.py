from dataclasses import dataclass, field
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ..common_utils import *
from .dp_net import MultiviewCondUnet, MultiviewCondUnetConfig
from .action_normalizer import ActionNormalizer
import torch.nn.functional as F



@dataclass
class DDPMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: int = 1
    prediction_type: str = "epsilon"


@dataclass
class DDIMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: int = 1
    set_alpha_to_one: int = 1
    steps_offset: int = 0
    prediction_type: str = "epsilon"


@dataclass
class DiffusionPolicyConfig:
    # algo
    use_ddpm: int = 1
    ddpm: DDPMConfig = field(default_factory=lambda: DDPMConfig())
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())
    action_horizon: int = 8
    prediction_horizon: int = 16
    shift_pad: int = 4
    # arch
    cond_unet: MultiviewCondUnetConfig = field(default_factory=lambda: MultiviewCondUnetConfig())
    use_additional_aug: bool = False


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon,
        obs_shape,
        prop_dim: int,
        action_dim: int,
        camera_views,
        cfg: DiffusionPolicyConfig,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.obs_shape = obs_shape
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.camera_views = camera_views
        self.cfg = cfg

        # for data augmentation in training
        self.aug = RandomShiftsAug(pad=cfg.shift_pad)
        if cfg.use_additional_aug:
            self.aug = ImageAugmentation(random_translate=True, color_jitter=True, random_rotate=False, random_color_cutout=True, gaussian_blur=True)
        # for action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.action_min = nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
        self.action_max = nn.Parameter(torch.zeros(action_dim) + 1, requires_grad=False)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)

        # we concat image in dataset & env_wrapper
        if self.obs_horizon > 1:
            obs_shape = (obs_shape[0] // self.obs_horizon, obs_shape[1], obs_shape[2])

        self.net = MultiviewCondUnet(
            obs_shape,
            obs_horizon,
            prop_dim,
            camera_views,
            action_dim,
            cfg.cond_unet,
        )

        if cfg.use_ddpm:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.ddpm.num_train_timesteps,
                beta_schedule=cfg.ddpm.beta_schedule,
                clip_sample=bool(cfg.ddpm.clip_sample),
                prediction_type=cfg.ddpm.prediction_type,
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.ddim.num_train_timesteps,
                beta_schedule=cfg.ddim.beta_schedule,
                clip_sample=bool(cfg.ddim.clip_sample),
                set_alpha_to_one=bool(cfg.ddim.set_alpha_to_one),
                steps_offset=cfg.ddim.steps_offset,
                prediction_type=cfg.ddim.prediction_type,
            )
        self.to("cuda")

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

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, cpu=True, sim=False):
        assert not self.training

        unsqueezed = False
        if obs[self.camera_views[0]].dim() == 3:
            unsqueezed = True
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)

        bsize = obs[self.camera_views[0]].size(0)
        device = obs[self.camera_views[0]].device

        # pure noise input to begine with
        noisy_action = torch.randn(
            (bsize, self.cfg.prediction_horizon, self.action_dim), device=device
        )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_actions = []
        for k in self.noise_scheduler.timesteps:
            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_actions.append(noisy_action)

        action = noisy_action
        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        action = action[:, self.obs_horizon - 1 : self.cfg.action_horizon]
        
        if not sim:
            action = self.action_normalizer.denormalize(action)

        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    def loss(self, batch, avg=True, aug=True):
        obs = {}
        for k, v in batch.obs.items():
            if aug and (k in self.camera_views):
                obs[k] = self.aug(v.float())
            else:
                obs[k] = v

        # if needed, action has been transformed & normalized in dataset
        actions = batch.action["action"]
        actions = self.action_normalizer.normalize(actions)
        assert actions.min() >= -1.001 and actions.max() <= 1.001

        bsize = actions.size(0)
        noise = torch.randn(actions.shape, device=actions.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config["num_train_timesteps"],
            size=(bsize,),
            device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore

        noise_pred = self.net.predict_noise(obs, noisy_actions, timesteps)[0]
        # loss: [batch, num_action, action_dim]
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(2)

        assert "valid_action" in batch.obs
        valid_action = batch.obs["valid_action"]
        assert loss.size() == valid_action.size()
        loss = ((loss * valid_action).sum(1) / valid_action.sum(1))

        if avg:
            loss = loss.mean()
        return loss

@dataclass
class ConditionalDiffusionPolicyConfig:
    # algo
    use_ddpm: int = 1
    ddpm: DDPMConfig = field(default_factory=lambda: DDPMConfig())
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())
    action_horizon: int = 8
    prediction_horizon: int = 16
    shift_pad: int = 4
    # arch
    cond_unet: MultiviewCondUnetConfig = field(default_factory=lambda: MultiviewCondUnetConfig())


class ConditionalDiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon,
        obs_shape,
        prop_dim: int,
        action_dim: int,
        camera_views,
        cfg: ConditionalDiffusionPolicyConfig,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.obs_shape = obs_shape
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.camera_views = camera_views
        self.cfg = cfg

        # for data augmentation in training
        self.aug = RandomShiftsAug(pad=cfg.shift_pad)
        # for action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.action_min = nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
        self.action_max = nn.Parameter(torch.zeros(action_dim) + 1, requires_grad=False)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)

        # we concat image in dataset & env_wrapper
        if self.obs_horizon > 1:
            obs_shape = (obs_shape[0] // self.obs_horizon, obs_shape[1], obs_shape[2])

        self.net = MultiviewCondUnet(
            obs_shape,
            obs_horizon,
            prop_dim,
            camera_views,
            action_dim,
            cfg.cond_unet,
        )

        if cfg.use_ddpm:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.ddpm.num_train_timesteps,
                beta_schedule=cfg.ddpm.beta_schedule,
                clip_sample=bool(cfg.ddpm.clip_sample),
                prediction_type=cfg.ddpm.prediction_type,
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.ddim.num_train_timesteps,
                beta_schedule=cfg.ddim.beta_schedule,
                clip_sample=bool(cfg.ddim.clip_sample),
                set_alpha_to_one=bool(cfg.ddim.set_alpha_to_one),
                steps_offset=cfg.ddim.steps_offset,
                prediction_type=cfg.ddim.prediction_type,
            )
        self.to("cuda")

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

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, cpu=True):
        assert not self.training

        unsqueezed = False
        if obs[self.camera_views[0]].dim() == 3:
            unsqueezed = True
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)

        bsize = obs[self.camera_views[0]].size(0)
        device = obs[self.camera_views[0]].device

        # pure noise input to begine with
        noisy_action = torch.randn(
            (bsize, self.cfg.prediction_horizon, self.action_dim), device=device
        )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_actions = []
        for k in self.noise_scheduler.timesteps:
            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_actions.append(noisy_action)

        action = noisy_action
        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        action = action[:, self.obs_horizon - 1 : self.cfg.action_horizon]

        action = self.action_normalizer.denormalize(action)

        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    def loss(self, batch, avg=True, aug=True, return_image_emb=False):
        obs = {}
        for k, v in batch.obs.items():
            if aug and (k in self.camera_views):
                obs[k] = self.aug(v.float())
            else:
                obs[k] = v

        # if needed, action has been transformed & normalized in dataset
        actions = batch.action["action"]
        actions = self.action_normalizer.normalize(actions)
        assert actions.min() >= -1.001 and actions.max() <= 1.001

        bsize = actions.size(0)
        noise = torch.randn(actions.shape, device=actions.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config["num_train_timesteps"],
            size=(bsize,),
            device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore
        
        noise_pred, image_emb = self.net.predict_noise(obs, noisy_actions, timesteps, extra_conditioning=obs['proto'])
        # loss: [batch, num_action, action_dim]
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(2)

        assert "valid_action" in batch.obs
        valid_action = batch.obs["valid_action"]
        assert loss.size() == valid_action.size()
        loss = ((loss * valid_action).sum(1) / valid_action.sum(1))

        if avg:
            loss = loss.mean()
        if return_image_emb:
            return loss, image_emb
        return loss

@dataclass
class DiffusionPolicyWithAuxLossConfig(DiffusionPolicyConfig):
    use_aux_loss: bool = False
    aux_loss_weight: float = 1
    distance_type: str = 'l2'
    idm_loss_weight: float = 1
    use_idm_loss: bool = False

class ActionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)  # input_dim * 2 for current and next embeddings
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, curr_embeddings, next_embeddings):
        x = torch.cat((curr_embeddings, next_embeddings), dim=-1)  # Concatenate embeddings
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Predict action


class DiffusionPolicyWithAuxLoss(DiffusionPolicy):
    def __init__(
        self,
        obs_horizon,
        obs_shape,
        prop_dim: int,
        action_dim: int,
        camera_views,
        cfg: DiffusionPolicyWithAuxLossConfig,
    ):
        super().__init__(obs_horizon, obs_shape, prop_dim, action_dim, camera_views, cfg)
        self.use_aux_loss = cfg.use_aux_loss
        self.aux_loss_weight = cfg.aux_loss_weight
        self.distance_type = cfg.distance_type
        self.use_idm_loss = cfg.use_idm_loss
        self.idm_loss_weight = cfg.idm_loss_weight
        self.action_predictor = ActionPredictor(input_dim=self.net.encoder.repr_dim, output_dim=action_dim)  # Updated line


    def _compute_auxiliary_loss(self, real_obs_dict, sim_obs_dict):
        # Use the first image encoder from the MultiViewEncoder
        real_embeddings = []
        sim_embeddings = []
        
        for j, camera in enumerate(self.camera_views):
            real_embedding = self.net.encoder.encoders[j](real_obs_dict[camera], flatten=True)
            sim_embedding = self.net.encoder.encoders[j](sim_obs_dict[camera], flatten=True)
            real_embeddings.append(real_embedding)
            sim_embeddings.append(sim_embedding)

        # If there are multiple cameras, take the average of their embeddings
        real_embedding = torch.stack(real_embeddings).mean(dim=0) if len(real_embeddings) > 1 else real_embeddings[0]
        sim_embedding = torch.stack(sim_embeddings).mean(dim=0) if len(sim_embeddings) > 1 else sim_embeddings[0]
        
        if self.distance_type == 'cosine':
            loss = self._cosine_distance_loss(real_embedding, sim_embedding)
        elif self.distance_type == 'l2':
            loss = self._l2_distance_loss(real_embedding, sim_embedding)
        elif "contrastive" in self.distance_type:
            loss = self._contrastive_loss(real_embedding, sim_embedding, 
                                        "l2" if "l2" in self.distance_type else "cosine")
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return loss
    
    def _cosine_distance_loss(self, real_embeddings, sim_embeddings):
        cosine_similarity = F.cosine_similarity(real_embeddings, sim_embeddings, dim=-1)
        loss = 1 - cosine_similarity.mean()
        return loss

    def _l2_distance_loss(self, real_embeddings, sim_embeddings):
        loss = F.mse_loss(real_embeddings, sim_embeddings)
        return loss

    def _contrastive_loss(self, real_embeddings, sim_embeddings, distance_type='cosine'):
        """
        Implements InfoNCE contrastive loss.
        
        Args:
            real_embeddings: Tensor of shape (B, 512)
            sim_embeddings: Tensor of shape (B, 512)
            distance_type: String, either 'l2' or 'cosine'
            
        Returns:
            loss: Scalar tensor with the InfoNCE loss
        """
        import torch
        import torch.nn.functional as F
        batch_size = real_embeddings.shape[0]
        
        # For numerical stability
        temperature = 1
        
        if distance_type == 'cosine':
            # Normalize embeddings for cosine similarity
            real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
            sim_embeddings = F.normalize(sim_embeddings, p=2, dim=1)
            
            # Compute similarity matrix (B x B)
            # Each entry [i, j] represents similarity between real_i and sim_j
            similarity_matrix = torch.matmul(real_embeddings, sim_embeddings.T) / temperature
            
        elif distance_type == 'l2':
            # Calculate pairwise L2 distances
            # Expand dimensions to compute distances between all pairs
            r = real_embeddings.unsqueeze(1)  # (B, 1, 512)
            s = sim_embeddings.unsqueeze(0)   # (1, B, 512)
            
            # Compute squared L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
            l2_dist = torch.sum((r - s)**2, dim=2)  # (B, B)
            
            # Convert distances to similarities (negative distances divided by temperature)
            similarity_matrix = -l2_dist / temperature
        
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")
        
        # Create labels: diagonal elements (indices where i=j) are the positive pairs
        labels = torch.arange(batch_size, device=real_embeddings.device)
        
        # InfoNCE loss (equivalent to cross-entropy with positive pairs as targets)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

    def aux_loss(self, batch, avg=True, aug=True):
        real_obs_dict = {}
        sim_obs_dict = {}
        for real_obs, sim_obs in zip(batch.real_obs, batch.sim_obs):
            for k, v in real_obs.items():
                v = v.unsqueeze(0)
                if aug and (k in self.camera_views):
                    v = self.aug(v.float())
                if k in real_obs_dict:
                    real_obs_dict[k] = torch.cat((real_obs_dict[k], v), dim=0)
                else:
                    real_obs_dict[k] = v
            for k, v in sim_obs.items():
                v = v.unsqueeze(0)
                if aug and (k in self.camera_views):
                    v = self.aug(v.float())
                if k in sim_obs_dict:
                    sim_obs_dict[k] = torch.cat((sim_obs_dict[k], v), dim=0)
                else:
                    sim_obs_dict[k] = v

        aux_loss = self.aux_loss_weight * self._compute_auxiliary_loss(real_obs_dict, sim_obs_dict)
        
        if avg:
            aux_loss = aux_loss.mean()
        return aux_loss
    
    def _compute_idm_loss(self, curr_embeddings, next_embeddings, target_actions):
        predicted_action = self.action_predictor(curr_embeddings, next_embeddings)
        # Compute MSE loss between predicted actions and target actions
        loss = F.mse_loss(predicted_action, target_actions)
        return loss

    def idm_loss(self, batch, avg=True, aug=True):
        obs_dict = {}
        next_obs_dict = {}
        
        for obs, next_obs in zip(batch.obs, batch.next_obs):
            for k, v in obs.items():
                v = v.unsqueeze(0)
                if aug and (k in self.camera_views):
                    v = self.aug(v.float())
                if k in obs_dict:
                    obs_dict[k] = torch.cat((obs_dict[k], v), dim=0)
                else:
                    obs_dict[k] = v
            for k, v in next_obs.items():
                v = v.unsqueeze(0)
                if aug and (k in self.camera_views):
                    v = self.aug(v.float())
                if k in next_obs_dict:
                    next_obs_dict[k] = torch.cat((next_obs_dict[k], v), dim=0)
                else:
                    next_obs_dict[k] = v

        # Compute embeddings for both real and next observations
        curr_embeddings = self.net.encoder(obs_dict)
        next_embeddings = self.net.encoder(next_obs_dict)

        # Normalize actions from the batch
        target_actions = self.action_normalizer.normalize(batch.action["action"])

        # Compute IDM loss based on the embeddings
        idm_loss = self._compute_idm_loss(curr_embeddings, next_embeddings, target_actions)

        if not self.use_idm_loss:
            idm_loss *= 0
        if avg:
            idm_loss = idm_loss.mean()
        return idm_loss

    
    @torch.no_grad()
    def mse_action_loss(self, batch):
        obs = {}
        for k, v in batch.obs.items():
            obs[k] = v

        # Normalize actions from the batch
        actions = batch.action["action"]
        actions = self.action_normalizer.normalize(actions)

        bsize = actions.size(0)
        noisy_action = torch.randn(
            (bsize, self.cfg.prediction_horizon, self.action_dim), device=actions.device
        )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_actions = []
        for k in self.noise_scheduler.timesteps:
            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_actions.append(noisy_action)

        action = noisy_action[:, self.obs_horizon - 1 :]

        # Compute the MSE loss between predicted and true actions
        loss = nn.functional.mse_loss(action, actions, reduction='mean')
        # compute the average norm of the cached image emb
        avg_emb_norm = torch.norm(cached_image_emb, dim=-1).mean()
        return loss, avg_emb_norm


def test():
    obs_shape = (3, 96, 96)
    prop_dim = 9
    action_dim = 7
    camera_views = ["agentview"]
    cfg = DiffusionPolicyConfig()

    policy = DiffusionPolicy(1, obs_shape, prop_dim, action_dim, camera_views, cfg).cuda()
    policy.train(False)

    obs = {
        "agentview": torch.rand(1, 3, 96, 96).cuda(),
        "prop": torch.rand(1, 9).cuda(),
    }
    policy.act(obs)


if __name__ == "__main__":
    test()
