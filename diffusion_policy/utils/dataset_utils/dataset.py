from dataclasses import dataclass
from collections import defaultdict, namedtuple
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import zarr
import random

from ..common_utils import get_all_files
# from interactive_scripts.dataset_recorder import ActMode


class ObsProcessor:
    def __init__(self, camera_names: list[str], target_size: int):
        self.camera_names = camera_names
        self.target_size = target_size
        self.rescale_transform = transforms.Resize(
            (target_size, target_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,  # type: ignore
        )

    def process(self, obs: dict):
        processed_obs = {}
        for k, v in obs.items():
            if k == "proprio":
                processed_obs["prop"] = torch.from_numpy(v.astype(np.float32))

            if k not in self.camera_names:
                continue

            v = torch.from_numpy(v.copy())
            v = v.permute(2, 0, 1)
            v = self.rescale_transform(v)
            processed_obs[k] = v
        
        return processed_obs


Batch = namedtuple("Batch", ["obs", "action"])
PairedBatch = namedtuple("PairedBatch", ["real_obs", "sim_obs"])

@dataclass
class BCDatasetConfig:
    paths: list[str]
    validation_paths: list[str]
    camera_views: str = "wrist_view"
    image_size: int = 96
    real_pairing: str = ''
    sim_pairing: str = ''
    validation_real_pairing: str = '' 
    validation_sim_pairing: str = ''  
    robot_play_data: str = ''

@dataclass
class RHYMEDatasetConfig:
    paths: list[str]
    cond_path: str 
    proto_pred_path: str
    camera_views: str = "wrist_view"
    image_size: int = 96
    snap_frames: int = 75
    
    def __post_init__(self):
        pass


class BCDataset:
    def __init__(self, cfg: BCDatasetConfig, load_only_one=False):
        self.cfg = cfg
        self.load_only_one = load_only_one
        self.camera_views = cfg.camera_views.split("+")
        self.input_processor = ObsProcessor(self.camera_views, cfg.image_size)

        self.episodes: list[list[dict]] = []
        for path in cfg.paths:
            self.episodes.extend(self._load_and_process_episodes(path))
            # if 'real' in path:
            #     for _ in range(499):
            #         self.episodes.extend(self._load_and_process_episodes(path))
        self.validation_episodes: list[list[dict]] = []
        for path in cfg.validation_paths:
            self.validation_episodes.extend(self._load_and_process_episodes(path))


        self.paired_episodes = []
        if cfg.real_pairing and cfg.sim_pairing:
            self.paired_episodes = self._load_pairing(cfg.real_pairing, cfg.sim_pairing)
        self.validation_paired_episodes = []
        if cfg.validation_real_pairing and cfg.validation_sim_pairing:
            self.validation_paired_episodes = self._load_pairing(
                cfg.validation_real_pairing, cfg.validation_sim_pairing
            )

        self.idx2entry = {}
        for episode_idx, episode in tqdm(enumerate(self.episodes)):
            for step_idx in range(len(episode)):
                self.idx2entry[len(self.idx2entry)] = (episode_idx, step_idx)

        self.val_idx2entry = {}
        for episode_idx, episode in tqdm(enumerate(self.validation_episodes)):
            for step_idx in range(len(episode)):
                self.val_idx2entry[len(self.val_idx2entry)] = (episode_idx, step_idx)

        self.pairing_idx2entry = {}
        for episode_idx, episode in tqdm(enumerate(self.paired_episodes)):
            for step_idx in range(len(episode)):
                self.pairing_idx2entry[len(self.pairing_idx2entry)] = (episode_idx, step_idx)

        self.val_pairing_idx2entry = {}
        for episode_idx, episode in tqdm(enumerate(self.validation_paired_episodes)):
            for step_idx in range(len(episode)):
                self.val_pairing_idx2entry[len(self.val_pairing_idx2entry)] = (episode_idx, step_idx)

        print(f"Dataset loaded from {cfg.paths}")
        print(f"  episodes: {len(self.episodes)}")
        print(f"  steps: {len(self.idx2entry)}")
        print(f"  paired episodes: {len(self.paired_episodes)}")
        print(f"  paired steps: {len(self.pairing_idx2entry)}")
        print(f"  validation paired episodes: {len(self.validation_paired_episodes)}")
        print(f"  validation paired steps: {len(self.val_pairing_idx2entry)}")
        print(f"  avg episode len: {len(self.idx2entry) / len(self.episodes):.1f}")

    @property
    def action_dim(self) -> int:
        return self.episodes[0][0]["action"].size(0)

    @property
    def obs_shape(self) -> tuple[int]:
        return self.episodes[0][0][self.camera_views[0]].size()

    @property
    def prop_dim(self) -> int:
        return self.episodes[0][0]["prop"].size(0)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]

    def process_observation(self, obs):
        return self.input_processor.process(obs)
    
    def _load_pairing(self, real_pairing_path, sim_pairing_path):
        print(f"loading pairing data from {real_pairing_path} and {sim_pairing_path}")
        real_npz_files = list(sorted(get_all_files(real_pairing_path, "npz")))
        sim_npz_files = list(sorted(get_all_files(sim_pairing_path, "npz")))

        if self.load_only_one:
            real_npz_files = real_npz_files[:1]
            sim_npz_files = sim_npz_files[:1]

        paired_episodes = []

        for real_file, sim_file in zip(real_npz_files, sim_npz_files):
            real_raw_episode = np.load(real_file, allow_pickle=True)["episode"]
            sim_raw_episode = np.load(sim_file, allow_pickle=True)["episode"]

            combined_episode = []

            num_samples = min(100, len(sim_raw_episode))
            real_indices = np.linspace(0, len(sim_raw_episode) - 1, num_samples, dtype=int)
            # sim_indices = np.linspace(0, len(sim_raw_episode) - 1, num_samples, dtype=int)

            for real_idx in real_indices:
                real_timestep = real_raw_episode[real_idx]
                sim_timestep = sim_raw_episode[real_idx]

                real_processed_timestep = self.process_observation(real_timestep["obs"])
                sim_processed_timestep = self.process_observation(sim_timestep["obs"])

                combined_timestep = {
                    "real_obs": real_processed_timestep,
                    "sim_obs": sim_processed_timestep
                }
                combined_episode.append(combined_timestep)

            paired_episodes.append(combined_episode)

        return paired_episodes

    def _load_and_process_episodes(self, path):
        print(f"loading data from {path}")
        npz_files = list(sorted(get_all_files(path, "npz")))
        if self.load_only_one:
            npz_files = npz_files[:1]

        all_episodes: list[list[dict]] = []

        for episode_idx, f in enumerate(sorted(npz_files)):
            success_msg = ""
            raw_episode = np.load(f, allow_pickle=True)["episode"]
            episode = []

            for t, timestep in enumerate(raw_episode):
                # if timestep["mode"].value == ActMode.Waypoint.value:
                #     continue

                action = timestep["action"]
                if action[-1] < 0.5:
                    action[-1] = 0.0

                processed_timestep = {
                    "is_dense": False,
                    # "is_dense": torch.tensor(float(timestep["mode"] == ActMode.Dense)),
                    "action": torch.from_numpy(action).float(),
                }
                processed_timestep.update(self.process_observation(timestep["obs"]))
                
                episode.append(processed_timestep)

                if not success_msg and timestep.get("reward", 0) > 0:
                    success_msg = f", success since {len(episode)}"

            print(f"episode {episode_idx}, len: {len(episode)}" + success_msg)
            all_episodes.append(episode)

        return all_episodes

    def get_action_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        action_max = self.episodes[0][0]["action"] * 0
        action_min = self.episodes[0][0]["action"] * 0

        # Process training episodes
        for episode in self.episodes:
            for timestep in episode:
                action_max = torch.maximum(action_max, timestep["action"])
                action_min = torch.minimum(action_min, timestep["action"])
        
        # Process validation episodes
        for episode in self.validation_episodes:
            for timestep in episode:
                action_max = torch.maximum(action_max, timestep["action"])
                action_min = torch.minimum(action_min, timestep["action"])

        print(f"raw action value range, the model should do all the normalization:")
        for i in range(len(action_min)):
            print(
                f"  dim {i}, min: {action_min[i].item():.5f}, max: {action_max[i].item():.5f}"
            )
        
        return action_min, action_max

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.Tensor(np.array(v)).to(device)

        action = {"action": batch.pop("action")}
        ret = Batch(obs=batch, action=action)
        return ret
    
    def _convert_pairing_to_batch(self, samples, device):
        # iterate over every element in samples['real_obs'] and samples['sim_obs'], and cast the values associated with the keys in those dictionaries onto device
        for i in range(len(samples['real_obs'])):
            samples['real_obs'][i] = {k: v.to(device) for k, v in samples['real_obs'][i].items()}
            samples['sim_obs'][i] = {k: v.to(device) for k, v in samples['sim_obs'][i].items()}
        ret = PairedBatch(real_obs=samples['real_obs'], sim_obs=samples['sim_obs'])
        return ret

    def _stack_actions(self, idx, begin, action_len, validation=False):
        """stack actions in [begin, end)
        
        Args:
            idx: Index in the dataset
            begin: Starting timestep
            action_len: Number of actions to stack
            validation: Whether to use validation episodes
        """
        idx_map = self.val_idx2entry if validation else self.idx2entry
        episodes = self.validation_episodes if validation else self.episodes
        
        episode_idx, step_idx = idx_map[idx]
        episode = episodes[episode_idx]

        actions = []
        valid_actions = []
        for action_idx in range(begin, begin + action_len):
            if action_idx < 0:
                actions.append(torch.zeros_like(episode[step_idx]["action"]))
                valid_actions.append(0)
            elif action_idx < len(episode):
                actions.append(episode[action_idx]["action"])
                valid_actions.append(1)
            else:
                actions.append(torch.zeros_like(actions[-1]))
                valid_actions.append(0)

        valid_actions = torch.tensor(valid_actions, dtype=torch.float32)
        actions = torch.stack(actions, dim=0)
        return actions, valid_actions

    def sample_dp(self, batchsize, action_pred_horizon, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.idx2entry[idx]
            entry: dict = self.episodes[episode_idx][step_idx]

            actions, valid_actions = self._stack_actions(
                idx, step_idx, action_pred_horizon, validation=False
            )
            assert torch.equal(actions[0], entry["action"])

            samples["valid_action"].append(valid_actions)
            for k, v in entry.items():
                if k == "action":
                    samples[k].append(actions)
                else:
                    samples[k].append(v)
        
        return self._convert_to_batch(samples, device)
    
    def sample_pairing(self, batchsize, device):
        if len(self.pairing_idx2entry) == 0:
            return None
        indices = np.random.choice(len(self.pairing_idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.pairing_idx2entry[idx]
            entry: dict = self.paired_episodes[episode_idx][step_idx]

            for k, v in entry.items():
                samples[k].append(v)

        return self._convert_pairing_to_batch(samples, device)

    def sample_validation(self, batchsize, action_pred_horizon, device):
        if len(self.val_idx2entry) == 0:
            return None
            
        indices = np.random.choice(len(self.val_idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.val_idx2entry[idx]
            entry: dict = self.validation_episodes[episode_idx][step_idx]

            actions, valid_actions = self._stack_actions(
                idx, step_idx, action_pred_horizon, validation=True
            )
            assert torch.equal(actions[0], entry["action"])

            samples["valid_action"].append(valid_actions)
            for k, v in entry.items():
                if k == "action":
                    samples[k].append(actions)
                else:
                    samples[k].append(v)
        
        return self._convert_to_batch(samples, device)
    
    def sample_validation_pairing(self, batchsize, device):
        """Sample a batch of validation paired data."""
        if len(self.val_pairing_idx2entry) == 0:
            return None
            
        indices = np.random.choice(len(self.val_pairing_idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            episode_idx, step_idx = self.val_pairing_idx2entry[idx]
            entry: dict = self.validation_paired_episodes[episode_idx][step_idx]

            for k, v in entry.items():
                samples[k].append(v)

        return self._convert_pairing_to_batch(samples, device)

def visualize_episode(episode, image_size, camera):
    from ..common_utils import generate_grid, plot_images, RandomShiftsAug
    import os

    aug = RandomShiftsAug(pad=6)

    is_dense = []
    action_dims = [[] for _ in range(7)]

    for timestep in episode:
        action = timestep["action"]
        is_dense.append(timestep["is_dense"])
        for i, adim_val in enumerate(action):
            action_dims[i].append(adim_val.item())

    fig, axes = generate_grid(cols=7, rows=1)
    for idx, adim_vals in enumerate(action_dims):
        axes[idx].plot(adim_vals)
        axes[idx].set_title(f"action dim {idx}")

        for i, dense in enumerate(is_dense):
            if dense > 0:
                axes[idx].axvspan(i, i + 1, facecolor="green", alpha=0.3, label="dense")

        axes[idx].set_xlim(0, len(is_dense))

    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "actions.png"))

    images = [obs[camera] for obs in episode]
    images = images[::8]
    images = aug(torch.stack(images).float())
    images = [img.permute(1, 2, 0).numpy().astype(int) for img in images]
    fig = plot_images(images)
    path = os.path.join(os.path.dirname(__file__), "observations.png")
    print(f"saving image to {path}")
    fig.savefig(path)


def test():
    cfg = BCDatasetConfig(
        path="data/robot_pick_fork",
        camera_views="agent1_image+agent2_image",
        image_size=96,
    )
    dataset = BCDataset(cfg)
    dataset.get_action_range()
    visualize_episode(dataset.episodes[0], cfg.image_size, "agent1_image")


if __name__ == "__main__":
    test()
