from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import glob

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisodeReal
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import h5py 
import numpy as np
from tqdm import trange 
from scipy.spatial.transform import Rotation as R
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'diffusion_policy'))
from utils.common_utils.scaling_utils import *

import imageio
@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Sim2Real"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = True
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    num_trajectories: int = 500
    """number of trajectories to collect"""
    trajectory_length: int = 100
    """length of each trajectory in timesteps"""

    # Algorithm specific arguments
    env_id: str = "Kitchen-Subgoals-Eval"
    """the id of the environment"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    reward_mode: str = "dense"
    """reward type"""
    randomize_init_config: bool = True
    """whether or not to randomize initial configuration of objects"""
    obj_noise: float = 0.0
    """obj observation noise"""
    randomize_camera: bool = False

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def collect_single_trajectory(agent, eval_envs, trajectory_length, device, seed):
    """Collect a single trajectory and return success status and episode data."""
    episode = []
    eval_obs, info = eval_envs.reset()
    episode_success = 0

    action_space_low, action_space_high = torch.from_numpy(eval_envs.single_action_space.low).to(device), torch.from_numpy(eval_envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    for step in trange(trajectory_length):
        # only for shoe on rack
        # if episode_success > 15:
        if episode_success > 5:
            print(f"Episode length: {step}")
            break
            
        with torch.no_grad():
            # Get observation
            obs_dict = process_observation_datacollect(eval_obs, eval_envs)
            obs_dict['real_proprio'] = info['proprio'][0].cpu().numpy()
            # Get action from policy
            action = clip_action(agent.get_action(eval_obs, deterministic=True))
            # Take step in environment
            if not info['is_tcp_near_obj'][0]:
                action[..., -1] = 1
            # Only for shoe on rack
            # if episode_success > 0:
            #     action[..., :-1] = 0
            #     if episode_success < 7:
            #         action[..., 0] = 1
            #     if episode_success > 5:
            #         action[..., 2] = -0.3
                
            eval_obs, reward, terminated, truncated, info = eval_envs.step(action)
            # Only for letter arrange
            # action[..., -1] = torch.clamp(action[..., -1], min=0.3)
            
            # Calculate action deltas
            action_deltas = forward_scaling(action, eval_envs)
            action_deltas[-1] = (action[0][-1] + 1)/2
            
            # Store observation and action pair
            episode.append({
                'obs': obs_dict,
                'action': action_deltas
            })
            
            # Check for success
            if episode_success:
                episode_success += 1
            if not episode_success and info.get('success')[0]:
                episode_success += 1
            
    return episode_success, episode

def save_trajectory(episode, eval_output_dir, video_dir):
    """Save successful trajectory to file."""
    existing_demos = glob.glob(os.path.join(eval_output_dir, "demo*.npz"))
    next_idx = len(existing_demos)
    demo_path = os.path.join(eval_output_dir, f"demo{next_idx:05d}.npz")
    # breakpoint()
    # writes images in episode to eval_output_dir
    image_array = [timestep_data['obs']['zed_sim_images'] for timestep_data in episode]
    image_array = np.array(image_array)
    writer = imageio.get_writer(os.path.join(video_dir, f"demo{next_idx:05d}.mp4"), fps=5)
    for image in image_array:
        writer.append_data(image)
    writer.close()
    np.savez(demo_path, episode=episode)
    print(f"Successful demo saved to {demo_path}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Set up checkpoint and output directories
    if args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        eval_output_dir = os.path.join(checkpoint_dir, f"dp_trajectories")
        # Clear existing trajectories directory if it exists
        if os.path.exists(eval_output_dir):
            import shutil
            shutil.rmtree(eval_output_dir)
        os.makedirs(eval_output_dir, exist_ok=True)
        video_dir = os.path.join(checkpoint_dir, f"collected_videos/")
        if os.path.exists(video_dir):
            import shutil
            shutil.rmtree(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        print(f"Saving trajectories to {eval_output_dir}")

        eval_failure_output_dir = os.path.join(checkpoint_dir, f"dp_trajectories-failure")
        if os.path.exists(eval_failure_output_dir):
            import shutil
            shutil.rmtree(eval_failure_output_dir)
        os.makedirs(eval_failure_output_dir, exist_ok=True)
        video_dir_failure = os.path.join(checkpoint_dir, f"collected_videos_failure/")
        if os.path.exists(video_dir_failure):
            import shutil
            shutil.rmtree(video_dir_failure)
        os.makedirs(video_dir_failure, exist_ok=True)
        
    else:
        raise ValueError("Please provide a checkpoint path using --checkpoint")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", reward_mode=args.reward_mode, randomize_init_config=args.randomize_init_config, obj_noise=args.obj_noise, parallel_in_single_scene=True, randomize_camera=args.randomize_camera, randomize_lighting=args.randomize_camera)
    print(env_kwargs['obj_noise'])
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(eval_envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint))
    
    successful_trajectories = 0
    attempt = 0
    max_attempts = args.num_trajectories * 10
    
    while successful_trajectories < args.num_trajectories and attempt < max_attempts:
        print(f"Collecting trajectory attempt {attempt + 1} (successful so far: {successful_trajectories}/{args.num_trajectories})")
        agent.eval()
        
        success, episode = collect_single_trajectory(agent, eval_envs, args.trajectory_length, device, args.seed)
        
        if success:
            save_trajectory(episode, eval_output_dir, video_dir)
            successful_trajectories += 1
        else:
            save_trajectory(episode, eval_failure_output_dir, video_dir_failure)
            # successful_trajectories += 1
            print("Trajectory unsuccessful, trying again...")
        
        attempt += 1
    
    if successful_trajectories < args.num_trajectories:
        print(f"Warning: Only collected {successful_trajectories}/{args.num_trajectories} successful trajectories after {max_attempts} attempts")
    else:
        print(f"Successfully collected {args.num_trajectories} trajectories")
    
    eval_envs.close()