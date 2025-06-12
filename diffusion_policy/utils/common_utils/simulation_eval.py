from dataclasses import dataclass
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import gymnasium as gym
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from common_utils.stopwatch import Stopwatch
from common_utils.recorders import DatasetRecorder
from common_utils import cprint, wrap_ruler
from torch.distributions.normal import Normal
from common_utils.scaling_utils import *

# Import scaling utils
from common_utils.scaling_utils import inverse_scaling
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles



@dataclass
class EvalConfig:
    # checkpoint path
    checkpoint_path: str = ""  # Path to the policy checkpoint
    # environment config
    env_id: str = ""
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "dense"
    randomize_init_config: bool = True
    obj_noise: float = 0.0
    num_eval_envs: int = 1
    # evaluation params
    show_camera: int = 0
    num_episodes: int = 10  # Number of episodes to run (regardless of success)
    freq: float = 5
    max_len: int = 100
    use_tcp: bool = False
    save_trajectories: bool = False
    # output directory
    output_dir: Optional[str] = None  # If None, will be derived from checkpoint_path

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """PPO Agent architecture for expert policy."""
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

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

def setup_environment(cfg: EvalConfig):
    """
    Set up the ManiSkill simulation environment.
    
    Args:
        cfg: Configuration for the environment
        
    Returns:
        The vectorized simulation environment
    """
    env_kwargs = dict(
        obs_mode="state", 
        render_mode="rgb_array", 
        sim_backend="gpu", 
        reward_mode=cfg.reward_mode, 
        randomize_init_config=cfg.randomize_init_config, 
        obj_noise=cfg.obj_noise, 
        parallel_in_single_scene=True
    )
    
    if cfg.control_mode is not None:
        env_kwargs["control_mode"] = cfg.control_mode
        
    eval_envs = gym.make(cfg.env_id, num_envs=cfg.num_eval_envs, **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(eval_envs, cfg.num_eval_envs, ignore_terminations=True, record_metrics=True)
    
    return eval_envs

def clip_action(action: torch.Tensor, env):
    action_space_low, action_space_high = torch.from_numpy(env.single_action_space.low).to('cuda'), torch.from_numpy(env.single_action_space.high).to('cuda')
    return torch.clamp(action.detach(), action_space_low, action_space_high)

def run_episode(
    policy,
    dataset,
    env,
    freq: float,
    max_len: int,
    stopwatch: Stopwatch,
    show_camera: int,
    recorder: DatasetRecorder = None,
    use_tcp: bool = False,
):
    """Run a single episode."""
    assert not policy.training
    episode = []
    success = False
    ever_grasped = False

    with stopwatch.time("reset"):
        obs, info = env.reset()

    cached_actions = []
    scaling_factor = 1
    first_grasp = 0

    for step in range(max_len):
        with stopwatch.time("observe"):
            raw_obs_dict, processed_obs = process_observation(obs, dataset, env)
            if 'proprio' in info:
                raw_obs_dict['real_proprio'] = info['proprio'][0].cpu().numpy()
                    
        # Generate actions based on policy type
        if hasattr(policy, 'get_action'):  # For PPO or similar policies
            with stopwatch.time("act"):
                action = clip_action(policy.get_action(processed_obs, deterministic=True), env)
                env_action = action[0].detach().cpu().numpy()
        else:  # For diffusion policies
            if len(cached_actions) == 0:
                with stopwatch.time("act"):
                    action_seq = policy.act(processed_obs, sim=False)
                
                for action in action_seq.split(1, dim=0):
                    split_action = action.clone()
                    split_action[:, :-1] /= scaling_factor
                    for _ in range(scaling_factor):
                        cached_actions.append(split_action.squeeze(0))
            
            action = cached_actions.pop(0)
            ee_pos, ee_euler, gripper_open = action.split([3, 3, 1])
            # if first_grasp < 2 and gripper_open < 0.7:
            #     first_grasp += 1
            #     gripper_open[0] = 1
            gripper_open = gripper_open * 0.05 - 0.01
            env_action = np.concatenate([
                ee_pos.detach().cpu().numpy(),
                ee_euler.detach().cpu().numpy(),
                [gripper_open.item()]
            ])
            env_action_unscaled = env_action[:]
            if recorder:
                recorder.record(None, raw_obs_dict, env_action_unscaled, click_pos=None)
            
            env_action = inverse_scaling(np.expand_dims(env_action, axis=0), env)

        # Save current state data
        episode.append({
            'obs': raw_obs_dict,
            'action': env_action
        })
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(env_action).to(obs.device))
        
        # Check if object is grasped in this step
        if 'is_grasped' in info and torch.any(info['is_grasped'][0]):
            ever_grasped = True
            
        # Check for success
        if info.get('success', [False])[0]:
            success = True
            break

    if recorder:
        recorder.end_episode(save=True)
    
    return {"success": success, "episode": episode, "length": step + 1, "ever_grasped": ever_grasped}

def run_episode_pose(
    policy,
    dataset,
    env,
    freq: float,
    max_len: int,
    stopwatch: Stopwatch,
    show_camera: int,
    recorder: DatasetRecorder = None,
    use_tcp: bool = False,
):
    """Run a single episode with pd_ee_pose control mode."""
    assert not policy.training
    episode = []
    success = False
    ever_grasped = False

    with stopwatch.time("reset"):
        obs, info = env.reset()

    cached_actions = []
    scaling_factor = 1

    for step in range(max_len):
        with stopwatch.time("observe"):
            raw_obs_dict, processed_obs = process_observation(obs, dataset, env)
            if 'proprio' in info:
                raw_obs_dict['real_proprio'] = info['proprio'][0].cpu().numpy()
                    
        # Generate actions based on policy type
        if hasattr(policy, 'get_action'):  # For PPO or similar policies
            with stopwatch.time("act"):
                action = clip_action(policy.get_action(processed_obs, deterministic=True), env)
                
                # For pd_ee_pose, we need to convert the action to absolute pose format
                # Extract position, orientation, and gripper components
                ee_pos = action[0, :3].detach().cpu().numpy()
                ee_euler = action[0, 3:6].detach().cpu().numpy()
                gripper_open = action[0, 6].detach().cpu().numpy()
                
                # Apply gripper thresholding similar to the second file
                if gripper_open < 0.5:
                    gripper_open = -1  # Closed gripper
                
                # Combine into robot pose format expected by pd_ee_pose
                env_action = np.array([*ee_pos, *ee_euler, gripper_open])
                
        else:  # For diffusion policies
            if len(cached_actions) == 0:
                with stopwatch.time("act"):
                    action_seq = policy.act(processed_obs, sim=False)
                
                for action in action_seq.split(1, dim=0):
                    split_action = action.clone()
                    split_action[:, :-1] /= scaling_factor
                    for _ in range(scaling_factor):
                        cached_actions.append(split_action.squeeze(0))
            
            action = cached_actions.pop(0)
            ee_pos, ee_euler_predicted, gripper_open = action.split([3, 3, 1])
            ee_quat = R.from_euler('xyz', ee_euler_predicted.cpu().numpy(), degrees=False).as_quat()
            ee_euler = matrix_to_euler_angles(quaternion_to_matrix(torch.tensor([ee_quat[-1], ee_quat[0], ee_quat[1], ee_quat[2]])), "XYZ")
            
            # Convert gripper value similar to what's done in the second file
            gripper_value = gripper_open.item()
            if gripper_value < 0.5:
                gripper_value = -1  # Closed gripper
                
            # Combine position, orientation and gripper state
            env_action = np.concatenate([
                ee_pos.detach().cpu().numpy(),
                ee_euler.detach().cpu().numpy(),
                [gripper_value]
            ])
            
            if recorder:
                env_action_unscaled = env_action[:]
                recorder.record(None, raw_obs_dict, env_action_unscaled, click_pos=None)
            
            # No need for inverse_scaling with pd_ee_pose as it uses absolute values

        # Save current state data
        episode.append({
            'obs': raw_obs_dict,
            'action': env_action
        })
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(env_action).to(obs.device))
        
        # Check if object is grasped in this step
        if 'is_grasped' in info and torch.any(info['is_grasped'][0]):
            ever_grasped = True
            
        # Check for success
        if info.get('success', [False])[0]:
            success = True
            break

    if recorder:
        recorder.end_episode(save=True)
    
    return {"success": success, "episode": episode, "length": step + 1, "ever_grasped": ever_grasped}

# Run evaluation with an existing environment for exactly K episodes
def run_evaluation(
    eval_env,
    policy, 
    dataset, 
    cfg: EvalConfig, 
    epoch_num: int, 
    save_dir: str, 
    stat,
    seed: int = 1
):
    """Run exactly K evaluation episodes and collect statistics using an existing environment"""
    print(f"\n{wrap_ruler('Simulation Evaluation - Epoch ' + str(epoch_num))}")
    
    # Setup recording if needed
    recorder = None
    if cfg.save_trajectories:
        recorder_dir = Path(save_dir, f"sim_rollouts_epoch_{epoch_num}")
        recorder_dir.mkdir(parents=True, exist_ok=True)
        recorder = DatasetRecorder(str(recorder_dir))
    
    # Run evaluation
    stopwatch = Stopwatch()
    successful_trajectories = 0
    grasped_trajectories = 0
    episode_lengths = []
    successful_episode_lengths = []
    
    # Run exactly K episodes
    for episode_num in range(cfg.num_episodes):
        # Set seeds for reproducibility
        episode_seed = seed + episode_num + (epoch_num * 1000)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        
        print(f"Running evaluation episode {episode_num + 1}/{cfg.num_episodes}")
        
        if cfg.control_mode == "pd_ee_delta_pose":
            result = run_episode(
                policy, dataset, eval_env, cfg.freq, cfg.max_len, 
                stopwatch, cfg.show_camera, recorder, 
                use_tcp=cfg.use_tcp
            )
        if cfg.control_mode == "pd_ee_pose":
            result = run_episode_pose(
                policy, dataset, eval_env, cfg.freq, cfg.max_len, 
                stopwatch, cfg.show_camera, recorder, 
                use_tcp=cfg.use_tcp
            )

        episode_lengths.append(result["length"])
        
        if result["success"]:
            print(f"Episode successful! Length: {result['length']}")
            successful_trajectories += 1
            successful_episode_lengths.append(result["length"])
        else:
            print(f"Episode unsuccessful. Length: {result['length']}")
            
        # Track grasp statistics
        if result["ever_grasped"]:
            grasped_trajectories += 1
    
    # Calculate and report statistics
    success_rate = (successful_trajectories / cfg.num_episodes) * 100
    grasp_rate = (grasped_trajectories / cfg.num_episodes) * 100
    avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
    avg_success_length = np.mean(successful_episode_lengths) if successful_episode_lengths else 0
    
    print(f"Evaluation Statistics:")
    print(f"Success Rate: {success_rate:.2f}% ({successful_trajectories}/{cfg.num_episodes})")
    print(f"Grasp Rate: {grasp_rate:.2f}% ({grasped_trajectories}/{cfg.num_episodes})")
    print(f"Average Episode Length: {avg_episode_length:.2f}")
    print(f"Average Successful Episode Length: {avg_success_length:.2f}")
    
    # Record statistics to logging
    stat[f"eval/success_rate"].append(success_rate)
    stat[f"eval/grasp_rate"].append(grasp_rate)
    stat[f"eval/avg_episode_length"].append(avg_episode_length)
    stat[f"eval/avg_success_length"].append(avg_success_length)
    stat[f"eval/episodes"].append(cfg.num_episodes)
    
    # Save detailed statistics to file
    stats_path = os.path.join(save_dir, f"eval_stats_epoch_{epoch_num}.txt")
    with open(stats_path, "w") as f:
        f.write(f"Epoch: {epoch_num}\n")
        f.write(f"Environment: {cfg.env_id}\n")
        f.write(f"Episodes: {cfg.num_episodes}\n")
        f.write(f"Success Rate: {success_rate:.2f}% ({successful_trajectories}/{cfg.num_episodes})\n")
        f.write(f"Grasp Rate: {grasp_rate:.2f}% ({grasped_trajectories}/{cfg.num_episodes})\n")
        f.write(f"Average Episode Length: {avg_episode_length:.2f}\n")
        f.write(f"Average Successful Episode Length: {avg_success_length:.2f}\n")
    
    print(f"Statistics saved to {stats_path}")
    
    return success_rate

# For backward compatibility
def evaluate_policy(
    policy, 
    dataset, 
    cfg: EvalConfig, 
    epoch_num: int, 
    save_dir: str, 
    stat,
    seed: int = 1
):
    """
    Create an environment and run evaluation (legacy function for compatibility)
    """
    eval_env = setup_environment(cfg)
    try:
        success_rate = run_evaluation(
            eval_env, policy, dataset, cfg, epoch_num, save_dir, stat, seed
        )
    finally:
        eval_env.close()
    
    return success_rate

if __name__ == "__main__":
    import rich.traceback
    from train_with_sim_eval import load_model
    import common_utils
    import sys
    import torch.backends.cudnn
    
    rich.traceback.install()
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Parse command line arguments
    cfg = tyro.cli(EvalConfig)
    
    # Validate checkpoint path
    if not cfg.checkpoint_path:
        print("Error: No checkpoint path specified. Use --checkpoint_path to specify a model checkpoint.")
        sys.exit(1)
    
    if not os.path.exists(cfg.checkpoint_path):
        print(f"Error: Checkpoint file {cfg.checkpoint_path} does not exist.")
        sys.exit(1)
    
    # Setup output directory
    if cfg.output_dir is None:
        # Derive output directory from checkpoint path
        checkpoint_dir = os.path.dirname(cfg.checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(cfg.checkpoint_path))[0]
        cfg.output_dir = os.path.join(checkpoint_dir, f"eval_{cfg.env_id}_{checkpoint_name}")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"{wrap_ruler('Evaluating policy on environment: ' + cfg.env_id)}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Output directory: {cfg.output_dir}")
    
    # Create a stat tracker
    stat = common_utils.MultiCounter(cfg.output_dir, False)
    
    # Set random seed for reproducibility
    seed = 0
    common_utils.set_all_seeds(seed)
    
    # Load the model
    print(f"{wrap_ruler('Loading model from checkpoint')}")
    policy, dataset, _ = load_model(cfg.checkpoint_path, "cuda")
    
    # Setup DDIM for faster inference if using diffusion policy
    if hasattr(policy, 'cfg') and hasattr(policy.cfg, 'use_ddpm') and policy.cfg.use_ddpm:
        cprint(f"Warning: override to use ddim with step 10 for faster inference")
        policy.cfg.use_ddpm = 0
        policy.cfg.ddim.num_inference_timesteps = 10
    
    policy.eval()
    
    # Run evaluation
    print(f"{wrap_ruler('Starting evaluation')}")
    success_rate = evaluate_policy(
        policy=policy,
        dataset=dataset,
        cfg=cfg,
        epoch_num=0,  # Use 0 for standalone evaluation
        save_dir=cfg.output_dir,
        stat=stat,
        seed=seed
    )
    
    print(f"\n{wrap_ruler('Evaluation Complete')}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Results saved to: {cfg.output_dir}")