import os
import sys
import glob
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.backends.cudnn
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing import Optional
import torch.nn as nn
from ..dataset_utils.dataset import BCDataset



import gymnasium as gym
from mani_skill.utils import gym_utils
def process_image(image):
    """Process rendered image to the correct format."""
    # Side crop so it's 1920x1440
    image = image[:, 240:-240]
    # image = image[:, 480:]
    # Resize down to 960x720
    image = cv2.resize(image, (960, 720))
    return image

def process_observation(obs, dataset: BCDataset, env):
    """
    Process raw observation into a format suitable for the diffusion policy.
    
    Args:
        obs: Raw observation from the environment
        dataset: BCDataset object for processing
        env: Environment for rendering
        
    Returns:
        Raw observation dictionary and processed observation for the policy
    """
    ee_pos = obs[0, :3].cpu().numpy()
    ee_quat = np.roll(obs[0, 3:7].cpu().numpy(), -1)  # Convert from wxyz to xyzw
    
    obs_dict = {
        'ee_pos': ee_pos,
        'ee_quat': ee_quat,
        'ee_euler': R.from_quat(ee_quat).as_euler('xyz'),
        'gripper_open': float(obs[0, 7]/0.04),  # Normalize gripper to 0-1 range
        'zed_sim_images': process_image(env.render().cpu().numpy()[0])
    }
    
    obs_dict['proprio'] = np.concatenate([
        obs_dict['ee_pos'], 
        obs_dict['ee_euler'], 
        [obs_dict['gripper_open']]
    ])
    
    # Process for model input
    processed_obs = dataset.process_observation(obs_dict)
    for k, v in processed_obs.items():
        processed_obs[k] = v.cuda()
        
    return obs_dict, processed_obs

def process_observation_datacollect(obs, env):
    """
    Process raw observation into a format suitable for the diffusion policy.
    
    Args:
        obs: Raw observation from the environment
        dataset: BCDataset object for processing
        env: Environment for rendering
        
    Returns:
        Raw observation dictionary and processed observation for the policy
    """
    ee_pos = obs[0, :3].cpu().numpy()
    ee_quat = np.roll(obs[0, 3:7].cpu().numpy(), -1)  # Convert from wxyz to xyzw
    
    obs_dict = {
        'ee_pos': ee_pos,
        'ee_quat': ee_quat,
        'ee_euler': R.from_quat(ee_quat).as_euler('xyz'),
        'gripper_open': float(obs[0, 7]/0.04),  # Normalize gripper to 0-1 range
        'zed_sim_images': process_image(env.render().cpu().numpy()[0])
    }
    
    obs_dict['proprio'] = np.concatenate([
        obs_dict['ee_pos'], 
        obs_dict['ee_euler'], 
        [obs_dict['gripper_open']]
    ])
    
    return obs_dict

def inverse_scaling(action, env):
    """
    Inverse scale the action from normalized space [-1, 1] to the environment's action space.
    
    Args:
        action: The action in normalized space
        env: The environment instance
        
    Returns:
        Scaled action that fits within the environment's action space
    """
    # Get the action space bounds from the environment using the configured control mode
    action = torch.Tensor(action).to(env.unwrapped.device)
    control_mode = env.unwrapped.agent.control_mode
    arm_controller = env.unwrapped.agent.controllers[control_mode].controllers['arm']
    gripper_controller = env.unwrapped.agent.controllers[control_mode].controllers['gripper']
    pos_rot_action = arm_controller._inv_clip_and_scale_action(action[:,:-1])[0]
    gripper_action = gym_utils.inv_clip_and_scale_action(action[:,-1:], gripper_controller.action_space_low, gripper_controller.action_space_high)[0]
    
    scaled_action = np.concatenate([pos_rot_action.cpu(), gripper_action.cpu()])
    return scaled_action

def forward_scaling(action, env):
    """
    Forward scale the action from normalized space [-1, 1] to the environment's action space.
    
    Args:
        action: The action in normalized space
        env: The environment instance
        
    Returns:
        Scaled action that fits within the environment's action space
    """
    # Get the action space bounds from the environment using the configured control mode
    control_mode = env.unwrapped.agent.control_mode
    arm_controller = env.unwrapped.agent.controllers[control_mode].controllers['arm']
    gripper_controller = env.unwrapped.agent.controllers[control_mode].controllers['gripper']

    pos_rot_action = arm_controller._preprocess_action(action[:,:-1])[0]
    gripper_action = gripper_controller._preprocess_action(action[:,-1:])[0]
    scaled_action = np.concatenate([pos_rot_action.cpu(), gripper_action.cpu()])

    return scaled_action