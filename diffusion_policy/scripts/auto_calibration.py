from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import glob

import gymnasium as gym
import tyro
from copy import deepcopy
# ManiSkill specific imports
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs.pose import Pose

import numpy as np
from tqdm import tqdm
import cv2
import imageio
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from pathlib import Path
from scipy.spatial.transform import Rotation
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles

@dataclass
class Args:
    # Algorithm specific arguments
    env_id: str = ""
    """the id of the environment"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""

    input_dir: str = "demos"
    """Directory containing both robot demonstration (.npz) and object pose (.npy) files with matching prefixes"""
    create_video: bool = False
    """if true, create a video of generated simulation images"""

def _crop(image):
    # remove 240 from each side
    return image[:, 240:-240]

def get_all_files(root, file_extension, contain=None) -> list[str]:
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if file_extension is not None:
                if f.endswith(file_extension):
                    if contain is None or contain in os.path.join(folder, f):
                        files.append(os.path.join(folder, f))
            else:
                if contain in f:
                    files.append(os.path.join(folder, f))
    return files

def process_demo_file(demo_path, object_pose_path, output_path, env, create_video=True):
    # Initialize robot and object
    robot_file = np.load(demo_path, allow_pickle=True)['episode']
    
    if object_pose_path is not None and os.path.exists(object_pose_path):
        object_poses = torch.tensor(np.load(object_pose_path, allow_pickle=True)).float().to('cuda')
        for i in range(len(object_poses)):
            pose_data = object_poses[i][0]
            env.unwrapped.start_poses[i] = pose_data

    eval_obs, _ = env.reset()
    
    robot_poses = [step['obs']['proprio'] for step in robot_file]
    robot_pos_quat = [(step['obs']['ee_pos'], step['obs']['ee_quat'], step['obs']['gripper_open']) for step in robot_file]

    # recording simulation image output
    sim_images = []
    sim_record = []
    finished = False
    
    for i in tqdm(range(len(robot_poses)), desc=f"Processing {Path(demo_path).name}", leave=False):
        ee_pos, ee_quat, gripper_open = robot_pos_quat[i]
        ee_euler = matrix_to_euler_angles(quaternion_to_matrix(torch.tensor([ee_quat[-1], ee_quat[0], ee_quat[1], ee_quat[2]])), "XYZ")
        robot_pose = np.hstack([ee_pos, ee_euler.cpu().numpy(), gripper_open])

        if robot_pose[-1] < 0.5:
            robot_pose[-1] = -1

        obs, reward, terminated, truncated, info = env.step(robot_pose)
        
        finished = finished or terminated
        sim_img = env.render().cpu().numpy()[0]
        sim_img = cv2.resize(_crop(sim_img), (960, 720))
        sim_images.append(sim_img)
        
        step_info = deepcopy(robot_file[i])
        
        # Handle the case where these keys might not exist
        if 'zed_image' in step_info['obs']:
            del step_info['obs']['zed_image']
        if 'zed_depth' in step_info['obs']:
            del step_info['obs']['zed_depth']
            
        step_info['obs']['zed_sim_images'] = sim_img
        sim_record.append(step_info)

    sim_record = {"episode": sim_record}
    
    # Save the processed data
    np.savez(output_path.with_suffix('.npz'), **sim_record)

    # Create standard simulation video if requested
    if create_video and sim_images:
        height, width, layers = sim_images[0].shape
        fps = 5
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path.with_suffix('.mp4'), fourcc, fps, (width, height))
        for img in sim_images:
            video.write(img)
        video.release()
        
    return finished

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Get all npz files in the input directory
    demo_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
    if not demo_files:
        print(f"No .npz files found in {args.input_dir}")
        exit(1)
    
    # Create output directory
    # Back out one directory from input_dir and create real2sim_paired
    parent_dir = os.path.dirname(os.path.abspath(args.input_dir))
    output_dir = os.path.join(parent_dir, "real2sim_paired")
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize environment
    env_kwargs = dict(
        obs_mode="state", 
        render_mode="rgb_array", 
        sim_backend="gpu", 
        reward_mode="dense", 
        randomize_init_config=False, 
        obj_noise=0.0, 
        parallel_in_single_scene=True,
    )
    env_kwargs["control_mode"] = "pd_ee_pose"
    env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True, auto_reset=False)

    # Process each demo file
    print(f"Processing {len(demo_files)} demonstration files...")
    for demo_path in tqdm(demo_files, desc="Overall Progress"):
        demo_filename = Path(demo_path).name
        demo_prefix = Path(demo_path).stem  # Gets filename without extension
        
        # Find corresponding object pose file with same prefix
        object_pose_path = os.path.join(args.input_dir, f"{demo_prefix}.npy")
        if not os.path.exists(object_pose_path):
            print(f"Warning: No matching object pose file found for {demo_filename}")
            object_pose_path = None
        
        output_path = Path(output_dir, demo_filename)
        
        try:
            finished = process_demo_file(
                demo_path, 
                object_pose_path, 
                output_path, 
                env, 
                create_video=args.create_video
            )
            print(f"Processed {demo_filename} -> {output_path}")
        except Exception as e:
            print(f"Error processing {demo_filename}: {e}")
    
    print(f"All files processed. Results saved to {output_dir}")
    
    # Create side-by-side videos after all processing is done
    print("Creating side-by-side comparison videos...")
    
    real_files = get_all_files(args.input_dir, "npz")
    sim_files = get_all_files(output_dir, "npz")
    
    # Create a dictionary to map file names to their paths
    real_dict = {os.path.basename(f): f for f in real_files}
    sim_dict = {os.path.basename(f): f for f in sim_files}
    
    for filename in set(real_dict.keys()).intersection(sim_dict.keys()):
        real_episode = np.load(real_dict[filename], allow_pickle=True)["episode"]
        sim_episode = np.load(sim_dict[filename], allow_pickle=True)["episode"]
        
        frames = []
        for real_timestep, sim_timestep in zip(real_episode, sim_episode):
            # Get the real frame
            real_frame = real_timestep['obs']['zed_sim_images']
            # Get the simulation frame
            sim_frame = sim_timestep['obs']['zed_sim_images']
            
            if real_frame.shape == (720, 960, 3) and sim_frame.shape == (720, 960, 3):
                # Stack frames horizontally
                stacked_frame = np.hstack((real_frame, sim_frame))
                frames.append(stacked_frame)

        # If we have frames to write
        if frames:
            # Create output video using imageio
            output_video_path = os.path.join(output_dir, f"stacked_{filename.replace('.npz', '.mp4')}")
            writer = imageio.get_writer(output_video_path, fps=5, quality=7)
            
            for frame in tqdm(frames, desc=f"Processing {filename}"):
                writer.append_data(frame)
                
            writer.close()
            print(f"Saved stacked video for {filename} to {output_video_path}")

    print(f"All files processed. Results saved to {output_dir}")