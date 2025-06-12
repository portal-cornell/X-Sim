from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.train_utils import process_namedtuple_batch
from ..dataset_utils.common import unnormalize_data


def process_cv2_color(color):
    return list(map(int, (color * 255)))[:3]


def o3d_to_img_array(vis, crop_percentage=0.3, resize=None):
    """
    Convert an Open3D visualizer to an image array, with pixel values
    as uint8 between [0, 255]

    Arguments:
        vis (open3d.visualization.Visualizer): The Open3D visualizer.
        crop_percentage (float): The percentage of the image to crop.
            e.g.: 0.3 means take the center 40% of the image.
        resize: The size to resize the image to.
    """
    o3d_img = vis.capture_screen_float_buffer(False)
    img_arr = (np.array(o3d_img) * 255).astype(np.uint8)
    h, w = img_arr.shape[:2]
    if crop_percentage > 0.0:
        left = int(w * (crop_percentage))
        right = int(w * (1 - crop_percentage))
        top = int(h * (crop_percentage))
        bottom = int(h * (1 - crop_percentage))
        img_arr = img_arr[top:bottom, left:right]
    if resize:
        img_arr = cv2.resize(img_arr, resize)
    return img_arr


def get_track_colors(
    action_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, plt.cm.ScalarMappable]]:
    """
    Initialize common visualization parameters.

    Args:
        action_horizon (int): The number of steps in the action sequence.
        add_grasp_info_to_tracks (bool): Whether to add grasp information to points.

    Returns:
        Tuple containing:
        - track_colors (np.ndarray): Colors for tracking visualization.
        - grasp_colors (np.ndarray): Colors for grasping visualization.
        - cmap_dict (Dict[str, plt.cm.ScalarMappable]): Dictionary of color maps.
    """
    track_cmap = plt.get_cmap("autumn_r")
    grasp_cmap = plt.get_cmap("winter")
    values = np.linspace(0, 1, action_horizon)
    track_colors = track_cmap(values)
    grasp_colors = grasp_cmap(values)

    cmap_dict = {"track": track_cmap, "grasp": grasp_cmap}

    return track_colors, grasp_colors, cmap_dict


def visualize_image_tracks_2d(
    policy: nn.Module,
    dataloader: DataLoader,
    stats: Dict[str, np.ndarray],
    action_horizon: int,
    device: str = "cuda",
    max_frames: int = 100,
    add_grasp_info_to_tracks: bool = False,
) -> np.ndarray:
    """
    Generates visualizations for predicted tracks on pointclouds and saves them as a GIF.
    """
    # Set up color maps for tracks and grasps
    track_cmap = plt.get_cmap("autumn_r")
    grasp_cmap = plt.get_cmap("winter")
    values = np.linspace(0, 1, action_horizon)
    track_colors = track_cmap(values)
    grasp_colors = grasp_cmap(values)

    frames = []
    dims_per_point = 2
    for idx, batch in tqdm(
        enumerate(iter(dataloader)),
        total=len(dataloader),
        desc="Generating Visualizations",
    ):
        batch = process_namedtuple_batch(batch, device)
        # Remove the batch dim -- denoised_action.shape == (action_horizon, num_points*2)
        image, state_cond, first_action = (
            batch.obs[0],  # (1, C, H, W)
            batch.state_cond[0],  # (1, num_points*2)
            batch.action[0, 0],  # (num_points*2) -- should be same values as state_cond
        )
        denoised_action = policy.act(image, state_cond, first_action_abs=first_action)
        denoised_action = denoised_action.cpu().numpy()
        # Remove obs_horizon dim -- last_image.shape == (H, W, C)
        last_image = image[-1].cpu().numpy().transpose(1, 2, 0)
        last_image = unnormalize_data(last_image, stats=stats["obs"]).astype(np.uint8)
        action = unnormalize_data(denoised_action, stats=stats["action"])
        action, terminal = action[:, :-1], action[:, -1]

        vis = last_image.copy()
        if add_grasp_info_to_tracks:
            action, grasp = action[:, :-1], action[:, -1]
        action = action.reshape((action_horizon, -1, dims_per_point))
        for timestep, action_t in enumerate(action):
            if terminal[timestep] > 0.5:
                color = (0, 0, 0)
            elif add_grasp_info_to_tracks:
                # Select colors based on grasp values
                color = (
                    grasp_colors[timestep]
                    if grasp[timestep] <= 0.5
                    else track_colors[timestep]
                )
            else:
                color = track_colors[timestep]
            color = process_cv2_color(color)
            for u, v in action_t:
                cv2.circle(vis, (int(u), int(v)), 2, color, -1)

        # added as (H, W, C)
        frames.append(vis)
        if len(frames) > max_frames:
            break
    return np.array(frames)


def save_frames(frames: np.ndarray, file_extension_type: str, file_name: str, fps=10):
    """
    Given a set of frames, use imageio to save them as a file_extension_type
    """
    import imageio

    if file_extension_type == "gif":
        imageio.mimsave(f"{file_name}.gif", frames, fps=fps)
    elif file_extension_type == "mp4":
        with imageio.get_writer(f"{file_name}.mp4", fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ValueError(f"Unsupported file extension type: {file_extension_type}")

    print(f"Visualization saved as '{file_name}.{file_extension_type}'")
