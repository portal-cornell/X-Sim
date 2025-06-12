"""
Utility module for loading metadata from npz files for xsim tasks.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from mani_skill import PACKAGE_ASSET_DIR


def load_task_metadata(task_name: str, asset_dir: str) -> Dict[str, Any]:
    """
    Load task metadata from npz file.
    
    Args:
        task_name: Name of the task (e.g., 'mustard_place', 'corn_in_basket')
        asset_dir: Asset directory name (e.g., 'kitchen_env', 'tabletop_env')
    
    Returns:
        Dictionary containing environment and task metadata
    """
    metadata_path = Path(PACKAGE_ASSET_DIR) / "xsim" / asset_dir / task_name / "metadata.npz"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load npz file
    npz_data = np.load(metadata_path, allow_pickle=True)
    
    # Reconstruct the metadata structure
    metadata = {
        "environment": {},
        "task": {},
        "robot_pos": npz_data["robot_pos"].tolist()
    }
    
    # Extract environment metadata
    for key in npz_data.keys():
        if key.startswith("env_"):
            env_key = key[4:]  # Remove 'env_' prefix
            value = npz_data[key]
            
            # Convert numpy arrays back to appropriate Python types
            if value.ndim == 0 and value.dtype.kind in ['U', 'S']:  # String
                metadata["environment"][env_key] = str(value)
            elif value.ndim == 0:  # Scalar
                metadata["environment"][env_key] = value.item()
            else:  # Array
                metadata["environment"][env_key] = value.tolist()
    
    # Extract task metadata
    for key in npz_data.keys():
        if key.startswith("task_"):
            task_key = key[5:]  # Remove 'task_' prefix
            value = npz_data[key]
            
            # Convert numpy arrays back to appropriate Python types
            if value.ndim == 0 and value.dtype.kind in ['U', 'S']:  # String
                metadata["task"][task_key] = str(value)
            elif value.ndim == 0:  # Scalar
                metadata["task"][task_key] = value.item()
            elif value.dtype == bool:  # Boolean array
                metadata["task"][task_key] = bool(value.item()) if value.ndim == 0 else value.tolist()
            elif value.dtype.kind in ['U', 'S']:  # String array
                metadata["task"][task_key] = value.tolist()
            else:  # Numeric array
                metadata["task"][task_key] = value.tolist()
    
    return metadata


def get_env_to_data_map(task_name: str, asset_dir: str) -> Dict[str, Any]:
    """
    Get environment configuration map for a task.
    
    Args:
        task_name: Name of the task
        asset_dir: Asset directory name
    
    Returns:
        Dictionary in the format expected by the original env_to_data_map
    """
    metadata = load_task_metadata(task_name, asset_dir)
    env_config = metadata["environment"]
    
    return {
        env_config["name"]: {
            "kitchen_to_robot_transform": env_config["kitchen_to_robot_transform"],
            "robot_qpos": env_config["robot_qpos"],
            "camera": {
                "eye": env_config["camera_eye"],
                "target": env_config["camera_target"]
            },
            "ground_altitude": env_config["ground_altitude"],
            "kitchen_mesh_name": env_config["kitchen_mesh_name"],
            "asset_base_path": env_config["asset_base_path"]
        }
    }


def get_task_to_data_map(task_name: str, asset_dir: str) -> Dict[str, Any]:
    """
    Get task configuration map for a task.
    
    Args:
        task_name: Name of the task
        asset_dir: Asset directory name
    
    Returns:
        Dictionary in the format expected by the original task_to_data_map
    """
    metadata = load_task_metadata(task_name, asset_dir)
    task_config = metadata["task"]
    
    return {
        task_name: task_config
    }


def get_robot_pos(task_name: str, asset_dir: str) -> list:
    """
    Get robot position offset for a task.
    
    Args:
        task_name: Name of the task
        asset_dir: Asset directory name
    
    Returns:
        List of robot position offsets [x, y, z]
    """
    metadata = load_task_metadata(task_name, asset_dir)
    return metadata["robot_pos"] 