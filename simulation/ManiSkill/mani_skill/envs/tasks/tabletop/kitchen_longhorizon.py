from typing import Any, Dict, Union, List

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_angle, axis_angle_to_quaternion
from mani_skill.utils.building.ground import build_ground
from mani_skill import PACKAGE_ASSET_DIR
import transforms3d


@register_env("Kitchen-LongHorizon", max_episode_steps=600)
class KitchenLongHorizonEnv(BaseEnv):
    """
    **Task Description:**
    A task where the objective is to grasp objects and move them to target goal positions.

    **Randomizations:**
    - objects' xy positions are randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. They are placed flat on the table
    - objects' z-axis rotations are randomized to random angles
    - the target goal positions (marked by green spheres) of the objects have their xy positions randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - all objects' positions are within `goal_thresh` (default 0.04m) euclidean distance of their goal positions
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "panda_ninja", "fetch"]
    agent: Union[Panda, Fetch]
    cube_half_size = 0.025
    goal_thresh = 0.04
    angle_goal_thresh = 0.15 # 8 degrees
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def __init__(
        self,
        *args,
        robot_uids="panda_ninja",
        robot_init_qpos_noise=0.02,
        randomize_init_config=False,
        obj_noise=0.0,
        obj_names=['basket_centered'],  # Now accepts a list of object names
        demo_name='longhorizon-pick',
        num_waypoints=3,
        visualize=False,
        rotation_reward=True,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize_init_config = randomize_init_config

        self.obj_noise = obj_noise
        self.obj_names = obj_names if isinstance(obj_names, list) else [obj_names]  # Ensure it's a list
        self.obj_files = [f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/{obj_name}/mesh/{obj_name}.obj" 
                        for obj_name in self.obj_names]

        self.kitchen_file = f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/Kitchen.obj"
        self.kitchen_to_robot_transform = torch.tensor([0.4147 + 0.01, 0.1753 - 0.01, 0.1875 - 0.01])

        self.visualize = visualize
        self.rotation_reward = rotation_reward

        self.demo_name = demo_name
        # We'll have a demo path for each object
        self.demo_paths = [f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/{obj_name}/demos/{self.demo_name}.npy"
                          for obj_name in self.obj_names]
        
        self.start_idx = 115  # index of the first waypoint we want to compute reward w.r.t.
        self.end_idx = 278  # index of the goal destination
        self.num_waypoints = num_waypoints  # num waypoints, including goal (set to 1 for goal-conditioned)

        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig(
            uid="base_camera", 
            pose=pose, 
            width=128, 
            height=128, 
            # fov=np.pi / 2,
            near=0.01, 
            far=100, 
            intrinsic=[
                [5.331700e+02, 0.000000e+00, 4.862300e+02],
                [0.000000e+00, 5.331000e+02, 2.614790e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00],
            ]
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose_new = sapien_utils.look_at([-0.03740435, -0.45771217,  0.61309804], 
                                    [0.77913187, -0.02136986,  0.23511127])
        return CameraConfig(
            uid="render_camera",  
            pose=pose_new, 
            width=960, 
            height=540, 
            # fov=1,  
            near=0.01, 
            far=100, 
            intrinsic=[
                [5.331700e+02, 0.000000e+00, 4.862300e+02],
                [0.000000e+00, 5.331000e+02, 2.614790e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00],
            ]
        )
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**23, max_rigid_patch_count=2**22
            ),
            sim_freq=100,
            control_freq=5,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0]))

    def _load_kitchen(self, scene):
        builder = scene.create_actor_builder()
        density = 1000
        builder.add_multiple_convex_collisions_from_file(
            filename=self.kitchen_file,
            material = None,
            density=density,
            decomposition="coacd",
            decomposition_params={
                "threshold": 0.01,
                "preprocess_resolution": 50,
                "mcts_nodes": 25,
                "mcts_iterations": 200,
                "mcts_max_depth": 4,
                },
        )
        
        builder.add_visual_from_file(
            filename=self.kitchen_file,
        )
        builder.initial_pose = sapien.Pose()
        mesh = builder.build_kinematic(name="kitchen")
        
        return mesh

    def load_obj(self, scene, obj_file, name="obj", kinematic=False):
        breakpoint()
        builder = scene.create_actor_builder()
        density = 1000
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_file,
            material=None,
            density=density,
            decomposition="coacd",
        )
        builder.add_visual_from_file(filename=obj_file)
        builder.initial_pose = sapien.Pose()
        if kinematic:
            obj = builder.build_kinematic(name=name)
        else:
            obj = builder.build(name=name)
        return obj
    
    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.kitchen_mesh = self._load_kitchen(self.scene)
        
        # Load all objects and their demo data
        self.interval = (self.end_idx - self.start_idx) // self.num_waypoints
        self.poses = []
        self.waypoints = []
        self.waypoint_dist_reward_scaling = []
        self.start_poses = []
        self.goal_poses = []
        self.obj_meshes = []
        self.goal_sites = []
        for i, obj_name in enumerate(self.obj_names):
            obj_file = self.obj_files[i]
            demo_path = self.demo_paths[i]
            # Load the object mesh

            obj_mesh = self.load_obj(self.scene, obj_file, name=obj_name)
            self.obj_meshes.append(obj_mesh)
            
            # Load the demo data
            pose_data = torch.tensor(np.load(demo_path)).float().to(self.device)[i]  # Assumes just 1 object at index 0
            self.poses.append(pose_data)
            
            # Select waypoints for this object
            obj_waypoints = []
            obj_waypoints.append(pose_data[0])
            for j in range(self.start_idx, self.end_idx - self.interval, self.interval):
                obj_waypoints.append(pose_data[j])
            obj_waypoints.append(pose_data[self.end_idx])
            
            # Calculate distance reward scaling for this object
            obj_scaling = torch.tensor([
                1 / torch.norm(obj_waypoints[j][:3] - obj_waypoints[j+1][:3], p=2).item()
                for j in range(len(obj_waypoints) - 1)
            ]).to(self.device)
            self.waypoint_dist_reward_scaling.append(obj_scaling)
            
            # Remove the first waypoint (starting position)
            obj_waypoints.pop(0)
            self.waypoints.append(obj_waypoints)
            
            # Store start and goal poses
            self.start_poses.append(pose_data[0])
            self.goal_poses.append(pose_data[-1])
            
            # Set the rotation of all waypoints and goal to match the start pose
            for j in range(len(obj_waypoints)):
                obj_waypoints[j][3:] = pose_data[0][3:7]  # Set rotation to start pose rotation
        
        # Initialize visualization sites for waypoints if needed
        if self.visualize:
            self.waypoint_sites = []
            for i, obj_waypoints in enumerate(self.waypoints):
                obj_sites = []
                for j, waypoint in enumerate(obj_waypoints):
                    site_name = f"{self.obj_names[i]}_waypoint_{j}"
                    waypoint_site = self.load_obj(self.scene, self.obj_files[i], name=site_name, kinematic=True)
                    waypoint_site.set_pose(Pose.create_from_pq(p=waypoint[:3], q=waypoint[3:]))
                    obj_sites.append(waypoint_site)
                self.waypoint_sites.append(obj_sites)
        
        # Initialize tracking variables
        num_objects = len(self.obj_names)
        self.goal_quat = torch.zeros((self.num_envs, num_objects, 4), device=self.device)
        for i in range(num_objects):
            self.goal_quat[:, i] = self.start_poses[i][3:7]  # Set goal rotation to start pose rotation
        
        self.current_subgoal_idx = torch.zeros((self.num_envs, num_objects), dtype=torch.int32, device=self.device)
        
        self.goal_xyz = torch.zeros((self.num_envs, num_objects, 3), device=self.device)
        for i in range(num_objects):
            self.goal_xyz[:, i, :3] = self.goal_poses[i][:3]
        
        # Create goal visualization sites
        for i, obj_name in enumerate(self.obj_names):
            goal_site = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size / 2,
                color=[0, 1, 0, 0.5],
                name=f"{obj_name}_goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self._hidden_objects.append(goal_site)
            self.goal_sites.append(goal_site)
        
        # TCP visualization site
        self.tcp_site = actors.build_cube(
            self.scene,
            half_size=0.01,
            color=[1, 1, 0, 1],
            name="tcp_pose",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.tcp_site)
        
        # Calculate interval (should be the same for all objects)
        self.interval = (self.end_idx - self.start_idx) // self.num_waypoints

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            # Set up kitchen position
            kitchen_xyz = torch.zeros((b, 3))
            kitchen_xyz[:, :3] = self.kitchen_to_robot_transform
            
            # Apply randomization to z-coordinate if needed
            z_random = torch.zeros((b, 1))
            if self.randomize_init_config:
                z_random = torch.rand((b, 1)) * 0.03 - 0.015
                kitchen_xyz[:, 2:3] += z_random
            
            # Set kitchen pose
            kitchen_rot = transforms3d.quaternions.axangle2quat(
                np.array([1, 0, 0]), theta=np.deg2rad(90)
            )
            self.kitchen_mesh.set_pose(Pose.create_from_pq(p=kitchen_xyz, q=kitchen_rot))
            
            # Initialize each object
            for i, obj_mesh in enumerate(self.obj_meshes):
                # Set initial position
                xyz = torch.zeros((b, 3))
                xyz[:, :3] = self.start_poses[i][:3]
                
                # Apply randomization if needed
                if self.randomize_init_config:
                    xyz[:, :2] += torch.rand((b, 2)) * 0.05 - 0.025
                    xyz[:, 2:3] += z_random  # Apply same z-random as kitchen
                
                # Set initial rotation
                quat = torch.zeros((b, 4))
                quat[:, ] = self.start_poses[i][3:7]
                qs = torch.zeros((b, 4))
                qs[:, 0] = 1
                
                if self.randomize_init_config:
                    qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, bounds=(0, np.pi/6))
                
                fixed_rot = Pose.create_from_pq(q=qs) * Pose.create_from_pq(q=quat)
                
                # Set object pose
                obj_mesh.set_pose(Pose.create_from_pq(p=xyz, q=fixed_rot.q))
                
                # Update goal position with randomization if needed
                self.goal_xyz[env_idx, i, :3] = self.goal_poses[i][:3]
                if self.randomize_init_config:
                    self.goal_xyz[env_idx, i, 2:3] += z_random
                
                # Update goal quaternion
                self.goal_quat[env_idx, i] = self.start_poses[i][3:7]
                
                # Update visualization if needed
                if self.visualize:
                    self.goal_sites[i].set_pose(Pose.create_from_pq(
                        p=self.goal_xyz[env_idx, i, :3], 
                        q=self.goal_quat[env_idx, i]
                    ))
                    
                    # Update waypoint visualizations
                    for j, waypoint in enumerate(self.waypoints[i]):
                        wp_xyz = waypoint[:3].clone()
                        wp_quat = waypoint[3:].clone()
                        # Apply same z-randomization as objects if needed
                        if self.randomize_init_config:
                            wp_xyz[2:3] += z_random[0, 0]
                        self.waypoint_sites[i][j].set_pose(Pose.create_from_pq(p=wp_xyz, q=wp_quat))
            
            # Reset subgoal tracking indices
            self.current_subgoal_idx[env_idx] = 0
            
            # Set robot initial configuration
            qpos = np.array(
                [0.0, -0.7853, -0.0, -2.3561, 0.0, 1.5707, 0.7853, 0.04, 0.04]
            )
            if self._enhanced_determinism:
                qpos = (
                    self._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0.0]))
    
    def _get_obs_agent(self):
        return dict()
    
    def _get_obs_extra(self, info: Dict):
        num_objects = len(self.obj_names)
        
        # Create observation dict with basic information
        obs = dict(
            ee_pose=self.agent.tcp.pose.raw_pose,
            gripper_width=self.agent.robot.get_qpos()[:, -1:],
            goal_pos=self.goal_xyz,   # Shape: (num_envs, num_objects, 3)
            goal_rot=self.goal_quat,  # Shape: (num_envs, num_objects, 4)
            num_objects=torch.tensor([num_objects], device=self.device),
        )
        
        # Add object state observations if needed
        if "state" in self.obs_mode:
            obj_poses = []
            for obj_mesh in self.obj_meshes:
                obj_obs = obj_mesh.pose.raw_pose
                obj_noise = torch.rand_like(obj_obs) * (self.obj_noise * 2) - self.obj_noise
                obj_poses.append(obj_obs + obj_noise)
            
            # Stack all object poses together
            obj_poses = torch.stack(obj_poses, dim=1)  # Shape: (num_envs, num_objects, 7)
            obs.update(
                obj_poses=obj_poses,
            )
        
        return obs
    
    def evaluate(self):
        """Evaluates the current state of the environment to determine task completion.
        
        This method checks if all objects are placed at their goal positions and the robot is static.
        
        Returns:
            dict: A dictionary containing boolean values for:
                - success: True if all conditions are met for all objects
                - is_obj_placed: Boolean tensor indicating if each object is at its goal
                - is_robot_static: True if robot is not moving
                - is_grasped: Boolean tensor indicating if each object is grasped
        """
        num_objects = len(self.obj_meshes)
        
        # Check if objects are at their goal positions
        is_obj_placed = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        is_grasped = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        
        for i, obj_mesh in enumerate(self.obj_meshes):
            # Check if this object is at its goal position
            is_obj_placed[:, i] = (
                torch.linalg.norm(self.goal_xyz[:, i] - obj_mesh.pose.p, axis=1)
                <= self.goal_thresh
            )
            
            # Check if this object is being grasped
            is_grasped[:, i] = self.agent.is_grasping(obj_mesh)
        
        # Visualize TCP if needed
        if self.visualize:
            self.tcp_site.set_pose(self.agent.tcp.pose)
        
        # Check if robot is static
        is_robot_static = self.agent.is_static(0.2)
        
        # Success requires all objects to be placed and robot to be static
        all_objects_placed = torch.all(is_obj_placed, dim=1)
        
        return {
            "success": all_objects_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        This reward function is designed to encourage the robot to move objects towards their goal positions
        while also tracking the waypoints along the way.

        The reward function is composed of three main components:
        1. Reaching reward - encourage moving TCP towards the closest object
        2. Waypoint tracking rewards - encourage moving objects along their waypoints
        3. Static reward - encourage being still when objects are at their goals
        """
        info['extra_data'] = {}
        reward = torch.zeros(self.num_envs, device=self.device)
        num_objects = len(self.obj_meshes)
        
        # 1. Reaching reward - encourage moving TCP towards closest object
        tcp_to_obj_dists = []
        for obj_mesh in self.obj_meshes:
            dist = torch.linalg.norm(
                obj_mesh.pose.p - self.agent.tcp.pose.p, axis=1
            )
            tcp_to_obj_dists.append(dist)
        
        # Find the closest object to the TCP
        tcp_to_obj_dists = torch.stack(tcp_to_obj_dists, dim=1)  # Shape: (num_envs, num_objects)
        closest_obj_dist, closest_obj_idx = torch.min(tcp_to_obj_dists, dim=1)
        
        # Compute reaching reward based on distance to closest object
        reaching_reward = 1 - torch.tanh(3 * closest_obj_dist)
        reaching_reward += 1 - torch.tanh(30 * closest_obj_dist)  # fine grained amount that only really applies when super close to object
        reaching_reward /= 2
        info['extra_data']['reaching_reward'] = reaching_reward
        reward += reaching_reward
        
        # 2. Waypoint tracking rewards for each object
        for i, obj_mesh in enumerate(self.obj_meshes):
            is_obj_grasped = info["is_grasped"][:, i]
            
            # Add grasping reward for this object
            obj_grasping_reward = is_obj_grasped * reaching_reward
            info['extra_data'][f'is_grasped_reward_{i}'] = obj_grasping_reward
            reward += obj_grasping_reward
            
            # Calculate distances and angles to all waypoints for this object
            waypoint_distances = torch.stack([
                torch.linalg.norm(waypoint[:3] - obj_mesh.pose.p, axis=1)
                for waypoint in self.waypoints[i]
            ], dim=1)
            
            waypoint_angles = torch.stack([
                torch.abs(quaternion_angle(waypoint[3:], obj_mesh.pose.q))
                for waypoint in self.waypoints[i]
            ], dim=1)
            
            # Update progress through waypoints for this object
            curr_idx_range = torch.arange(self.num_envs)
            obj_current_idx = self.current_subgoal_idx[:, i]
            
            reached_waypoint = waypoint_distances[curr_idx_range, obj_current_idx] < self.goal_thresh
            if self.rotation_reward:
                reached_waypoint &= (waypoint_angles[curr_idx_range, obj_current_idx] < self.angle_goal_thresh)
            
            self.current_subgoal_idx[:, i] = torch.clamp(
                obj_current_idx + reached_waypoint, 
                0, 
                len(self.waypoints[i]) - 1
            )
            
            # Compute waypoint reward based on current subgoal for this object
            obj_current_idx = self.current_subgoal_idx[:, i]
            dist_to_current = waypoint_distances[curr_idx_range, obj_current_idx]
            waypoint_reward = 1 - torch.tanh(
                self.waypoint_dist_reward_scaling[i][obj_current_idx] * dist_to_current
            )
            
            if self.rotation_reward:
                angle_to_current = waypoint_angles[curr_idx_range, obj_current_idx]
                waypoint_reward += (1 - torch.tanh(angle_to_current))
            
            info['extra_data'][f'waypoint_reward_{i}'] = waypoint_reward
            reward += waypoint_reward * is_obj_grasped + obj_current_idx.float()
        
        # 3. Static reward - encourage being still when all objects are at their goals
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        info['extra_data']['static_reward'] = static_reward
        
        # Add static reward based on how many objects are placed
        all_placed = torch.all(info["is_obj_placed"], dim=1)
        reward += static_reward * all_placed
        
        # Large reward for success
        reward[info["success"]] = 10
        
        return reward
    
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # Calculate normalization factor based on number of waypoints and objects
        num_objects = len(self.obj_meshes)
        total_waypoints = sum(len(wp) for wp in self.waypoints)
        normalization_factor = 5 + total_waypoints
        
        return self.compute_dense_reward(obs=obs, action=action, info=info) / normalization_factor