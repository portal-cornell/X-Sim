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
from scipy.spatial.transform import Rotation as R

# Import metadata loader
from mani_skill.envs.utils.metadata_loader import get_env_to_data_map, get_task_to_data_map, get_robot_pos


@register_env("Corn-in-Basket", max_episode_steps=300)
class CornInBasketEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_ninja", "panda_ninja_slow", "fetch"]
    agent: Union[Panda, Fetch]
    cube_half_size = 0.025

    def __init__(
        self,
        *args,
        robot_init_qpos_noise=0.02,
        randomize_init_config=False,
        obj_noise=0.0,
        environment_name="kitchen_env",
        task_name="corn_in_basket",
        visualize=False,
        randomize_robot_pos=False,
        **kwargs,
    ):
        # Load metadata from npz files
        self.env_to_data_map = get_env_to_data_map(task_name, environment_name)
        self.task_to_data_map = get_task_to_data_map(task_name, environment_name)
        self._robot_pos_offset = get_robot_pos(task_name, environment_name)
        
        # Validate environment name
        self.environment_name = environment_name
        if self.environment_name not in self.env_to_data_map:
            raise ValueError(f"Environment '{environment_name}' not found. Available: {list(self.env_to_data_map.keys())}")
        
        # Validate task name
        self.task_name = task_name
        if self.task_name not in self.task_to_data_map:
            raise ValueError(f"Task '{task_name}' not found. Available: {list(self.task_to_data_map.keys())}")
        
        # Store configurations
        self.env_config = self.env_to_data_map[self.environment_name]
        self.task_config = self.task_to_data_map[self.task_name]
        
        # Basic parameters
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize_init_config = randomize_init_config
        self.obj_noise = obj_noise
        self.visualize = visualize
        self.randomize_robot_pos = randomize_robot_pos
        
        # Task parameters from configuration
        self.robot_uids = self.task_config['robot_uids']
        self.obj_names = self.task_config['obj_names']
        self.obj_names = self.obj_names if isinstance(self.obj_names, list) else [self.obj_names]
        self.manip_idx = self.task_config['manip_idx']
        self.randomize_objects_list = self.task_config['randomize_objects_list']
        self.demo_name = self.task_config['demo_name']
        self.num_waypoints = self.task_config['num_waypoints']
        self.goal_thresh = self.task_config['goal_thresh']
        self.angle_goal_thresh = self.task_config['angle_goal_thresh']
        self.rotation_reward = self.task_config['rotation_reward']
        self.require_grasp = self.task_config['require_grasp']
        self.fixed_rotation = self.task_config['fixed_rotation']
        
        # Randomization parameters (kept as variables in the file)
        self.xy_rand = 0.025
        self.rot_rand = np.pi / 8
        
        # Object configuration
        asset_base_path = self.env_config['asset_base_path']
        self.obj_files = [f"{PACKAGE_ASSET_DIR}/{asset_base_path}/{obj_name}/mesh/{obj_name}.obj" 
                         for obj_name in self.obj_names]

        # Kitchen configuration
        self.kitchen_file = f"{PACKAGE_ASSET_DIR}/{asset_base_path}/{self.env_config['kitchen_mesh_name']}.obj"
        self.kitchen_to_robot_transform = torch.tensor(self.env_config['kitchen_to_robot_transform'])

        # Demo and waypoint configuration
        self.demo_paths = [f"{PACKAGE_ASSET_DIR}/{asset_base_path}/{obj_name}/demos/{self.demo_name}.npy"
                          for obj_name in self.obj_names]

        # Robot configuration
        self.robot_qpos = self.env_config['robot_qpos']
        
        # Waypoint configuration
        self.start_idx = self.task_config['first_waypoint_idx']
        self.end_idx = self.task_config['last_waypoint_idx']

        super().__init__(*args, robot_uids=self.robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig(
            uid="base_camera", 
            pose=pose, 
            width=128, 
            height=128, 
            near=0.01, 
            far=100, 
            intrinsic=[
                [5.331700e+02, 0.000000e+00, 4.862300e+02],
                [0.000000e+00, 5.331000e+02, 2.614790e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00],
            ]
        )]

    @property
    def _default_human_render_camera_configs(self):
        pose_new = sapien_utils.look_at(
            self.env_config['camera']['eye'],
            self.env_config['camera']['target']
        )
        return CameraConfig(
            uid="render_camera",  
            pose=pose_new, 
            width=960, 
            height=540,
            near=0.01, 
            far=100, 
            intrinsic=[
                [5.331700e+02, 0.000000e+00, 4.862300e+02],
                [0.000000e+00, 5.331000e+02, 2.614790e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00],
            ],
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
        """Load the kitchen mesh based on environment configuration."""
        builder = scene.create_actor_builder()
        density = 1000
        builder.add_multiple_convex_collisions_from_file(
            filename=self.kitchen_file,
            material=None,
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
        
        builder.add_visual_from_file(filename=self.kitchen_file)
        builder.initial_pose = sapien.Pose()
        mesh = builder.build_kinematic(name="kitchen")
        
        return mesh

    def _load_obj(self, scene, obj_file, name="obj", kinematic=False):
        """Load an object mesh."""
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
        if kinematic or name == "basket":
            obj = builder.build_kinematic(name=name)
        else:
            obj = builder.build(name=name)
        return obj
    
    def _load_scene(self, options: dict):
        # Load ground and kitchen
        self.ground = build_ground(
            self.scene, 
            altitude=self.env_config["ground_altitude"], 
            ground_color=[0.5, 0.5, 0.5, 1]
        )
        self.kitchen_mesh = self._load_kitchen(self.scene)
        
        # Load demo data and set up waypoints
        self._load_demo_data()
        self._setup_waypoints()
        self._create_visualization_sites()
        self._initialize_tracking_variables()

    def _load_demo_data(self):
        """Load demonstration data for all objects."""
        self.poses = []
        self.start_poses = []
        self.goal_poses = []
        self.obj_meshes = []
        
        for i, obj_name in enumerate(self.obj_names):
            obj_file = self.obj_files[i]
            demo_path = self.demo_paths[i]
            
            # Load the object mesh
            obj_mesh = self._load_obj(
                self.scene, 
                obj_file, 
                name=obj_name, 
                kinematic=(i != self.manip_idx)
            )
            self.obj_meshes.append(obj_mesh)
            
            # Load the demo data
            pose_data = torch.tensor(np.load(demo_path)).float().to(self.device)[i]
            if self.fixed_rotation:
                pose_data[:,3:] = pose_data[0,3:]
            self.poses.append(pose_data)
            
            # Store start and goal poses
            self.start_poses.append(pose_data[0])
            self.goal_poses.append(pose_data[-1])

        # Set default waypoint indices if not provided
        if self.start_idx is None:
            self.start_idx = 0
        if self.end_idx is None:
            self.end_idx = len(self.poses[0]) - 1

    def _setup_waypoints(self):
        """Setup waypoints and reward scaling for all objects."""
        self.interval = (self.end_idx - self.start_idx) // self.num_waypoints
        self.waypoints = []
        self.waypoint_dist_reward_scaling = []
        self.waypoint_angle_reward_scaling = []
        
        for i, pose_data in enumerate(self.poses):
            # Select waypoints for this object
            obj_waypoints = []
            obj_waypoints.append(pose_data[0])  # Start pose
            
            for j in range(self.start_idx, self.end_idx - self.interval, self.interval):
                obj_waypoints.append(pose_data[j])
            obj_waypoints.append(pose_data[self.end_idx])  # Goal pose
            
            # Calculate distance reward scaling
            obj_scaling = torch.tensor([
                1 / (torch.norm(obj_waypoints[j][:3] - obj_waypoints[j+1][:3], p=2).item())
                for j in range(len(obj_waypoints) - 1)
            ]).to(self.device)
            self.waypoint_dist_reward_scaling.append(obj_scaling)

            # Calculate angular reward scaling
            obj_angle_scaling = torch.tensor([
                1 / (torch.abs(quaternion_angle(
                    obj_waypoints[j][3:], 
                    obj_waypoints[j+1][3:]
                )).item() + 1e-6)
                for j in range(len(obj_waypoints) - 1)
            ]).to(self.device)
            # Prevent extremely large values
            obj_angle_scaling = torch.clamp(obj_angle_scaling, max=10.0)
            self.waypoint_angle_reward_scaling.append(obj_angle_scaling)
            
            # Remove the first waypoint (starting position) for tracking
            obj_waypoints.pop(0)
            self.waypoints.append(obj_waypoints)

    def _create_visualization_sites(self):
        """Create visualization sites for waypoints and goals."""
        # Create waypoint visualization sites
        if self.visualize:
            self.waypoint_sites = []
            for i, obj_waypoints in enumerate(self.waypoints):
                obj_sites = []
                for j, waypoint in enumerate(obj_waypoints):
                    site_name = f"{self.obj_names[i]}_waypoint_{j}"
                    waypoint_site = self._load_obj(
                        self.scene, 
                        self.obj_files[i], 
                        name=site_name, 
                        kinematic=True
                    )
                    waypoint_site.set_pose(Pose.create_from_pq(p=waypoint[:3], q=waypoint[3:]))
                    obj_sites.append(waypoint_site)
                self.waypoint_sites.append(obj_sites)
        
        # Create goal visualization sites
        self.goal_sites = []
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

    def _initialize_tracking_variables(self):
        """Initialize tracking variables for goals and waypoints."""
        num_objects = len(self.obj_names)
        
        # Initialize goal quaternions and positions
        self.goal_quat = torch.zeros((self.num_envs, num_objects, 4), device=self.device)
        self.goal_xyz = torch.zeros((self.num_envs, num_objects, 3), device=self.device)
        
        for i in range(num_objects):
            self.goal_quat[:, i] = self.start_poses[i][3:7] 
            self.goal_xyz[:, i, :3] = self.goal_poses[i][:3]
        
        # Initialize tracking indices
        self.current_subgoal_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.current_obj_to_manip_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            # Set up kitchen position
            kitchen_xyz = torch.zeros((b, 3))
            kitchen_xyz[:, :3] = self.kitchen_to_robot_transform
            
            # Set kitchen pose
            kitchen_rot = transforms3d.quaternions.axangle2quat(
                np.array([1, 0, 0]), theta=np.deg2rad(90)
            )
            self.kitchen_mesh.set_pose(Pose.create_from_pq(p=kitchen_xyz, q=kitchen_rot))
            
            # Initialize each object
            self._initialize_objects(env_idx, b)
            
            # Reset tracking indices
            self.current_subgoal_idx[env_idx] = 0
            self.current_obj_to_manip_idx[env_idx] = self.manip_idx
            
            # Set robot initial configuration
            self._initialize_robot(env_idx, b)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int):
        """Initialize object positions and orientations."""
        for i, obj_mesh in enumerate(self.obj_meshes):
            # Set initial position
            xyz = torch.zeros((b, 3))
            xyz[:, :3] = self.start_poses[i][:3]
            
            # Apply randomization if needed
            if self.randomize_init_config and i in self.randomize_objects_list:
                xyz[:, :2] += torch.rand((b, 2)) * (self.xy_rand * 2) - self.xy_rand
            
            # Set initial rotation
            quat = torch.zeros((b, 4))
            quat[:, ] = self.start_poses[i][3:7]
            qs = torch.zeros((b, 4))
            qs[:, 0] = 1
            
            if self.randomize_init_config and i in self.randomize_objects_list:
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, bounds=(-self.rot_rand, self.rot_rand)
                )
            
            fixed_rot = Pose.create_from_pq(q=qs) * Pose.create_from_pq(q=quat)
            
            # Set object pose
            obj_mesh.set_pose(Pose.create_from_pq(p=xyz, q=fixed_rot.q))
            
            # Update goal quaternion
            self.goal_quat[env_idx, i] = self.goal_poses[i][3:7]
            
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
                    self.waypoint_sites[i][j].set_pose(Pose.create_from_pq(p=wp_xyz, q=wp_quat))

    def _initialize_robot(self, env_idx: torch.Tensor, b: int):
        """Initialize robot configuration."""
        qpos = np.array(self.robot_qpos)
        
        if self._enhanced_determinism:
            qpos = (
                self._batched_episode_rng[env_idx].normal(
                    0, self.robot_init_qpos_noise, len(qpos)
                ) + qpos
            )
        else:
            qpos = (
                self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                ) + qpos
            )
        
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        
        # Use robot position offset from metadata
        robot_pos = np.zeros((b, 3))
        robot_pos += np.array(self._robot_pos_offset)  # Apply offset from metadata
        if self.randomize_robot_pos:
            robot_pos += np.random.rand(b, 3) * 0.04 - 0.02
        self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))
    
    def _get_obs_agent(self):
        return dict()
    
    def _get_obs_extra(self, info: Dict):
        obs = {
            "ee_pose": self.agent.tcp.pose.raw_pose,
            "gripper_width": self.agent.robot.get_qpos()[:, -1:],
        }

        if self.randomize_robot_pos:
            obs["robot_pos"] = self.agent.robot.pose.p
        
        if "state" in self.obs_mode:
            self._add_state_observations(obs)

        if self.require_grasp:
            obs['is_grasped'] = info['is_grasped']

        return obs

    def _add_state_observations(self, obs: Dict):
        """Add state-based observations for the manipulated object."""
        obj_poses_list = []
        desired_goals_list = []
        goal_distances_list = []
        goal_angle_diffs_list = []
        
        i = self.manip_idx
        obj_mesh = self.obj_meshes[i]
        
        # Current object pose with noise
        obj_obs = obj_mesh.pose.raw_pose
        obj_noise = torch.rand_like(obj_obs) * (self.obj_noise * 2) - self.obj_noise
        obj_poses = obj_obs + obj_noise
        obj_poses[:, 3:7] = obj_poses[:, 3:7]
        obj_poses_list.append(obj_poses)

        # Current desired waypoint pose
        waypoints_obj_tensor = torch.stack(self.waypoints[i])
        indices_obj = self.current_subgoal_idx
        desired_goal_obj = waypoints_obj_tensor[indices_obj]
        desired_goal_obj[:, 3:7] = desired_goal_obj[:, 3:7]
        desired_goals_list.append(desired_goal_obj)

        # Compute position and rotation differences
        pos_diff = desired_goal_obj[:, :3] - obj_poses[:, :3]
        goal_distances_list.append(pos_diff)
        
        angle_diff = torch.abs(quaternion_angle(
            desired_goal_obj[:, 3:7], 
            obj_poses[:, 3:7]
        )).unsqueeze(-1)
        goal_angle_diffs_list.append(angle_diff)
        
        # Update observations
        obs.update(
            achieved_goal=torch.cat(obj_poses_list, dim=1),
            desired_goal=torch.cat(desired_goals_list, dim=1),
            goal_position_diff=torch.cat(goal_distances_list, dim=1), 
            goal_rotation_diff=torch.cat(goal_angle_diffs_list, dim=1),
        )
    
    def evaluate(self):
        """Evaluate task completion status."""
        num_objects = len(self.obj_meshes)
        is_obj_placed = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        is_obj_rotated = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        is_grasped = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        
        for i, obj_mesh in enumerate(self.obj_meshes):
            # Check position (XY only)
            is_obj_placed[:, i] = (
                torch.linalg.norm(self.goal_xyz[:, i, :3] - obj_mesh.pose.p[:, :3], axis=1) <= self.goal_thresh
            )
            
            # Check rotation
            is_obj_rotated[:, i] = (
                torch.abs(quaternion_angle(self.goal_quat[:, i], obj_mesh.pose.q)) <= self.angle_goal_thresh
            ) 
        
            # Check grasp
            is_grasped[:, i] = self.agent.is_grasping(obj_mesh)
        
        # Visualize TCP if needed
        if self.visualize:
            self.tcp_site.set_pose(self.agent.tcp.pose)
        
        # Check robot static state
        is_robot_static = self.agent.is_static(0.1)
        
        # Success conditions
        all_objects_placed = torch.all(is_obj_placed, dim=1)
        all_objects_rotated = torch.all(is_obj_rotated, dim=1)

        return {
            "success": all_objects_placed & all_objects_rotated & is_robot_static,
            "all_obj_placed": all_objects_placed,
            "all_obj_rotated": all_objects_rotated,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Compute dense reward with three main components:
        1. Reaching reward - encourage moving TCP towards the target object
        2. Waypoint tracking rewards - encourage moving objects along waypoints
        3. Static reward - encourage being still when objects are at goals
        """
        info['extra_data'] = {}
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Reaching reward
        reward += self._compute_reaching_reward(info)
        
        # 2. Waypoint tracking reward
        reward += self._compute_waypoint_reward(info)
        
        # 3. Static reward
        reward += self._compute_static_reward(info)
        
        # Success bonus
        reward[info["success"]] += 1
        
        return reward

    def _compute_reaching_reward(self, info: Dict):
        """Compute reward for reaching the target object."""
        tcp_to_obj_dists = []
        for obj_mesh in self.obj_meshes:
            dist = torch.linalg.norm(obj_mesh.pose.p - self.agent.tcp.pose.p, axis=1)
            tcp_to_obj_dists.append(dist)
        
        tcp_to_obj_dists = torch.stack(tcp_to_obj_dists, dim=1)
        relevant_obj_dist = tcp_to_obj_dists[
            torch.arange(self.num_envs, device=self.device), 
            self.current_obj_to_manip_idx
        ]
        
        reaching_reward = 1 - torch.tanh(3 * relevant_obj_dist)
        reaching_reward += 1 - torch.tanh(30 * relevant_obj_dist)
        reaching_reward /= 2
        
        info['extra_data']['reaching_reward'] = reaching_reward
        
        if self.require_grasp:
            reaching_reward += info['is_grasped'][
                torch.arange(self.num_envs, device=self.device), 
                self.current_obj_to_manip_idx
            ]
        
        return reaching_reward

    def _compute_waypoint_reward(self, info: Dict):
        """Compute reward for waypoint tracking."""
        i = self.manip_idx
        obj_mesh = self.obj_meshes[i]
        
        # Calculate distances and angles to all waypoints
        waypoint_distances = torch.stack([
            torch.linalg.norm(waypoint[:3] - obj_mesh.pose.p[:, :3], axis=1)
            for waypoint in self.waypoints[i]
        ], dim=1)
        
        waypoint_angles = torch.stack([
            torch.abs(quaternion_angle(waypoint[3:], obj_mesh.pose.q))
            for waypoint in self.waypoints[i]
        ], dim=1)
        
        # Update waypoint progress
        curr_idx_range = torch.arange(self.num_envs)
        obj_current_idx = self.current_subgoal_idx
        
        reached_waypoint = waypoint_distances[curr_idx_range, obj_current_idx] < self.goal_thresh
        if self.rotation_reward:
            reached_waypoint &= (waypoint_angles[curr_idx_range, obj_current_idx] < self.angle_goal_thresh)
        
        # Compute waypoint reward
        dist_to_current = waypoint_distances[curr_idx_range, obj_current_idx]
        waypoint_reward = 1 - torch.tanh(
            self.waypoint_dist_reward_scaling[i][obj_current_idx] * dist_to_current
        )
        
        if self.rotation_reward:
            angle_to_current = waypoint_angles[curr_idx_range, obj_current_idx]
            angle_reward = 1 - torch.tanh(
                self.waypoint_angle_reward_scaling[i][obj_current_idx] * angle_to_current
            )
            waypoint_reward += angle_reward
        
        info['extra_data'][f'waypoint_reward_{i}'] = waypoint_reward
        
        # Progress bonus
        waypoint_reward += 2 * obj_current_idx.float()
        
        # Update subgoal index
        self.current_subgoal_idx = torch.clamp(
            self.current_subgoal_idx + reached_waypoint, 
            0, 
            len(self.waypoints[0]) - 1
        )
        
        return waypoint_reward

    def _compute_static_reward(self, info: Dict):
        """Compute reward for being static when task is complete."""
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        info['extra_data']['static_reward'] = static_reward
        
        # Apply static reward only when objects are properly placed
        all_placed = info["all_obj_placed"]
        all_rotated = info["all_obj_rotated"]
        
        return static_reward * all_placed * all_rotated
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute normalized dense reward."""
        total_waypoints = sum(len(wp) for wp in self.waypoints)
        normalization_factor = 5 + total_waypoints
        
        return self.compute_dense_reward(obs=obs, action=action, info=info) / normalization_factor
