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



@register_env("Corn-Basket-Env-New", max_episode_steps=300)
class CornBasketEnvNew(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "panda_ninja_slow", "fetch"]
    agent: Union[Panda, Fetch]
    cube_half_size = 0.025
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    env_to_data_map = {
            "tabletop": {
                "kitchen_to_robot_transform": [0.2060+0.012, 0.7336, 0.3448 - 0.01],
                "robot_qpos": [ 0.1534, -0.7669, -0.1598, -2.2560, -0.1310,  1.4803,  0.8555, 0.04, 0.04],
                "camera": {
                    "eye": [1.10358784, -0.09247672,  0.92705452],
                    "target": [0.53148106, 0.06763309, 0.19122654]
                },
                "ground_altitude": -0.7,
            },
            "flat_kitchen": {
                # "kitchen_to_robot_transform": [0.4547, 0.1753 - 0.01, 0.1875 - 0.01 + 0.02],
                "kitchen_to_robot_transform": [0.4547, 0.1753 - 0.01, 0.1875 - 0.01],
                "robot_qpos": [0.2611934, -0.4642156 ,  0.08966457, -2.3954022 ,  0.15271077, 2.0086708 ,  0.9597888, 0.04, 0.04],
                # "robot_qpos": [0.10474431, -0.4642156 ,  0.08966457, -2.3954022 ,  0.15271077, 2.0086708 ,  0.9597888, 0.04, 0.04],
                
                "camera": {
                    "eye": [-0.01686193, -0.39387421, 0.6638793],
                    "target": [0.81217533, 0.01006028, 0.27718283]
                },
                "ground_altitude": -0.6,
            }
        }

    def __init__(
        # self,
        # *args,
        # robot_uids="panda_ninja",
        # robot_init_qpos_noise=0.02,
        # randomize_init_config=False,
        # obj_noise=0.0,
        # environment_name="tabletop",
        # obj_names=['LetterA', 'LetterI'],  
        # pick_place_idx=1,
        # randomize_objects_list=[1],
        # demo_name='AI-correct',
        # num_waypoints=1,
        # first_waypoint_idx=14,
        # last_waypoint_idx=58,
        # goal_thresh = 0.013,
        # angle_goal_thresh = 0.10,
        # visualize=False,
        # rotation_reward=True,
        # require_grasp=False,
        # **kwargs,

        self,
        *args,
        robot_uids="panda_ninja_slow",
        robot_init_qpos_noise=0.02,
        randomize_init_config=False,
        obj_noise=0.0,
        environment_name="flat_kitchen",
        obj_names=['corn_new', 'basket_centered'],  
        pick_place_idx=0,
        randomize_objects_list=[0],
        demo_name='corn_in_basket_recalibrated',
        num_waypoints=5,
        first_waypoint_idx=70,
        last_waypoint_idx=170,
        goal_thresh = 0.03,
        angle_goal_thresh = 0.2,
        visualize=False,
        rotation_reward=True,
        require_grasp=True,
        **kwargs,
    ):
        self.environment_name = environment_name
        assert self.environment_name in ["tabletop", "flat_kitchen"]
        
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize_init_config = randomize_init_config

        self.obj_noise = obj_noise
        self.obj_names = obj_names if isinstance(obj_names, list) else [obj_names]  # Ensure it's a list
        self.obj_files = [f"{PACKAGE_ASSET_DIR}/portal_lab/{self.environment_name}/{obj_name}/mesh/{obj_name}.obj" 
                        for obj_name in self.obj_names]
        self.single_obj_pick_place_idx = pick_place_idx

        self.kitchen_file = f"{PACKAGE_ASSET_DIR}/portal_lab/{self.environment_name}/{'Kitchen' if self.environment_name == 'flat_kitchen' else 'tabletop'}.obj"
        self.kitchen_to_robot_transform = torch.tensor(self.env_to_data_map[self.environment_name]['kitchen_to_robot_transform'])

        self.visualize = visualize
        self.rotation_reward = rotation_reward

        self.demo_name = demo_name
        self.demo_paths = [f"{PACKAGE_ASSET_DIR}/portal_lab/{self.environment_name}/{obj_name}/demos/{self.demo_name}.npy"
                          for obj_name in self.obj_names]
        self.goal_thresh = goal_thresh
        self.angle_goal_thresh = angle_goal_thresh
        self.randomize_objects_list = randomize_objects_list
        self.xy_rand = 0.025
        self.rot_rand = np.pi / 8

        self.robot_qpos = self.env_to_data_map[self.environment_name]['robot_qpos']
        self.start_idx = first_waypoint_idx
        self.end_idx = last_waypoint_idx
        self.num_waypoints = num_waypoints  # num waypoints, including goal (set to 1 for goal-conditioned)
        self.require_grasp = require_grasp
        self.randomize_robot_pos = False

        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # pose_new = sapien_utils.look_at([0.95304242 - 0.15, -0.07714972 + 0.1 - 0.5,  0.97090369 - 0.4], 
        #                             [0.49877198, 0.09168973, 0.09618567])
        # pose_new = sapien_utils.look_at([0.95304242, -0.07714972 + 0.1,  0.97090369], 
        #                             [0.49877198, 0.09168973, 0.09618567])
        # pose_new = sapien_utils.look_at([0.2060 + 0.35, -0.7336, 0.3448 + 0.245], 
        #                             [0.2060 + 0.35, 0.7336, 0.3448 + 0.045])        
        pose_new = sapien_utils.look_at(self.env_to_data_map[self.environment_name]['camera']['eye'],
                                        self.env_to_data_map[self.environment_name]['camera']['target']) 
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
        builder = scene.create_actor_builder()
        density = 1000
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_file,
            material=None,
            density=density,
            decomposition="coacd",
            # decomposition_params={
            #     "threshold": 0.01,
            # }
        )
        builder.add_visual_from_file(filename=obj_file)
        builder.initial_pose = sapien.Pose()
        if kinematic or name == "basket_centered":
            obj = builder.build_kinematic(name=name)
        else:
            obj = builder.build(name=name)
        return obj
    
    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, altitude=self.env_to_data_map[self.environment_name]["ground_altitude"], ground_color=[0.5,0.5,0.5,1])
        self.kitchen_mesh = self._load_kitchen(self.scene)
        
        # Load all objects and their demo data
        self.interval = (self.end_idx - self.start_idx) // self.num_waypoints
        self.poses = []
        self.waypoints = []
        self.waypoint_dist_reward_scaling = []
        self.waypoint_angle_reward_scaling = []
        self.start_poses = []
        self.goal_poses = []
        self.obj_meshes = []
        self.goal_sites = []
        for i, obj_name in enumerate(self.obj_names):
            obj_file = self.obj_files[i]
            demo_path = self.demo_paths[i]
            # Load the object mesh

            obj_mesh = self.load_obj(self.scene, obj_file, name=obj_name, kinematic=(i != self.single_obj_pick_place_idx))
            self.obj_meshes.append(obj_mesh)
            
            # Load the demo data
            pose_data = torch.tensor(np.load(demo_path)).float().to(self.device)[i]  # Assumes just 1 object at index 0
            # pose_data[:, 2] += 0.02
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

            
            # Calculate angular differences between consecutive waypoints
            obj_angle_scaling = torch.tensor([
                1 / (torch.abs(quaternion_angle(obj_waypoints[j][3:], 
                                            obj_waypoints[j+1][3:])).item() + 1e-6)
                for j in range(len(obj_waypoints) - 1)
            ]).to(self.device)
            # Handle case where consecutive rotations are identical (angle = 0)
            obj_angle_scaling = torch.clamp(obj_angle_scaling, max=10.0)  # Prevent extremely large values
            self.waypoint_angle_reward_scaling.append(obj_angle_scaling)
            
            # Remove the first waypoint (starting position)
            obj_waypoints.pop(0)
            self.waypoints.append(obj_waypoints)
            
            # Store start and goal poses
            self.start_poses.append(pose_data[0])
            self.goal_poses.append(pose_data[-1])
            
        
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
            self.goal_quat[:, i] = self.start_poses[i][3:7] 
        
        self.current_subgoal_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.current_obj_to_manip_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        
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
        
        self.interval = (self.end_idx - self.start_idx) // self.num_waypoints

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
            for i, obj_mesh in enumerate(self.obj_meshes):
                # Set initial position
                xyz = torch.zeros((b, 3))
                xyz[:, :3] = self.start_poses[i][:3]
                
                # Apply randomization if needed
                if self.randomize_init_config and i in self.randomize_objects_list:
                    xyz[:, :2] += torch.rand((b, 2)) * (self.xy_rand*2) - (self.xy_rand)
                
                # Set initial rotation
                quat = torch.zeros((b, 4))
                quat[:, ] = self.start_poses[i][3:7]
                qs = torch.zeros((b, 4))
                qs[:, 0] = 1
                
                if self.randomize_init_config and i in self.randomize_objects_list:
                    qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, bounds=(-self.rot_rand, self.rot_rand))
                
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
            
            # Reset subgoal tracking indices
            self.current_subgoal_idx[env_idx] = 0

            # HARD CODED TO OBJ AT INDEX 
            self.current_obj_to_manip_idx[env_idx] = self.single_obj_pick_place_idx
            
            # Set robot initial configuration
            qpos = np.array(
                self.robot_qpos
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
            robot_pos = np.zeros((b, 3))
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
            obj_poses_list = []
            desired_goals_list = []
            goal_distances_list = []  # New list for position differences
            goal_angle_diffs_list = []  # New list for rotation differences
            
            i, obj_mesh = self.single_obj_pick_place_idx, self.obj_meshes[self.single_obj_pick_place_idx]
            # for i, obj_mesh in enumerate(self.obj_meshes):
            # Current object pose
            obj_obs = obj_mesh.pose.raw_pose
            obj_noise = torch.rand_like(obj_obs) * (self.obj_noise * 2) - self.obj_noise
            obj_poses = obj_obs + obj_noise
            obj_poses[:, 3:7] = obj_poses[:, 3:7]
            obj_poses_list.append(obj_poses)

            # Current desired waypoint pose for this object
            waypoints_obj_tensor = torch.stack(self.waypoints[i])  # (num_waypoints, 7)
            indices_obj = self.current_subgoal_idx  # (num_envs,)
            desired_goal_obj = waypoints_obj_tensor[indices_obj]  # (num_envs, 7)
            desired_goal_obj[:, 3:7] = desired_goal_obj[:, 3:7]
            desired_goals_list.append(desired_goal_obj)

            # Compute position difference (XY distance)
            pos_diff = desired_goal_obj[:, :3] - obj_poses[:, :3]  # (num_envs, 3)
            goal_distances_list.append(pos_diff)
            
            # Compute rotation difference (angle between quaternions)
            angle_diff = torch.abs(quaternion_angle(
                desired_goal_obj[:, 3:7], 
                obj_poses[:, 3:7]
            )).unsqueeze(-1)  # (num_envs, 1)
            goal_angle_diffs_list.append(angle_diff)
            
            # Concatenate all object poses and goals
            obj_poses_concat = torch.cat(obj_poses_list, dim=1)  # (num_envs, num_objects * 7)
            desired_goals_concat = torch.cat(desired_goals_list, dim=1)  # (num_envs, num_objects * 7)
            
            # Concatenate all position and rotation differences
            goal_distances_concat = torch.cat(goal_distances_list, dim=1)  # (num_envs, num_objects * 3)
            goal_angle_diffs_concat = torch.cat(goal_angle_diffs_list, dim=1)  # (num_envs, num_objects * 1)
            
            obs.update(
                achieved_goal=obj_poses_concat,
                desired_goal=desired_goals_concat,
                goal_position_diff=goal_distances_concat, 
                goal_rotation_diff=goal_angle_diffs_concat,
            )

        # obs["all_obj_placed"] = info["all_obj_placed"]
        # obs["all_obj_rotated"] = info["all_obj_rotated"]
        # obs["is_robot_static"] = info["is_robot_static"]

        if self.require_grasp:
            obs['is_grasped'] = info['is_grasped']

        return obs
    
    # def step(self, action):
    #     action[..., -1] = torch.clamp(action[..., -1], min=0.3)
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     return obs, reward, terminated, truncated, info
    
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
        is_obj_rotated = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        is_grasped = torch.zeros((self.num_envs, num_objects), dtype=torch.bool, device=self.device)
        
        for i, obj_mesh in enumerate(self.obj_meshes):
            # Check if this object is at its goal position ONLY CHECK XY POSITION
            is_obj_placed[:, i] = (
                torch.linalg.norm(self.goal_xyz[:, i, :3] - obj_mesh.pose.p[:, :3], axis=1)
                <= self.goal_thresh * 2.5
            )
            
            is_obj_rotated[:, i] = (
                torch.abs(quaternion_angle(self.goal_quat[:, i], obj_mesh.pose.q))
                    <= self.angle_goal_thresh * 15
                ) 
        
            # Check if this object is being grasped
            is_grasped[:, i] = self.agent.is_grasping(obj_mesh)
        
        # Visualize TCP if needed
        if self.visualize:
            self.tcp_site.set_pose(self.agent.tcp.pose)
        
        # Check if robot is static
        is_robot_static = self.agent.is_static(0.15)
        
        # Success requires all objects to be placed and robot to be static
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
        This reward function is designed to encourage the robot to move objects towards their goal positions
        while also tracking the waypoints along the way.

        The reward function is composed of three main components:
        1. Reaching reward - encourage moving TCP towards the closest object
        2. Waypoint tracking rewards - encourage moving objects along their waypoints
        3. Static reward - encourage being still when objects are at their goals
        """
        info['extra_data'] = {}
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Reaching reward - encourage moving TCP towards closest object
        tcp_to_obj_dists = []
        for obj_mesh in self.obj_meshes:
            dist = torch.linalg.norm(
                obj_mesh.pose.p - self.agent.tcp.pose.p, axis=1
            )
            tcp_to_obj_dists.append(dist)
        
        # Find the closest object to the TCP
        tcp_to_obj_dists = torch.stack(tcp_to_obj_dists, dim=1)  # Shape: (num_envs, num_objects)
        
        # Select the relevant object distance for each environment
        relevant_obj_dist = tcp_to_obj_dists[torch.arange(self.num_envs, device=self.device), self.current_obj_to_manip_idx]
    
        
        # Compute reaching reward based on distance to the relevant object
        reaching_reward = 1 - torch.tanh(3 * relevant_obj_dist)
        reaching_reward += 1 - torch.tanh(30 * relevant_obj_dist)  # fine-grained amount that only really applies when super close to object
        reaching_reward /= 2
        info['extra_data']['reaching_reward'] = reaching_reward
        reward += reaching_reward
        
        if self.require_grasp:
            reward += info['is_grasped'][torch.arange(self.num_envs, device=self.device), self.current_obj_to_manip_idx]
        
        all_obj_reached_waypoint = ~torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2. Waypoint tracking rewards for each object
        i, obj_mesh = self.single_obj_pick_place_idx, self.obj_meshes[self.single_obj_pick_place_idx]
        # for i, obj_mesh in enumerate(self.obj_meshes):
        # Calculate distances and angles to all waypoints for this object
        waypoint_distances = torch.stack([
            torch.linalg.norm(waypoint[:3] - obj_mesh.pose.p[:, :3], axis=1)
            for waypoint in self.waypoints[i]
        ], dim=1)
        
        waypoint_angles = torch.stack([
            torch.abs(quaternion_angle(waypoint[3:], obj_mesh.pose.q))
            for waypoint in self.waypoints[i]
        ], dim=1)
        
        # Update progress through waypoints for this object
        curr_idx_range = torch.arange(self.num_envs)
        obj_current_idx = self.current_subgoal_idx
        
        reached_waypoint = waypoint_distances[curr_idx_range, obj_current_idx] < self.goal_thresh
        if self.rotation_reward:
            reached_waypoint &= (waypoint_angles[curr_idx_range, obj_current_idx] < self.angle_goal_thresh)
        
        all_obj_reached_waypoint &= reached_waypoint

        # Compute waypoint reward based on current subgoal for this object
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
        # if self.require_grasp:
        #     reward += waypoint_reward * (self.current_obj_to_manip_idx != i | info['is_grasped'][torch.arange(self.num_envs, device=self.device), self.current_obj_to_manip_idx]) + obj_current_idx.float()
        # else:
        #     reward += waypoint_reward + obj_current_idx.float()
        reward += waypoint_reward + 2 * obj_current_idx.float()


        self.current_subgoal_idx = torch.clamp(
                self.current_subgoal_idx + all_obj_reached_waypoint, 
                0, 
                len(self.waypoints[0]) - 1
            )
        
        # 3. Static reward - encourage being still when all objects are at their goals
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        info['extra_data']['static_reward'] = static_reward
        
        # Add static reward based on how many objects are placed
        all_placed = info["all_obj_placed"]
        all_rotated = info["all_obj_rotated"]
        reward += static_reward * all_placed * all_rotated
        
        # Large reward for success
        reward[info["success"]] += 1
        
        return reward
    
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # Calculate normalization factor based on number of waypoints and objects
        total_waypoints = sum(len(wp) for wp in self.waypoints)
        normalization_factor = 5 + total_waypoints
        
        return self.compute_dense_reward(obs=obs, action=action, info=info) / normalization_factor
