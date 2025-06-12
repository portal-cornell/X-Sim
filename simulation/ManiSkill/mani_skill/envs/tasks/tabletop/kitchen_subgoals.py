from typing import Any, Dict, Union

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


@register_env("Kitchen-Subgoals", max_episode_steps=300)
class KitchenSubgoalsEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "panda_ninja", "fetch"]
    agent: Union[Panda, Fetch]
    cube_half_size = 0.025
    goal_thresh = 0.01
    angle_goal_thresh = 0.15 # 8 degrees
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5
    z_rand_amt = 0.0

    def __init__(
        self,
        *args,
        robot_uids="panda_ninja",
        robot_init_qpos_noise=0.02,
        randomize_init_config=False,
        obj_noise=0.0,
        obj_name='mustard_centered',
        demo_name='mustard_rl',
        num_waypoints=5,
        visualize=False,
        rotation_reward=True,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize_init_config = randomize_init_config

        self.obj_noise = obj_noise
        self.obj_name = obj_name
        self.obj_file = f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/{self.obj_name}/mesh/{self.obj_name}.obj"

        self.kitchen_file = f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/Kitchen.obj"
        self.kitchen_to_robot_transform = torch.tensor([0.4147, 0.1753 - 0.01, 0.1875 - 0.01])

        self.visualize = visualize
        self.rotation_reward = rotation_reward

        self.demo_name = demo_name
        self.demo_path = f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/{self.obj_name}/demos/{self.demo_name}.npy"
        self.waypoints = []
        self.poses = None
        self.start_pose = None
        self.goal_pose = None
        self.goal_xyz = None
        self.goal_quat = None
        self.current_subgoal_idx = None
        self.start_idx = 89 # index of the first waypoint we want to compute reward w.r.t.
        self.end_idx = 205 # index of the goal destination
        self.num_waypoints = num_waypoints # num waypoints, including goal (set to 1 for goal-conditioned)
        self.interval = None
        self.waypoint_sites = []


        self.ground = None
        self.kitchen_mesh = None
        self.obj_mesh = None
        self.goal_site = None
        self.tcp_site = None
        
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

    def load_obj(self, scene, name = "cube", kinematic = False):
        builder = scene.create_actor_builder()
        density = 1000
        builder.add_multiple_convex_collisions_from_file(
            filename=self.obj_file,
            material=None,
            density=density,
            decomposition="coacd",
        )
        builder.add_visual_from_file(filename=self.obj_file)
        builder.initial_pose = sapien.Pose()
        if kinematic:
            cube = builder.build_kinematic(name=name)
        else:
            cube = builder.build(name=name)
        return cube
    
    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, altitude=-0.6)
        self.kitchen_mesh = self._load_kitchen(self.scene)
        self.obj_mesh = self.load_obj(self.scene)
        self.init_robot_pose = torch.zeros((self.num_envs, 3)).to(self.device)


        self.poses = torch.tensor(np.load(self.demo_path)).float().to(self.device)[0] # Assumes just 1 object at index 0
        
        # # Add 5cm to z-coordinate of all poses
        # self.poses[1:-1, 2] += 0.02
        
        # select "num_waypoints" number of waypoints between start and end idxs by interpolating
        self.interval = (self.end_idx - self.start_idx) // self.num_waypoints
        self.waypoints = []
        self.waypoints.append(self.poses[0])
        for i in range(self.start_idx, self.end_idx - self.interval, self.interval):
            self.waypoints.append(self.poses[i])
        self.waypoints.append(self.poses[self.end_idx])
        self.waypoint_dist_reward_scaling = torch.tensor([
            1 / (torch.norm(self.waypoints[i][:3] - self.waypoints[i+1][:3], p=2).item() + 1e-5)
            for i in range(len(self.waypoints) - 1)
        ]).to(self.device)
        self.waypoints.pop(0)

        if self.visualize:
            self.waypoint_sites = []
            for i, waypoint in enumerate(self.waypoints):
                waypoint_site = self.load_obj(self.scene, name = f"Waypoint {i}", kinematic=True)
                waypoint_site.set_pose(Pose.create_from_pq(p=waypoint[:3], q = waypoint[3:]))
                self.waypoint_sites.append(waypoint_site)
        
        self.waypoints = torch.tensor([waypoint.cpu().tolist() for waypoint in self.waypoints]).to(self.device)
        
        self.start_pose = self.poses[0]
        self.goal_pose = self.poses[-1]
        self.current_subgoal_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        goal_xyz = torch.zeros((self.num_envs, 3), device=self.device)
        goal_xyz[:, :3] += self.goal_pose[:3]
        self.goal_xyz = goal_xyz

        goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        goal_quat[:, ] = self.waypoints[-1][3:7]
        self.goal_quat = goal_quat

        self.goal_site = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size / 2,
            color=[0, 1, 0, 0.5],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :3] = self.start_pose[:3]
            if self.randomize_init_config:
                xyz[:, :2] += torch.rand((b, 2)) * 0.05 - 0.025

            quat = torch.zeros((b, 4))
            quat[:, ] = self.start_pose[3:7]
            qs = torch.zeros((b, 4))
            qs[:, 0] = 1
            if self.randomize_init_config:
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, bounds=(0, np.pi/6))
            fixed_rot = Pose.create_from_pq(q=qs) * Pose.create_from_pq(q=quat)

            kitchen_xyz = torch.zeros((b, 3))
            kitchen_xyz[:, :3] = self.kitchen_to_robot_transform
            
            self.goal_xyz[env_idx, :3] = self.goal_pose[:3]

            if self.randomize_init_config:
                z_random = torch.rand((b, 1)) * (self.z_rand_amt*2) - self.z_rand_amt
                kitchen_xyz[:, 2:3] += z_random
                xyz[:, 2:3] += z_random 
                self.goal_xyz[env_idx, 2:3] += z_random


            # self.obj_mesh.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            self.obj_mesh.set_pose(Pose.create_from_pq(p=xyz, q=fixed_rot.q))

            self.goal_quat[env_idx,] = self.waypoints[-1][3:7]

            if self.visualize:
                self.goal_site.set_pose(Pose.create_from_pq(p=self.goal_xyz[env_idx, :3], q=self.goal_quat[env_idx]))

            kitchen_rot = transforms3d.quaternions.axangle2quat(
                np.array([1, 0, 0]), theta=np.deg2rad(90)
            )
            self.kitchen_mesh.set_pose(Pose.create_from_pq(p=kitchen_xyz, q=kitchen_rot))

            if self.visualize:
                for i, waypoint in enumerate(self.waypoints):
                    xyz[:] = waypoint[:3]
                    quat[:] = waypoint[3:]
                    self.waypoint_sites[i].set_pose(Pose.create_from_pq(p=xyz, q=quat))


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
            
            self.init_robot_pose[env_idx] = self.agent.tcp.pose.p[env_idx]

            self.current_subgoal_idx[env_idx] = 0
    
    def _get_obs_agent(self):
        return dict()
        # return dict(qpos=self.agent.robot.get_qpos())

    def _get_obs_extra(self, info: Dict):
        # obs = dict(
        #     ee_pose=self.agent.tcp.pose.raw_pose,
        #     gripper_width=self.agent.robot.get_qpos()[:, -1:],
        #     goal_pos=self.goal_xyz,
        #     goal_rot=self.goal_quat,
        # )
        
        # if "state" in self.obs_mode:
        #     obj_obs = self.obj_mesh.pose.raw_pose
        #     obj_noise = torch.rand_like(obj_obs) * (self.obj_noise * 2) - self.obj_noise
        #     obs.update(
        #         obj_pose=obj_obs+obj_noise,
        #     )
        # return obs
        # Standard observations (non-goal related)
        # Initialize the observation dictionary
        obs = {
            "ee_pose": self.agent.tcp.pose.raw_pose,
            "gripper_width": self.agent.robot.get_qpos()[:, -1:],
        }
        
        # Add object state with noise if in state observation mode
        if "state" in self.obs_mode:
            obj_obs = self.obj_mesh.pose.raw_pose
            obj_noise = torch.rand_like(obj_obs) * (self.obj_noise * 2) - self.obj_noise
            obs["obj_pose"] = obj_obs + obj_noise
        
        # Achieved goal (current object position and rotation combined)
        if self.rotation_reward:
            obs["achieved_goal"] = self.obj_mesh.pose.raw_pose  # Contains both pos and rot
        else:
            obs["achieved_goal"] = self.obj_mesh.pose.p  # Position only
        
        # Desired goal (current waypoint - position and rotation combined)
        current_idx = self.current_subgoal_idx
        if self.rotation_reward:
            obs["desired_goal"] = self.waypoints[current_idx]  # Contains both pos and rot
        else:
            obs["desired_goal"] = self.waypoints[current_idx][:, :3]  # Position only

        obs["is_grasped"] = info['is_grasped']
        obs["should_be_grasped"] = info["should_be_grasped"]
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        ungrasp_amt = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        obs["ungrasp_amt"] = ungrasp_amt


        # obs["is_obj_placed"] = info['is_obj_placed']
        # obs["is_obj_rotated"] = info['is_obj_rotated']
        # obs["init_tcp_pos"] = self.init_robot_pose
        
        # Final goal (position and rotation combined)
        # if self.rotation_reward:
        #     obs["final_goal"] = torch.cat([self.goal_xyz, self.goal_quat], dim=1)  # Combine pos and rot
        # else:
        #     obs["final_goal"] = self.goal_xyz  # Position only

        # Rewrite my obs like this
        # robot state
        # pose of all objects
        # pose of object currently need to be approached/grasped
        # is_grasped for that object
        # tcp dist to that object
        # pose of all objects needed at next waypoint
        
        return obs

    def evaluate(self):
        """Evaluates the current state of the environment to determine task completion.
        
        This method checks three conditions:
        1. If the object is placed at the goal position within the threshold
        2. If the robot is currently grasping the target object
        3. If the robot is static (not moving significantly)
        
        Returns:
            dict: A dictionary containing boolean values for:
                - success: True if all conditions are met
                - is_obj_placed: True if object is at goal position
                - is_robot_static: True if robot is not moving
                - is_grasped: True if object is grasped
        """
        is_obj_placed = (
            torch.linalg.norm(self.goal_pose[:3] - self.obj_mesh.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_obj_rotated = (
           torch.abs(quaternion_angle(self.goal_pose[3:], self.obj_mesh.pose.q))
            <= self.angle_goal_thresh
        ) 
        is_grasped = self.agent.is_grasping(self.obj_mesh)
        should_be_grasped = self.current_subgoal_idx != len(self.waypoints)-1
        
        if self.visualize:
            self.tcp_site.set_pose(self.agent.tcp.pose)
        
        is_robot_static = self.agent.is_static(0.1)
        is_obj_static = self.obj_mesh.is_static(0.01, 0.1)
        # is_obj_static = self.obj_mesh.is_static(0.01*1.5, 0.1*1.5)
        return {
            # "success": is_grasped & is_obj_placed & is_robot_static,
            "success": is_grasped & is_obj_placed & is_obj_rotated & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "is_obj_static": is_obj_static,
            "is_obj_rotated": is_obj_rotated,
            "should_be_grasped": should_be_grasped,
        }

    # def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Write documentation for the reward function   
        """
        This reward function is designed to encourage the robot to move the object towards the goal position while 
        also tracking the waypoints along the way.

        The reward function is composed of three main components:
        1. Reaching reward - encourage moving TCP towards object
        2. Waypoint tracking rewards - encourage moving towards waypoints
        3. Static reward - encourage being still when at goal
        """ 
        # info['extra_data'] = {}
        # reward = 0

        # # 1. Reaching reward - encourage moving TCP towards object
        # tcp_to_obj_dist = torch.linalg.norm(
        #     self.obj_mesh.pose.p - self.agent.tcp.pose.p, axis=1
        # )
        # reaching_reward = 1 - torch.tanh(3 * tcp_to_obj_dist)
        # reaching_reward += 1 - torch.tanh(30 * tcp_to_obj_dist) # fine grained amount that only really applies when super close to object
        # reaching_reward /= 2
        # info['extra_data']['reaching_reward'] = reaching_reward
        # reward += reaching_reward
        
        # is_grasped = info["is_grasped"]
        # info['extra_data']['is_grasped_reward'] = is_grasped * reaching_reward
        # reward += is_grasped * reaching_reward

        # # 2. Waypoint tracking rewards
        # # Calculate distances and angles to all waypoints
        # waypoint_distances = torch.stack([
        #     torch.linalg.norm(waypoint[:3] - self.obj_mesh.pose.p, axis=1)
        #     for waypoint in self.waypoints
        # ], dim=1)
        
        # waypoint_angles = torch.stack([
        #     torch.abs(quaternion_angle(waypoint[3:], self.obj_mesh.pose.q))
        #     for waypoint in self.waypoints
        # ], dim=1)

        # # Update progress through waypoints
        # curr_idx_range = torch.arange(self.num_envs)
        # reached_waypoint = waypoint_distances[curr_idx_range, self.current_subgoal_idx] < self.goal_thresh
        # if self.rotation_reward:
        #     reached_waypoint &= (waypoint_angles[curr_idx_range, self.current_subgoal_idx] < self.angle_goal_thresh)
        

        # # Compute waypoint reward based on current subgoal
        # dist_to_current = waypoint_distances[curr_idx_range, self.current_subgoal_idx]
        # waypoint_reward = 1 - torch.tanh(
        #     self.waypoint_dist_reward_scaling[self.current_subgoal_idx] * dist_to_current
        # )
        # if self.rotation_reward:
        #     angle_to_current = waypoint_angles[curr_idx_range, self.current_subgoal_idx]
        #     waypoint_reward += (1 - torch.tanh(angle_to_current))
        #     waypoint_reward /= 2

        # info['extra_data']['waypoint_reward'] = waypoint_reward
        # # reward += waypoint_reward * info["is_grasped"] + self.current_subgoal_idx.float()
        # reward += waypoint_reward * (info["is_grasped"] == info["should_be_grasped"]) + self.current_subgoal_idx.float()


        # obj_in_place = info["is_obj_placed"] * info["is_obj_rotated"]
        # # obj_in_place = info["is_obj_placed"]
        # info['extra_data']['obj_in_place'] = obj_in_place


        # # obj_placed_reward = info["is_obj_placed"] * info["is_obj_rotated"]
        # # obj_placed_reward = info["is_obj_placed"] * info["is_obj_rotated"] * ~info['is_grasped']
        # # info['extra_data']['obj_placed_reward'] = obj_placed_reward
        # # reward[obj_placed_reward] = (len(self.waypoints)-1) * 2
        
        # # ungrasp and static reward
        # # if torch.any(self.current_subgoal_idx == len(self.waypoints)-1):
        # #     breakpoint()
        
        # # v = torch.linalg.norm(self.obj_mesh.linear_velocity, axis=1)
        # # av = torch.linalg.norm(self.obj_mesh.angular_velocity, axis=1)
        # # obj_static_reward = 1 - torch.tanh(v * 10 + av)
        # robot_static_reward = 1 - torch.tanh(
        #     5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        # )
        # reward[obj_in_place] += (
        #     robot_static_reward
        # )[obj_in_place]


        # #HERE
        # gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        # is_obj_grasped = info["is_grasped"]
        # ungrasp_reward = (
        #     torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        # )
        # ungrasp_reward[
        #     ~is_obj_grasped
        # ] = 20  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can open
        # reward[obj_in_place] += ungrasp_reward[obj_in_place] * self.agent.is_static(0.08)[obj_in_place]

        # # keep the robot static at the end state, since the sphere may spin when being placed on top
        # # reward[info["is_obj_on_bin"]] = (
        # #     6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        # # )[info["is_obj_on_bin"]]

        # # gripper_opened = self.agent.robot.get_qpos()[:, -1]  # Gripper status
        # # gripper_open_reward = 1 - torch.tanh(100 * (0.04 - gripper_opened))  # Scale between 0 and 1
        # # reward += gripper_open_reward * obj_in_place 
        # # reward += ~info['is_grasped'] * obj_in_place

        # self.current_subgoal_idx = torch.clamp(
        #     self.current_subgoal_idx + reached_waypoint, 
        #     0, 
        #     len(self.waypoints) - 1
        # )

        # # tcp_to_orig_dist = torch.linalg.norm(
        # #     self.agent.tcp.pose.p - self.init_robot_pose, axis=1
        # # )
        # # tcp_return_reward = 1 - torch.tanh(3 * tcp_to_orig_dist)

        # # reward += tcp_return_reward * obj_placed_reward * ~info['is_grasped']

        # # 3. Encourage gripper to open when object is placed and rotated
        # # gripper_opened = self.agent.robot.get_qpos()[:, -1]  # Gripper status
        # # gripper_open_reward = 1 - torch.tanh(30 * (0.04 - gripper_opened))  # Scale between 0 and 1
        # # reward += gripper_open_reward * info["is_obj_placed"] * info['is_obj_rotated']  # No multiplier needed, as it's already scaled

        # # # 4. Static reward - encourage being still when at goal
        # # static_reward = 1 - torch.tanh(
        # #     5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        # # )
        # # info['extra_data']['static_reward'] = static_reward
        # # reward += static_reward * info["is_obj_placed"] * info['is_obj_rotated'] * ~info['is_grasped']
    
        # # obj_static_reward = 1 - 0.5 * torch.tanh(
        # #     10 * torch.linalg.norm(self.obj_mesh.linear_velocity, axis=1)
        # # ) - 0.5 * torch.tanh(
        # #     5 * torch.linalg.norm(self.obj_mesh.angular_velocity, axis=1)
        # # )
        # # info['extra_data']['obj_static_reward'] = obj_static_reward
        # # reward += obj_static_reward * info["is_obj_placed"] * info['is_obj_rotated'] * ~info['is_grasped']
        
        # # reward[info["success"]] = 20

        # return reward
        
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Write documentation for the reward function   
        """
        This reward function is designed to encourage the robot to move the object towards the goal position while 
        also tracking the waypoints along the way.

        The reward function is composed of three main components:
        1. Reaching reward - encourage moving TCP towards object
        2. Waypoint tracking rewards - encourage moving towards waypoints
        3. Static reward - encourage being still when at goal
        """ 
        info['extra_data'] = {}
        reward = 0

        # 1. Reaching reward - encourage moving TCP towards object
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj_mesh.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(3 * tcp_to_obj_dist)
        reaching_reward += 1 - torch.tanh(30 * tcp_to_obj_dist) # fine grained amount that only really applies when super close to object
        reaching_reward /= 2
        info['extra_data']['reaching_reward'] = reaching_reward
        reward += reaching_reward
        
        is_grasped = info["is_grasped"]
        info['extra_data']['is_grasped_reward'] = is_grasped * reaching_reward
        reward += is_grasped * reaching_reward

        # 2. Waypoint tracking rewards
        # Calculate distances and angles to all waypoints
        waypoint_distances = torch.stack([
            torch.linalg.norm(waypoint[:3] - self.obj_mesh.pose.p, axis=1)
            for waypoint in self.waypoints
        ], dim=1)
        
        waypoint_angles = torch.stack([
            torch.abs(quaternion_angle(waypoint[3:], self.obj_mesh.pose.q))
            for waypoint in self.waypoints
        ], dim=1)

        # Update progress through waypoints
        curr_idx_range = torch.arange(self.num_envs)
        reached_waypoint = waypoint_distances[curr_idx_range, self.current_subgoal_idx] < self.goal_thresh
        if self.rotation_reward:
            reached_waypoint &= (waypoint_angles[curr_idx_range, self.current_subgoal_idx] < self.angle_goal_thresh)
        
        # Compute waypoint reward based on current subgoal
        dist_to_current = waypoint_distances[curr_idx_range, self.current_subgoal_idx]
        waypoint_reward = 1 - torch.tanh(
            self.waypoint_dist_reward_scaling[self.current_subgoal_idx] * dist_to_current
        )
        if self.rotation_reward:
            angle_to_current = waypoint_angles[curr_idx_range, self.current_subgoal_idx]
            waypoint_reward += (1 - torch.tanh(angle_to_current))
            waypoint_reward /= 2

        info['extra_data']['waypoint_reward'] = waypoint_reward
        reward += waypoint_reward * info["is_grasped"] + self.current_subgoal_idx.float()
        info['extra_data']['waypoint+subgoal_idx'] = waypoint_reward * info["is_grasped"] + self.current_subgoal_idx.float()

        # 3. Static reward - encourage being still when at goal
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        info['extra_data']['static_reward'] = static_reward
        # reward += static_reward * info["is_obj_placed"]
        reward += static_reward * info["is_obj_placed"] * info["is_obj_rotated"]

        self.current_subgoal_idx = torch.clamp(
            self.current_subgoal_idx + reached_waypoint, 
            0, 
            len(self.waypoints) - 1
        )

        reward[info["success"]] += 1

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / (5+len(self.waypoints))



# def evaluate(self):
#         is_red_placed = (
#             torch.linalg.norm(self.red_goal_pose - self.red_cube.pose.p, axis=1)
#             <= self.goal_thresh
#         )
#         is_blue_placed = (
#             torch.linalg.norm(self.blue_goal_pose - self.blue_cube.pose.p, axis=1)
#             <= self.goal_thresh
#         )
#         is_red_grasped = self.agent.is_grasping(self.red_cube)
#         is_blue_grasped = self.agent.is_grasping(self.blue_cube)
#         is_robot_static = self.agent.is_static(0.2)
#         return {
#             "success": is_red_placed & is_blue_placed & is_robot_static,
#             "is_red_placed": is_red_placed,
#             "is_blue_placed": is_blue_placed,
#             "is_robot_static": is_robot_static,
#             "is_red_grasped": is_red_grasped,
#             "is_blue_grasped": is_blue_grasped,
#         }

#     def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
#         info['extra_data'] = {}

#         reward = 0
#         tcp_to_red_dist = torch.linalg.norm(
#             self.red_cube.pose.p - self.agent.tcp.pose.p, axis=1
#         )
#         red_reaching_reward = 1 - torch.tanh(5 * tcp_to_red_dist)
#         info['extra_data']['red_reaching_reward'] = red_reaching_reward
#         reward += red_reaching_reward

#         is_red_grasped = info["is_red_grasped"]
#         reward += is_red_grasped

#         red_to_goal_dist = torch.linalg.norm(
#             self.red_goal_pose - self.red_cube.pose.p, axis=1
#         )
#         place_reward = (1 - torch.tanh(5 * red_to_goal_dist))
#         info['extra_data']['red_place_reward'] = place_reward
#         reward += place_reward * is_red_grasped

#         red_static_reward = info["is_red_placed"]*(1-info["is_red_grasped"])
#         info['extra_data']['red_static_reward'] = red_static_reward
#         reward[red_static_reward] = 4

#         tcp_to_blue_dist = torch.linalg.norm(
#             self.blue_cube.pose.p - self.agent.tcp.pose.p, axis=1
#         )
#         blue_reaching_reward = 1 - torch.tanh(5 * tcp_to_blue_dist)
#         info['extra_data']['blue_reaching_reward'] = blue_reaching_reward
#         reward += blue_reaching_reward*red_static_reward

#         is_blue_grasped = info["is_blue_grasped"]
#         reward += is_blue_grasped*red_static_reward

#         blue_to_goal_dist = torch.linalg.norm(
#             self.blue_goal_pose - self.blue_cube.pose.p, axis=1
#         )
#         place_reward = (1 - torch.tanh(5 * blue_to_goal_dist))
#         info['extra_data']['blue_place_reward'] = place_reward
#         reward += place_reward * is_blue_grasped*red_static_reward

#         blue_static_reward = info["is_blue_placed"]*(1-info["is_blue_grasped"])
#         info['extra_data']['blue_static_reward'] = blue_static_reward
#         reward[blue_static_reward] = 8

#         return reward