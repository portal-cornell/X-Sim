from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, PandaNinja
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.custom_table import CustomTableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_angle, axis_angle_to_quaternion
from mani_skill import PACKAGE_ASSET_DIR
from sapien.physx import PhysxMaterial


@register_env("ApproachCube-v1-custom", max_episode_steps=50)
class ApproachCubeCustomEnv(BaseEnv):
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
    agent: Union[Panda, Fetch, PandaNinja]
    cube_half_size = 0.025
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda_ninja", robot_init_qpos_noise=0.02, randomize_init_config=False, obj_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize_init_config = randomize_init_config
        self.real_cube = True
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3+0.615, 0, 0.6], target=[-0.1+0.615, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.6+0.615, 0.7, 0.6], [0.615, 0.0, 0.35])
        # pose = sapien_utils.look_at([0.05255695, -0.57352776, 0.89239827], [0.75543028, -0.1582579, 0.31488633])
        pose = sapien_utils.look_at([-0.03740435, -0.45771217,  0.61309804], 
                                    [0.77913187, -0.02136986,  0.23511127])
        return CameraConfig("render_camera", pose, 960, 540, 1, 0.01, 100)
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=10,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = CustomTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        if not self.real_cube:
            self.cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[1, 0, 0, 1],
                name="cube",
                initial_pose=sapien.Pose(p=[0.7446, -0.2887, 0.2473 + self.cube_half_size]),
            )
        else:
            builder = self.scene.create_actor_builder()
            density = 1000
            builder.add_multiple_convex_collisions_from_file(
                filename=f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/cube_centered/mesh/cube_centered.obj",
                # filename=f"{PACKAGE_ASSET_DIR}/portal_lab/move_cube/mesh/red_block.obj",
                material=None,
                density=density,
                decomposition="coacd",
            )
            builder.add_visual_from_file(filename=f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/cube_centered/mesh/cube_centered.obj")
            builder.initial_pose = sapien.Pose(p=[0.3648, 0.1595, 0.1851 + 0.07])
            self.cube = builder.build(name='cube')

        if not self.real_cube:
            self.goal_site = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
        else:
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(filename=f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/cube_centered/mesh/cube_centered.obj")
            builder.initial_pose = sapien.Pose(p=[0.3648, 0.1595, 0.1851 + 0.07])
            self.goal_site = builder.build_kinematic(name='goal_site')

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
        self.poses = torch.tensor(np.load(f"{PACKAGE_ASSET_DIR}/portal_lab/flat_kitchen/cube_centered/demos/cube_lr.npy")).to(self.device)[0]



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            # xyz[:, :3] += torch.tensor([0.7446, -0.2887, 0.2473 + self.cube_half_size])
            xyz[:, :3] = self.poses[0, :3]
            if self.randomize_init_config:
                xyz[:, :2] += torch.rand((b, 2)) * 0.05 - 0.025

            quat = torch.zeros((b, 4))
            if self.real_cube:
                quat[:, ] = self.poses[0, 3:7]
            else:
                quat[:, 0] = 1
            fixed_rot = Pose.create_from_pq(q=quat)
            self.cube.set_pose(Pose.create_from_pq(p=xyz, q=fixed_rot.q))

            goal_xyz = torch.zeros((b, 3))
            # goal_xyz += torch.tensor([0.6912, 0.2484, 0.2473 + self.cube_half_size])
            # goal_xyz[:, :3] += torch.tensor([0.7446 - 0.06, 0, 0.2473 + self.cube_half_size + 0.05])
            goal_xyz[:, :3] = self.poses[-1, :3]
            goal_quat = torch.zeros((b, 4))
            if self.real_cube:
                goal_quat[:, ] = self.poses[-1, 3:7]
            else:
                goal_quat[:, 0] = 1
            goal_fixed_rot = Pose.create_from_pq(q=goal_quat)
            # goal_xyz[:, :2] += torch.rand((b, 2)) * 0.2 - 0.1

            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz, q=goal_fixed_rot.q))
    
    def _get_obs_agent(self):
        return dict()

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            # is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            ee_pose=self.agent.ee_link.pose.raw_pose,
            gripper_width=self.agent.robot.get_qpos()[:, -1:],
            goal_pos=self.goal_site.pose.p,
            goal_rot=self.goal_site.pose.q,
        )
        self.tcp_site.set_pose(self.agent.tcp.pose)
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                # tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                # obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                # obj_to_goal_rot=quaternion_angle(self.goal_site.pose.q, self.cube.pose.q),
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_obj_rotated = (
            torch.abs(quaternion_angle(self.goal_site.pose.q, self.cube.pose.q))
            <= (self.goal_thresh * 2)
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            # "success": is_grasped & is_obj_placed & is_robot_static,
            "success": is_obj_placed & is_robot_static & is_obj_rotated,
            "is_obj_placed": is_obj_placed,
            "is_obj_rotated": is_obj_rotated,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        info['extra_data'] = {}
        reward = 0
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(2 * tcp_to_obj_dist)
        info['extra_data']['reaching_reward'] = reaching_reward
        reward += reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(obj_to_goal_dist)
        info['extra_data']['place_reward'] = place_reward
        reward += place_reward * is_grasped
        
        rot_reward = 1 - torch.tanh(torch.abs(quaternion_angle(self.goal_site.pose.q, self.cube.pose.q)))
        info['extra_data']['rot_reward'] = rot_reward
        reward += rot_reward * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        info['extra_data']['static_reward'] = static_reward
        # reward += static_reward * info["is_obj_placed"]
        reward += static_reward * info["is_obj_placed"] * info["is_obj_rotated"]

        # reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
