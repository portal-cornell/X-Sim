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


@register_env("Kitchen-LoadVid", max_episode_steps=50)
class KitchenLoadVideoEnv(BaseEnv):
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
    goal_thresh = 0.05
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def __init__(self, *args, robot_uids="panda_ninja", robot_init_qpos_noise=0.02, randomize_init_config=False, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.randomize_init_config = randomize_init_config
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.05255695, -0.57352776, 0.89239827], 
                                    [0.75543028, -0.1582579, 0.31488633])

        pose_new = sapien_utils.look_at([0.01346585, -0.27562662,  0.28802274 + 0.3], 
                                    [0.91798244, 0.15073409, 0.29616359 - 0.2])

        # Validation pose
        # pose = sapien_utils.look_at([0.3448, 0.4595, 0.1851], 
        #                             [0.3648, 0.1595, 0.1851])


        return CameraConfig("render_camera", pose_new, 960, 540, 1, 0.01, 100)
    
    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         sim_freq=100,
    #         control_freq=10,
    #     )
    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
            sim_freq=100,
            control_freq=10,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0]))

    def _load_object(self, scene):
        builder = scene.create_actor_builder()
        density = 1000
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )

        # builder.add_nonconvex_collision_from_file(
        #     filename="assets/portal_lab/kitchen_xl_corrected.obj",
        #     pose = sapien.Pose(p=[1.5730, 0.1690, 0.1888]),
        # )


        builder.add_multiple_convex_collisions_from_file(
            # filename=f"{PACKAGE_ASSET_DIR}/portal_lab/kitchen_xl_corrected.obj",
            filename=f"{PACKAGE_ASSET_DIR}/portal_lab/new_kitchen_pretty/UV_Kitchen_Final3.obj",
            material = physical_material,
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
            # filename=f"{PACKAGE_ASSET_DIR}/portal_lab/kitchen_corrected.glb",
            filename=f"{PACKAGE_ASSET_DIR}/portal_lab/new_kitchen_pretty/UV_Kitchen_Final3.obj",
        )
        builder.initial_pose = sapien.Pose(p=[0.3648, 0.1595, 0.1851])
        mesh = builder.build_kinematic(name="kitchen")
        # mesh.set_pose(sapien.Pose(p=[0.5730, 0.1690, 0.1888]))
        
        return mesh

    def load_cube(self, scene):
        builder = scene.create_actor_builder()
        density = 1000
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=f"{PACKAGE_ASSET_DIR}/portal_lab/move_mustard/mesh/mustard.obj",
            material=physical_material,
            density=density,
            decomposition="coacd",
        )
        builder.add_visual_from_file(filename=f"{PACKAGE_ASSET_DIR}/portal_lab/move_mustard/mesh/mustard.obj")
        builder.initial_pose = sapien.Pose(p=[0.3648, 0.1595, 0.1851 + 0.07])
        cube = builder.build(name="cube")
        return cube
    
    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.mesh = self._load_object(self.scene)
        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0.6577, 0.2101, 0.2588 + self.cube_half_size]),
        # )
        self.cube = self.load_cube(self.scene)
        # set waypoints between cube and goal
        self.waypoints = [
            [0.72, -0.275, 0.35 + self.cube_half_size],
            [0.69, 0, 0.50 + self.cube_half_size],
            [0.6577, 0.2093, 0.3 + self.cube_half_size],
        ]

        self.poses = torch.tensor(np.load(f"{PACKAGE_ASSET_DIR}/portal_lab/move_mustard/mustard_test.npy")).to(self.device)
        self.start_pose = self.poses[0]
        # self.start_pose[0] += 0.06
        # self.start_pose[1] -= 0.06
        self.goal_pose = self.poses[-1]
        self.current_idx = 0

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :3] = self.start_pose[:3]
            if self.randomize_init_config:
                xyz[:, :2] += torch.rand((b, 2)) * 0.2 - 0.1

            quat = torch.zeros((b, 4))
            quat[:, ] = self.start_pose[3:7]
            fixed_rot = Pose.create_from_pq(q=quat)

            kitchen_xyz = torch.zeros((b, 3))
            kitchen_xyz[:, :3] = torch.tensor([0.3648, 0.1595, 0.185])
            if self.randomize_init_config:
                kitchen_xyz[:, :2] += torch.rand((b, 2)) * 0.1 - 0.05

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :3] += self.goal_pose[:3]

            if self.randomize_init_config:
                z_random = torch.rand((b, 1)) * 0.1 - 0.05
                kitchen_xyz[:, 2:3] += z_random
                xyz[:, 2:3] += z_random 
                goal_xyz[:, 2:3] += z_random


            # self.cube.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            self.cube.set_pose(Pose.create_from_pq(p=xyz, q=fixed_rot.q))

            
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

            # self.mesh.set_pose(sapien.Pose(p=[0.5730, 0.1690, 0.1888]))
            self.mesh.set_pose(Pose.create_from_pq(p=kitchen_xyz))



            qpos = np.array(
                [0.0, -0.7853, -0.0, -2.3561, 0.0, 1.5707, 0.7853, 0.04, 0.04]

                # Validation pose
                # [ 0.0590, 0.0735, 0.3940, -2.8002, 0.1348, 2.8352, 1.1367, 0.04, 0.04]
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
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        
        obs = dict(
            # is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            ee_pose=self.agent.ee_link.pose.raw_pose,
            gripper_width=self.agent.robot.get_qpos()[:, -1:],
            goal_pos=self.goal_site.pose.p,
            # goal_rot=self.goal_site.pose.q,
        )

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
            torch.linalg.norm(self.goal_pose[:3] - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        # self.cube.set_pose(Pose.create_from_pq(p=self.poses[self.current_idx, :3],
        #                                         q=self.poses[self.current_idx, 3:]))
        # self.current_idx += 2
        # self.current_idx = min(self.current_idx, self.poses.shape[0]-1)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_grasped & is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        info['extra_data'] = {}
        reward = 0
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        info['extra_data']['reaching_reward'] = reaching_reward
        reward += reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_pose[:3] - self.cube.pose.p, axis=1
        )
        place_reward = (1 - torch.tanh(5 * obj_to_goal_dist))
        info['extra_data']['place_reward'] = place_reward
        reward += place_reward * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        info['extra_data']['static_reward'] = static_reward
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

