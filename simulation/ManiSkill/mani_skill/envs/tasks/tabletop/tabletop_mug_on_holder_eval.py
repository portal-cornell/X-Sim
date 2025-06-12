from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.tabletop_mug_on_holder import TabletopMugOnHolderEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import torch
import random
import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R


@register_env("Tabletop-Mug-On-Holder-Eval", max_episode_steps=100)
class TabletopMugOnHolderEvalEnv(TabletopMugOnHolderEnv):
    """
    Evaluation version of KitchenSubgoalsEnv that terminates episodes when success is reached.
    
    This environment is identical to KitchenSubgoalsEnv but adds early termination when
    the success condition is met, making it more suitable for evaluation purposes.
    """

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            uid="render_camera",  
            pose=sapien.Pose(), 
            width=1920, 
            height=1080, 
            near=0.01, 
            far=100, 
            intrinsic=[
                [1.0663400e+03, 0.0000000e+00, 9.724600e+02],
                [0.0000000e+00, 1.0662000e+03, 5.229580e+02],
                [0.0000000e+00, 0.0000000e+00, 1.000000e+00],
            ],
            # shader_pack='rt',
            mount=self.cam_mount
        )
    
    
    def __init__(self, *args, randomize_lighting=False, randomize_camera=False, **kwargs):        
        # Store randomization flags
        self.randomize_lighting = randomize_lighting
        self.randomize_camera = randomize_camera
        
        # Define a list of lighting configurations
        self.lighting_configs = [
            # Default lighting setup
            {
                "ambient": [0.3, 0.3, 0.3],
                "main_light": {
                    "direction": [1, 0.5, -1], 
                    "color": [1, 1, 1],
                    "shadow": False,
                    "shadow_scale": 5,
                    "shadow_map_size": 2048
                },
                "fill_light": {"direction": [0, 0, -1], "color": [1, 1, 1]}
            },
            # Small variation 1 - slightly warmer light
            {
                "ambient": [0.5, 0.3, 0.2],
                "main_light": {
                    "direction": [1.5, 0.45, -0.8],
                    "color": [1.05, 0.98, 0.95],
                    "shadow": False,
                    "shadow_scale": 5,
                    "shadow_map_size": 2048
                },
                "fill_light": {"direction": [0.05, -0.05, -1], "color": [1.02, 1, 0.98]}
            },
            # Small variation 2 - slightly cooler light
            {
                "ambient": [0.1, 0.3, 0.5],
                "main_light": {
                    "direction": [0.95, 0.55, -1.05],
                    "color": [0.95, 0.98, 1.05],
                    "shadow": True,
                    "shadow_scale": 5,
                    "shadow_map_size": 2048
                },
                "fill_light": {"direction": [-0.05, 0.05, -0.98], "color": [0.98, 1, 1.02]}
            },
            # Small variation 3 - slightly dimmer light
            {
                "ambient": [0.15, 0.15, 0.15],
                "main_light": {
                    "direction": [1.02, 0.48, -1.02],
                    "color": [0.95, 0.95, 0.95],
                    "shadow": True,
                    "shadow_scale": 5,
                    "shadow_map_size": 2048
                },
                "fill_light": {"direction": [0.02, -0.02, -1.02], "color": [0.95, 0.95, 0.95]}
            },
            # Small variation 3 - slightly dimmer light
            {
                "ambient": [0.7, 0.7, 0.7],
                "main_light": {
                    "direction": [1.02, 0.48, -1.02],
                    "color": [0.95, 0.95, 0.95],
                    "shadow": True,
                    "shadow_scale": 5,
                    "shadow_map_size": 2048
                },
                "fill_light": {"direction": [0.02, -0.02, -1.02], "color": [0.95, 0.95, 0.95]}
            }
        ]
        # Get the default camera pose
        self.default_eye = np.array([0.11364682, 0.61086124, 1.10074165])
        self.default_target = np.array([0.69425251, 0.12618053, 0.4465386])
        
        # Define random variation ranges (in meters for position, radians for angle)
        self.pose_variation = {
            "eye_xy": 0.03,      # Max horizontal displacement
            "eye_z": 0.03,      # Max vertical displacement
            "target_xy": 0.03,  # Max target point horizontal shift
            "target_z": 0.03   # Max target point vertical shift
        }
        
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Call parent class initialization
        super()._initialize_episode(env_idx, options)
        
        b = len(env_idx)
        
        # Apply lighting configuration
        for i in range(b):
            scene = self.scene.sub_scenes[i]
            
            # Remove existing lights by removing their entities
            for entity in scene.entities:
                if entity.name == "directional_light":
                    scene.remove_entity(entity)
            
            if self.randomize_lighting:
                # Apply random lighting config
                light_idx = random.randint(0, len(self.lighting_configs) - 1)
                config = self.lighting_configs[light_idx]
            else:
                # Apply default lighting config (first config in the list)
                config = self.lighting_configs[0]
            
            scene.ambient_light = config["ambient"]
            
            if config["main_light"]:
                self.scene.add_directional_light(
                    direction=config["main_light"]["direction"],
                    color=config["main_light"]["color"],
                    shadow=config["main_light"].get("shadow", False),
                    shadow_scale=config["main_light"].get("shadow_scale", 1),
                    shadow_map_size=config["main_light"].get("shadow_map_size", 2048),
                    scene_idxs=[i]
                )
            
            if config["fill_light"]:
                self.scene.add_directional_light(
                    direction=config["fill_light"]["direction"],
                    color=config["fill_light"]["color"],
                    scene_idxs=[i]
                )
        
        # Handle camera pose
        for i in range(b):
            pose = sapien_utils.look_at(self.default_eye, self.default_target)
            pose = Pose.create(pose)
            if self.randomize_camera:
                # Generate random camera pose
                pose = pose * Pose.create_from_pq(
                    p=torch.rand((1, 3), device=self.device) * torch.tensor(
                        [self.pose_variation["eye_xy"], 
                         self.pose_variation["eye_xy"], 
                         self.pose_variation["eye_z"]], 
                        device=self.device
                    ) * 2 - torch.tensor(
                        [self.pose_variation["eye_xy"], 
                         self.pose_variation["eye_xy"], 
                         self.pose_variation["eye_z"]], 
                        device=self.device
                    ),
                )
            
            # Update the camera mount pose for this scene
            self.cam_mount.set_pose(pose)

    def _load_lighting(self, options: dict):
        # We'll handle lighting in _initialize_episode instead
        pass

    def _load_scene(self, options: dict):
        # Call parent class scene loading
        super()._load_scene(options)
        
        # Create a camera mount actor
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")

    
    def _get_obs_extra(self, info):
        # Call parent class method
        obs = super()._get_obs_extra(info)  # Call the superclass method
        
        ee_pos = self.agent.ee_link.pose.p
        ee_euler = torch.tensor(R.from_quat(torch.roll(self.agent.ee_link.pose.q, shifts=-1, dims=-1).cpu().numpy()).as_euler('xyz')).to(self.device)
        gripper_open = self.agent.robot.get_qpos()[:, -1:]/0.04
    
        proprio = torch.cat([
            ee_pos, ee_euler, gripper_open
        ], dim=-1)

        info['proprio'] = proprio
        info['real_joint_angles'] = self.agent.robot.get_qpos()
        return obs

    def evaluate(self):
        # call super eval
        info = super().evaluate()
        tcp_obj_distance = torch.zeros((self.num_envs), device=self.device)
        is_tcp_near_obj = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)
        
        for i, obj_mesh in enumerate(self.obj_meshes):
            if i == self.single_obj_pick_place_idx:
                tcp_obj_distance = torch.linalg.norm(
                    obj_mesh.pose.p - self.agent.tcp.pose.p, axis=1
                )
                is_tcp_near_obj = tcp_obj_distance <= 0.06
        info.update({
            "tcp_obj_distance": tcp_obj_distance,
            "is_tcp_near_obj": is_tcp_near_obj,
        })
        return info