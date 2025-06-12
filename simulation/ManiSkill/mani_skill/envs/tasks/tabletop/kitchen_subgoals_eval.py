from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.kitchen_subgoals import KitchenSubgoalsEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import torch
import random
import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R


@register_env("Kitchen-Subgoals-Eval", max_episode_steps=60)
class KitchenSubgoalsEvalEnv(KitchenSubgoalsEnv):
    """
    Evaluation version of KitchenSubgoalsEnv that terminates episodes when success is reached.
    
    This environment is identical to KitchenSubgoalsEnv but adds early termination when
    the success condition is met, making it more suitable for evaluation purposes.
    """

    goal_thresh = 0.04
    # z_rand_amt = 0.0

    @property
    def _default_human_render_camera_configs(self):
        # Use identity pose since we'll control via mount
        # return CameraConfig(
        #     "render_camera", 
        #     pose=sapien.Pose(), 
        #     width=960, 
        #     height=540, 
        #     fov=1, 
        #     near=0.01, 
        #     far=100, 
        #     shader_pack='rt',
        #     mount=self.cam_mount
        # )
    
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
            # Default lighting setup (matching sapien_env.py)
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
            # Warm evening
            {
                "ambient": [0.3, 0.25, 0.2],
                "main_light": {"direction": [1, 1, -1], "color": [0.8, 0.7, 0.6]},
                "fill_light": {"direction": [-1, -0.5, -1], "color": [0.2, 0.2, 0.3]}
            },
            # Cool morning
            {
                "ambient": [0.4, 0.4, 0.5],
                "main_light": {"direction": [-1, 1, -1], "color": [0.6, 0.6, 0.7]},
                "fill_light": {"direction": [1, -0.5, -1], "color": [0.2, 0.2, 0.2]}
            },
            # Dim indoor
            {
                "ambient": [0.2, 0.2, 0.2],
                "main_light": {"direction": [0, 0, -1], "color": [0.3, 0.3, 0.3]},
                "fill_light": {"direction": [1, 0, -0.5], "color": [0.1, 0.1, 0.1]}
            }
        ]
        
        # Get the default camera pose
        self.default_eye = np.array([-0.01686193, -0.39387421, 0.6638793])
        self.default_target = np.array([0.81217533, 0.01006028, 0.27718283])
        
        # Define random variation ranges (in meters for position, radians for angle)
        self.pose_variation = {
            "eye_xy": 0.1,      # Max horizontal displacement
            "eye_z": 0.1,      # Max vertical displacement
            "target_xy": 0.05,  # Max target point horizontal shift
            "target_z": 0.05    # Max target point vertical shift
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
                    q=randomization.random_quaternions(
                        n=1, 
                        device=self.device, 
                        bounds=(-np.pi/24, np.pi/24)
                    )
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

    def step(self, action):
        # Call parent class step method
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Terminate episode if success is achieved (using element-wise OR)
        terminated = terminated | info["success"]
            
        return obs, reward, terminated, truncated, info
    
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
        return obs
