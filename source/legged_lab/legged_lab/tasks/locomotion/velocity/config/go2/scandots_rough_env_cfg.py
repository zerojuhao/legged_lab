from __future__ import annotations

import torch

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import ObservationsCfg, MySceneCfg, LocomotionVelocityRoughEnvCfg
import legged_lab.tasks.locomotion.velocity.mdp as mdp


@configclass
class RayCasterArrayCfg(RayCasterCfg):
    
    shape : tuple[int, int] = (-1, -1)
    
    def __post_init__(self):
        resolution = self.pattern_cfg.resolution
        size = self.pattern_cfg.size
        
        x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution)
        y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution)

        x_len = x.numel()
        y_len = y.numel()
        
        self.shape = (x_len, y_len)


@configclass
class ScandotsSceneCfg(MySceneCfg):
    """Configuration for the terrain scene with a legged robot.
    Change height_scanner to self-defined RayCasterArrayCfg
    """
    
    height_scanner = RayCasterArrayCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),    # offset, TODO: check it in viewer
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ScandotsObservationsCfg(ObservationsCfg):
    @configclass
    class SensorCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan_ch,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        
        def __post_init__(self):
            self.enable_corruption = True
        
    sensor: SensorCfg = SensorCfg()
    """configuration of scandots sensor.
    it would be stored in extras["observations"]["sensor"] and further used in rsl_rl (modified)
    refer to Isaac Lab's source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py
    """
    
    def __post_init__(self):
        # height scan in SensorCfg, not in PolicyCfg
        self.policy.height_scan = None

@configclass
class UnitreeGo2ScandotsRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    scene: ScandotsSceneCfg = ScandotsSceneCfg(num_envs=4096, env_spacing=2.5)
    observations : ScandotsObservationsCfg = ScandotsObservationsCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        

        
        