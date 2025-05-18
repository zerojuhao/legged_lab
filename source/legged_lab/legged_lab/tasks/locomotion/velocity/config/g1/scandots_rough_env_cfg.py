from __future__ import annotations
import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    ScandotsObservationsCfg, 
    ScandotsSceneCfg, 
    LocomotionVelocityRoughEnvCfg
)
import legged_lab.tasks.locomotion.velocity.mdp as mdp

from .rough_env_cfg import G1Rewards, G1RoughEnvCfg, G1RoughEnvCfg_PLAY

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class G1ScandotsRoughEnvCfg(G1RoughEnvCfg):
    
    scene: ScandotsSceneCfg = ScandotsSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ScandotsObservationsCfg = ScandotsObservationsCfg()
    rewards: G1Rewards = G1Rewards()
    
    def __post_init__(self):
        super().__post_init__()
    


@configclass
class G1ScandotsRoughEnvCfg_PLAY(G1ScandotsRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
