import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp

from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
)


@configclass
class AmpObservationsCfg():
    
    @configclass
    class AugPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        key_links_pos_b: ObsTerm = ObsTerm(
            func=mdp.key_links_pos, 
            params={
                "asset_cfg": SceneEntityCfg("frame_transformer")
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: AugPolicyCfg = AugPolicyCfg()
    
    @configclass
    class AmpCfg(ObsGroup):        
        dof_pos: ObsTerm = ObsTerm(func=mdp.joint_pos)
        dof_vel: ObsTerm = ObsTerm(func=mdp.joint_vel)
        # base_lin_vel_b: ObsTerm = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel_b: ObsTerm = ObsTerm(func=mdp.base_ang_vel)
        # base_pos_z: ObsTerm = ObsTerm(func=mdp.base_pos_z)  # TODO: consider terrain height
        key_links_pos_b: ObsTerm = ObsTerm(
            func=mdp.key_links_pos, 
            params={
                "asset_cfg": SceneEntityCfg("frame_transformer")
            }
        )
    
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 2
            self.flatten_history_dim = False    # if True, it will flatten each term history first and then concatenate them, 
                                                # which is not we want for AMP observations
                                                # Thus, we set it to False, and address it manually
    
    amp: AmpCfg = AmpCfg()
    

@configclass
class AmpEventCfg(EventCfg):
    """Configuration for amp events."""
    
    reset_base_rsi = EventTerm(
        func=mdp.ref_state_init_root, 
        mode="reset",
    )

    reset_robot_joints_rsi = EventTerm(
        func=mdp.ref_state_init_dof,
        mode="reset",
    )

    def __post_init__(self):
        
        self.reset_base = None
        self.reset_robot_joints = None


@configclass
class LocomotionAmpEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Environment configuration for the AMP locomotion task.
    """
    observations: AmpObservationsCfg = AmpObservationsCfg()
    events: AmpEventCfg = AmpEventCfg()
    
    motion_file_path: str = MISSING
    """Path to the motion file for AMP."""
    
    motion_cfg_path: str = MISSING
    """Path to the motion configuration file for AMP."""
    
    def __post_init__(self):
        
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        super().__post_init__()
        
        