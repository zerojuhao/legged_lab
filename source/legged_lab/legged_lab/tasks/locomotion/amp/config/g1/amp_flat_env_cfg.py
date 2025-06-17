# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg

import legged_lab.tasks.locomotion.velocity.mdp as mdp
from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import RewardsCfg, MySceneCfg
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

##
# Pre-defined configs
##
from legged_lab.assets.unitree import G1_29DOF_LOCK_WAIST_MINIMAL_CFG

from legged_lab import LEGGED_LAB_ROOT_DIR


@configclass
class G1AmpSceneCfg(MySceneCfg):
    pass

@configclass
class G1AmpRewards():
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.1,
    )
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_wrists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )


@configclass
class G1AmpFlatEnvCfg(LocomotionAmpEnvCfg):
    rewards: G1AmpRewards = G1AmpRewards()
    scene: G1AmpSceneCfg = G1AmpSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Scene
        self.scene.robot = G1_29DOF_LOCK_WAIST_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        
        # Observations
        self.observations.policy.height_scan = None
        self.observations.amp.key_links_pos_b.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=[
                "left_ankle_roll_link", 
                "right_ankle_roll_link", 
                "left_wrist_yaw_link", 
                "right_wrist_yaw_link",
                "waist_yaw_link",
                # "left_shoulder_roll_link",
                # "right_shoulder_roll_link",
                # "left_hip_pitch_link",
                # "right_hip_pitch_link",
                "left_elbow_link",
                "right_elbow_link",
                "left_knee_link", 
                "right_knee_link",
            ]
        )
        self.observations.amp.key_links_pos_b.params["local_pos_dict"] = {
            "left_ankle_roll_link": (0.0, 0.0, 0.0),
            "right_ankle_roll_link": (0.0, 0.0, 0.0),
            "left_wrist_yaw_link": (0.0415, 0.003, 0.0),
            "right_wrist_yaw_link": (0.0415, -0.003, 0.0),
            "waist_yaw_link": (0.0, 0.0, 0.4),
            # "left_shoulder_roll_link": (0.0, 0.0, 0.0),
            # "right_shoulder_roll_link": (0.0, 0.0, 0.0),
            # "left_hip_pitch_link": (0.0, 0.0, 0.0),
            # "right_hip_pitch_link": (0.0, 0.0, 0.0),
            "left_elbow_link": (0.0, 0.0, 0.0),
            "right_elbow_link": (0.0, 0.0, 0.0),
            "left_knee_link": (0.0, 0.0, 0.0),
            "right_knee_link": (0.0, 0.0, 0.0),
        }
        
        # Curriculum
        self.curriculum.terrain_levels = None
        
        # Events
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]

        # Rewards
        # For AMP, we only needs a few rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.alive.weight = 0.1
        
        self.rewards.dof_pos_limits.weight = -0.0
        self.rewards.joint_deviation_hip.weight = -0.0
        self.rewards.joint_deviation_arms.weight = -0.0
        self.rewards.joint_deviation_wrists.weight = -0.0
        self.rewards.joint_deviation_waist.weight = -0.0
        
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        
        # motion loader
        self.motion_file_path = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof_lock_waist", "retargeted_motion.pkl"
        )
        self.motion_cfg_path = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof_lock_waist", "retargeted.yaml"
        )


@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
