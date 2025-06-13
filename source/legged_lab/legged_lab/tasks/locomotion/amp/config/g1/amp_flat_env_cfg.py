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
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

from legged_lab import LEGGED_LAB_ROOT_DIR


@configclass
class G1AmpSceneCfg(MySceneCfg):
    frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        target_frames=[
            # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/pelvis"),
            # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_hip_pitch_link"),
            # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_knee_link"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link"),
            # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_shoulder_roll_link"),
            # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_elbow_pitch_link"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_zero_link"),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/pelvis", 
            #     name="head_link", 
            #     offset=OffsetCfg(
            #         pos=(0.0, 0.0, 0.4), 
            #         rot=(1.0, 0.0, 0.0, 0.0)
            #     )
            # ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            #     name="left_toe_link",
            #     offset=OffsetCfg(
            #         pos=(0.08, 0.0, 0.0), 
            #         rot=(1.0, 0.0, 0.0, 0.0)
            #     )
            # ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            #     name="right_toe_link",
            #     offset=OffsetCfg(
            #         pos=(0.08, 0.0, 0.0),
            #         rot=(1.0, 0.0, 0.0, 0.0)
            #     )
            # ),
        ], 
        debug_vis=False
    )

@configclass
class G1AmpRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
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
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


@configclass
class G1AmpFlatEnvCfg(LocomotionAmpEnvCfg):
    rewards: G1AmpRewards = G1AmpRewards()
    scene: G1AmpSceneCfg = G1AmpSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Events
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]

        # Rewards
        # For AMP, we only needs a few rewards
        self.rewards.track_lin_vel_xy_exp.weight = 24.0
        self.rewards.track_ang_vel_z_exp.weight = 8.0
        
        self.rewards.termination_penalty.weight = 0.0
        
        self.rewards.undesired_contacts = None # TODO
        
        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.ang_vel_xy_l2.weight = 0.0
        # self.rewards.dof_torques_l2.weight = 0.0
        # self.rewards.dof_acc_l2.weight = 0.0
        # self.rewards.action_rate_l2.weight = 0.0
        # self.rewards.feet_air_time.weight = 0.0
        # self.rewards.feet_slide.weight = 0.0
        # self.rewards.flat_orientation_l2.weight = 0.0
        # self.rewards.dof_pos_limits.weight = 0.0
        # self.rewards.joint_deviation_hip.weight = 0.0
        # self.rewards.joint_deviation_arms.weight = 0.0
        # self.rewards.joint_deviation_fingers.weight = 0.0
        # self.rewards.joint_deviation_torso.weight = 0.0
        
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.feet_slide = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_pos_limits = None
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        
        # motion loader
        self.motion_file_path = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "g1", "retargeted_motion.pkl"
        )
        self.motion_cfg_path = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "g1", "retargeted.yaml"
        )


@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
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
