# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_vel.scale = 0.05
        # no height scan
        self.observations.policy.height_scan = None
        
        self.observations.critic.base_ang_vel.scale = 0.2
        self.observations.critic.joint_vel.scale = 0.05
        # no height scan
        self.observations.critic.height_scan = None
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        
        self.rewards.alive.weight = 0.15
        
        # base
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height.weight = -10.0
        self.rewards.base_height.params["target_height"] = 0.78
        self.rewards.base_height.params["sensor_cfg"] = None  # no height scanner
        
        # joint
        self.rewards.dof_vel_l2.weight = -0.001
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.dof_energy.weight = -2e-5
        
        # feet
        self.rewards.feet_air_time = None
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_clearance.weight = 1.0
        # self.rewards.feet_clearance.params["target_feet_height"] = 0.15
        self.rewards.feet_gait.weight = 0.5
        
        # deviation
        self.rewards.joint_deviation_hip.weight = -1.0
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_waist.weight = -1.0

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],  # exclude ankle links
        )

class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
