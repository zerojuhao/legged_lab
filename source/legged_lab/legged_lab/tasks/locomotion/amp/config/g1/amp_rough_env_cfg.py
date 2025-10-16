import os
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.managers import MotionDataTermCfg
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg, MotionDataCfg

import isaaclab.terrains as terrain_gen

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

from legged_lab import LEGGED_LAB_ROOT_DIR

MOTIONDATA_DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

AMP_NUM_STEPS = 3

@configclass
class G1AmpRewards():
    """Reward terms for the MDP."""

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    
    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0,
        params={
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner")
        },
    )
    
    # -- Joint
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    dof_energy = RewTerm(func=mdp.joint_energy, weight=-2e-5)
    
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )
    
    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # feet_clearance = RewTerm(
    #     func=mdp.feet_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "base_height": 0.78,
    #         "target_feet_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5, 
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        }
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

@configclass
class G1WalkMotionDataCfg(MotionDataCfg):
    """Configuration for the G1 walk motion data."""

    # motion data term
    dataset: MotionDataTermCfg = MotionDataTermCfg(
        weight=1.0,
        motion_data_dir=os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "walk"),
        motion_data_weight={
            # "B9_-__Walk_turn_left_90_stageii": 1.0,
            "B10_-__Walk_turn_left_45_stageii": 1.0,
            # "B11_-__Walk_turn_left_135_stageii": 1.0, 
            # "B13_-__Walk_turn_right_90_stageii": 1.0, 
            "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            "B15_-__Walk_turn_around_stageii": 1.0, 
            "B22_-__side_step_left_stageii": 1.0, 
            "B23_-__side_step_right_stageii": 1.0, 
            "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
        },
        dof_names=MOTIONDATA_DOF_NAMES,
        key_links_mapping={
            # the keys are the names of the links in the motion dataset
            # the values are the names of the links in lab 
            "left_ankle_roll_link": "left_ankle_roll_link",
            "right_ankle_roll_link": "right_ankle_roll_link",
            "left_wrist_yaw_link": "left_wrist_yaw_link",
            "right_wrist_yaw_link": "right_wrist_yaw_link",
        },
        num_steps=AMP_NUM_STEPS,
    )

@configclass
class G1AmpRoughEnvCfg(LocomotionAmpEnvCfg):
    rewards: G1AmpRewards = G1AmpRewards()
    motion_data: G1WalkMotionDataCfg = G1WalkMotionDataCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # height scanner
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        # terrain
        self.scene.terrain.terrain_generator.sub_terrains = {   # type: ignore
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.4,
            ),
            # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            #     proportion=0.2,
            #     step_height_range=(0.05, 0.23),
            #     step_width=0.3,
            #     platform_width=3.0,
            #     border_width=1.0,
            #     holes=False,
            # ),
            # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            #     proportion=0.2,
            #     step_height_range=(0.05, 0.23),
            #     step_width=0.3,
            #     platform_width=3.0,
            #     border_width=1.0,
            #     holes=False,
            # ),
            # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            # ),
            # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # ),
            # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # ),
            "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.6,
                stone_height_max=0.0,
                stone_width_range=(0.4, 1.0),
                stone_distance_range=(0.0, 0.3),
                holes_depth=-5.0,
                platform_width=2.0,
                border_width=0.25,
            ),
        }
        self.scene.terrain.terrain_generator.curriculum = True  # type: ignore
        # make sure the curriculum is enabled
        
        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "walk"
        )
        
        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        # self.observations.policy.base_lin_vel = None
        self.observations.amp.key_links_pos_b.params["local_pos_dict"] = {
            "left_ankle_roll_link": (0.0, 0.0, 0.0),
            "right_ankle_roll_link": (0.0, 0.0, 0.0),
            "left_wrist_yaw_link": (0.0, 0.0, 0.0),
            "right_wrist_yaw_link": (0.0, 0.0, 0.0),
            # "waist_yaw_link": (0.0, 0.0, 0.4),
            # "left_shoulder_roll_link": (0.0, 0.0, 0.0),
            # "right_shoulder_roll_link": (0.0, 0.0, 0.0),
            # "left_hip_pitch_link": (0.0, 0.0, 0.0),
            # "right_hip_pitch_link": (0.0, 0.0, 0.0),
            # "left_elbow_link": (0.0, 0.0, 0.0),
            # "right_elbow_link": (0.0, 0.0, 0.0),
            # "left_knee_link": (0.0, 0.0, 0.0),
            # "right_knee_link": (0.0, 0.0, 0.0),
        }
        self.observations.amp.history_length = AMP_NUM_STEPS
        
        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "waist_yaw_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
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
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)
        
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        
        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "waist_yaw_link", "pelvis", ".*_shoulder_.*_link", ".*_elbow_link",
        ]
        

@configclass
class G1AmpRoughEnvCfg_PLAY(G1AmpRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.scene.height_scanner.debug_vis = True


