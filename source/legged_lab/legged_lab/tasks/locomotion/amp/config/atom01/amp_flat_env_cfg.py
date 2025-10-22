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

from legged_lab.assets.roboparty import ATOM01_CFG
from legged_lab import LEGGED_LAB_ROOT_DIR


MOTIONDATA_DOF_NAMES = [
    'left_thigh_yaw_joint',
    'left_thigh_roll_joint',
    'left_thigh_pitch_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_thigh_yaw_joint',
    'right_thigh_roll_joint',
    'right_thigh_pitch_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'torso_joint',
    'left_arm_pitch_joint',
    'left_arm_roll_joint',
    'left_arm_yaw_joint',
    'left_elbow_pitch_joint',
    'left_elbow_yaw_joint',
    'right_arm_pitch_joint',
    'right_arm_roll_joint',
    'right_arm_yaw_joint',
    'right_elbow_pitch_joint',
    'right_elbow_yaw_joint',
]

AMP_NUM_STEPS = 3



@configclass
class Atom01AmpRewards():
    """Reward terms for the MDP."""

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0, params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0)
    
    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)

    # -- Joint
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    dof_energy = RewTerm(func=mdp.joint_energy, weight=0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.0,
    )

    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
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
class Atom01WalkMotionDataCfg(MotionDataCfg):
    """Configuration for the Atom01 walk motion data."""

    # motion data term
    dataset: MotionDataTermCfg = MotionDataTermCfg(
        weight=1.0,
        # motion_data_dir=os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "atom01"),
        # motion_data_weight={
        #     "02_01_stageii":0.7,
        #     "39_11_stageii":0.15,
        #     "47_01_stageii":0.15,
        # },
        motion_data_dir=os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData","ACCAD", "Male1Walking"),
        motion_data_weight={
            '02_01_stageii': 0.1,
            '39_11_stageii': 0.1,
            '47_01_stageii': 0.1,

            # female walking datasets
            # "B1_-_stand_to_walk_stageii":0.05,
            # "B2_-_walk_to_stand_stageii":0.05,
            # "B2_-_walk_to_stand_t2_stageii":0.05,
            # "B3_-_walk1_stageii":0.05,
            # "B4_-_stand_to_walk_back_stageii":0.05,
            # "B5_-_walk_backwards_stageii":0.1,
            # "B6_-_walk_backwards_to_stand_stageii":0.05,
            # "B7_-_walk_backwards_turn_forwards_stageii":0.05,
            # "B9_-_walk_turn_left_(90)_stageii":0.05,
            # "B10_-_walk_turn_left_(45)_stageii":0.1,
            # "B11_-_walk_turn_left_(135)_stageii":0.05,
            # "B12_-_walk_turn_right_(90)_stageii":0.05,
            # "B13_-_walk_turn_right_(45)_stageii":0.1,
            # "B14_-_walk_turn_right_(135)_stageii":0.05,
            # "B15_-_walk_turn_around_(same_direction)_stageii":0.05,
            # "B16_-_walk_turn_change_direction_stageii":0.05,



            # male walking datasets
            'B4_-_Stand_to_Walk_backwards_stageii':0.1,
            'B5_-__Walk_backwards_stageii':0.1,
            'B9_-__Walk_turn_left_90_stageii':0.1,
            'B10_-__Walk_turn_left_45_stageii':0.1,
            # 'B11_-__Walk_turn_left_135_stageii':0.1,
            'B13_-__Walk_turn_right_90_stageii':0.1,
            'B14_-__Walk_turn_right_45_t2_stageii':0.1,
            'B15_-__Walk_turn_around_stageii':0.1,
        },        
        dof_names=MOTIONDATA_DOF_NAMES,
        key_links_mapping={},
        # key_links_mapping={
            # the keys are the names of the links in the motion dataset
            # the values are the names of the links in lab 

            # "left_ankle_roll_link": "left_ankle_roll_link",
            # "right_ankle_roll_link": "right_ankle_roll_link",
            # "left_wrist_yaw_link": "left_wrist_yaw_link",
            # "right_wrist_yaw_link": "right_wrist_yaw_link", 
            #     
        # },
        num_steps=AMP_NUM_STEPS,
    )


@configclass
class Atom01AmpFlatEnvCfg(LocomotionAmpEnvCfg):
    rewards: Atom01AmpRewards = Atom01AmpRewards()
    motion_data: Atom01WalkMotionDataCfg = Atom01WalkMotionDataCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        
        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.dataset.motion_data_dir = os.path.join(
            # LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_27dof", "walk"
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "ACCAD", "Male1Walking"
        )

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        # no height scan
        self.observations.image = None
        self.observations.amp.history_length = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        # self.curriculum.terrain_levels = None
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.stand_still.weight = 1.0

        self.rewards.alive.weight = 0.1
        
        # base
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        
        # joint
        self.rewards.dof_vel_l2.weight = -0.001
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_pos_limits.weight = -1
        self.rewards.dof_energy.weight = -2e-5
        
        # feet
        self.rewards.feet_slide.weight = -1.0

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],  # exclude ankle links
        )
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.785, 0.785)

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            ".*_thigh_.*_link", "base_link", ".*_arm_.*_link", ".*_elbow_.*_link",
        ]

@configclass
class Atom01AmpFlatEnvCfg_PLAY(Atom01AmpFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
