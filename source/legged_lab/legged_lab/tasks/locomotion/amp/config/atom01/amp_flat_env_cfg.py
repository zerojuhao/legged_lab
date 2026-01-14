import os
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

from legged_lab.assets.roboparty import ATOM01_CFG, ATOM01_LONG_CFG
from legged_lab import LEGGED_LAB_ROOT_DIR
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# 此处为 IsaacLab 中机器人的关节顺序
ISAACLAB_JOINT_ORDER= [
    'left_thigh_yaw_joint', 
    'right_thigh_yaw_joint', 
    'torso_joint', 
    'left_thigh_roll_joint', 
    'right_thigh_roll_joint', 
    'left_arm_pitch_joint', 
    'right_arm_pitch_joint', 
    'left_thigh_pitch_joint', 
    'right_thigh_pitch_joint', 
    'left_arm_roll_joint', 
    'right_arm_roll_joint', 
    'left_knee_joint', 
    'right_knee_joint', 
    'left_arm_yaw_joint', 
    'right_arm_yaw_joint', 
    'left_ankle_pitch_joint', 
    'right_ankle_pitch_joint', 
    'left_elbow_pitch_joint', 
    'right_elbow_pitch_joint', 
    'left_ankle_roll_joint', 
    'right_ankle_roll_joint', 
    'left_elbow_yaw_joint', 
    'right_elbow_yaw_joint'
]

# 此处为数据集中的关节顺序
DATASET_JOINT_ORDER = [
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

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    # "left_ankle_roll_link", 
    # "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
    # "left_arm_roll_link",
    # "right_arm_roll_link",
    # "left_knee_link",
    # "right_knee_link",
    # "left_elbow_pitch_link",
    # "right_elbow_pitch_link",
    "left_arm_yaw_link",
    "right_arm_yaw_link"
] # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
ANIMATION_TERM_NAME = "animation"
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
    base_height = RewTerm(func=mdp.base_height, weight=0, params={"target_height": 0.5})

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )
    ankle_collision_penalty = RewTerm(
        func=mdp.ankle_collision_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "distance_threshold": 0.15,
        },
    )
    low_speed_sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
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
    
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), 
            "threshold": 0.4},
    )
    
    feet_distance = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.20, "max": 0.40},
    )
    
    knee_distance = RewTerm(
        func=mdp.knee_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*knee.*"]), "min": 0.15, "max": 0.30},
    )
    
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=0,   # ⚠️ 一定要小
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
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
class Atom01AmpFlatEnvCfg(LocomotionAmpEnvCfg):
    rewards: Atom01AmpRewards = Atom01AmpRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = ATOM01_LONG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "atom01_long_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights={
            
            # # # 4
            # '02_02': 1, # walk
            # '16_34': 1, # walk to stop
            
            # "127_03": 1, # stand to run
            "127_04": 1, # walk to run 2.1
            "127_06": 1, # run 3.6
            # "143_02": 1, # run to stop
            # "143_03": 1, # stand to run
            
            # # # standstill 1
            "A1-_Stand_stageii": 1,
            
            
            # # # male2 walk 8
            # "B4_-_Stand_to_Walk_backwards_stageii":1,
            "B9_-__Walk_turn_left_90_stageii":1,
            "B10_-__Walk_turn_left_45_stageii":1,
            "B13_-__Walk_turn_right_90_stageii":1,
            "B14_-__Walk_turn_right_45_t2_stageii":1,
            # "B15_-__Walk_turn_around_stageii_turn":1,
            "B15_-__Walk_turn_around_stageii_walk":1, # 1.2451
            # "B22_-__side_step_left_stageii":1,
            # "B23_-__side_step_right_stageii":1,
            
            # # male2 run 8
            # "C1_-_stand_to_run_stageii": 1,
            # "C3_-_run_stageii": 1,
            "C4_-_run_to_walk_a_stageii": 1, # 2.3204
            # "C4_-_run_to_walk_stageii":1,
            # "C5_-_walk_to_run_stageii":1,
            "C12_-_run_turn_left_45_stageii":1,
            # "C15_-_run_turn_right_45_stageii":1,
            "C17_-_run_change_direction_stageii":1,
            
            # # 自设动作 1
            "move_back":1,
            "move_l":1,
            "move_r":1,
            "turn_l":1,
            "turn_r":1,

        }
        
        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
                
        # discriminator observations
        
        # self.observations.disc.key_body_pos_b.params = {
        #     "asset_cfg": SceneEntityCfg(
        #         name="robot", 
        #         body_names=KEY_BODY_NAMES, 
        #         preserve_order=True
        #     )
        # }
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # discriminator demostration observations
        
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        # self.observations.disc_demo.ref_root_lin_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        # self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME
     
        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.25
        self.rewards.alive.weight = 0.15
        
        # base
        # self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height.weight = -10.0
        
        # joint
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_regularization.weight = -1e-4
        # self.rewards.low_speed_sway_penalty.weight = -1e-2
        
        # feet
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_stumble.weight = -0.1
        # self.rewards.sound_suppression.weight = -5e-3
        # self.rewards.feet_air_time_positive_biped.weight = 1.0


        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],  # exclude ankle links
        )
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        # self.commands.base_velocity.ranges.zero_prob = (0.05, 0.05, 0.05)  # 采样零速度

        # self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 2.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        # self.commands.base_velocity.ranges.zero_prob = (0.1, 0.1, 0.1)  # 采样零速度
                
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4,1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.zero_prob = (0.1, 0.1, 0.1)  # 采样零速度
                
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        
        # self.terminations.base_contact.params["sensor_cfg"].body_names = [
        #     ".*_thigh_.*_link", "base_link", ".*_arm_.*_link", ".*_elbow_.*_link",
        # ]
        if self.__class__.__name__ == "Atom01AmpFlatEnvCfg":
            self.disable_zero_weight_rewards()
            
            
@configclass
class Atom01AmpFlatEnvCfg_PLAY(Atom01AmpFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None
