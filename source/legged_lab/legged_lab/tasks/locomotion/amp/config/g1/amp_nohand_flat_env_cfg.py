from isaaclab.utils import configclass
import isaaclab.utils.string as string_utils

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg

G1_NOHAND_CFG:ArticulationCfg = G1_MINIMAL_CFG.copy()
G1_NOHAND_CFG.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        )

joint_names_without_hand = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 
    'torso_joint', 
    'left_hip_roll_joint', 'right_hip_roll_joint', 
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
    'left_knee_joint', 'right_knee_joint', 
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
    'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 
    'left_ankle_roll_joint', 'right_ankle_roll_joint', 
    'left_elbow_roll_joint', 'right_elbow_roll_joint'
]


asset_cfg = SceneEntityCfg(
    name="robot",
    joint_names= joint_names_without_hand,
)

@configclass
class G1AmpNoHandFlatEnvCfg(G1AmpFlatEnvCfg):
    
    def __post_init__(self):
        
        # policy observations
        self.observations.policy.joint_pos.params["asset_cfg"] = asset_cfg
        self.observations.policy.joint_vel.params["asset_cfg"] = asset_cfg
        
        # amp observations
        self.observations.amp.dof_pos.params["asset_cfg"] = asset_cfg
        self.observations.amp.dof_vel.params["asset_cfg"] = asset_cfg
        
        # action space
        self.actions.joint_pos.joint_names = joint_names_without_hand
        
        
        # rewards
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
        
        super().__post_init__()
        
        
@configclass
class G1AmpNoHandFlatEnvCfg_PLAY(G1AmpNoHandFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.episode_length_s = 40.0
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
