import math
from isaaclab.utils import configclass
from legged_lab.tasks.locomotion.amp.config.g1.amp_rough_env_cfg import G1AmpRoughEnvCfg


@configclass
class G1AmpFlatEnvCfg(G1AmpRoughEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        
        # ------------------------------------------------------
        # motion loader
        # ------------------------------------------------------
        self.motion_loader.motion_weights = {
            # the motion names can be obtained by running `utils/print_motion_names.py`
            "08_09_poses": 1.0,      # walk fast in large step
            "08_03_poses": 1.0,      # walk fast in large step
            "08_04_poses": 1.0,      # walk slow in large step
            "77_02_poses": 1.0,      # stand
        }
        
        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        # no height scan
        self.observations.image = None
        
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.terrain_levels = None
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # For AMP, we only needs a few rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.alive.weight = 0.1
        
        self.rewards.dof_pos_limits.weight = -0.0
        self.rewards.joint_deviation_hip.weight = -0.0
        self.rewards.joint_deviation_arms.weight = -0.0
        self.rewards.joint_deviation_wrists.weight = -0.0
        self.rewards.joint_deviation_waist.weight = -0.0
        
        self.rewards.undesired_contacts.weight = 0.0
        
        self.rewards.feet_orientation_l2.weight = -1.0
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        

@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
