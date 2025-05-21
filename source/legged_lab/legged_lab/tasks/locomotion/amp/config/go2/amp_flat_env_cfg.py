from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2AmpFlatEnvCfg(LocomotionAmpEnvCfg):
    """
    Environment configuration for the AMP locomotion task.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # observations
        self.observations.amp.feet_pos.params["asset_cfg"].body_names = ".*_foot"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        
@configclass
class Go2AmpFlatEnvCfg_PLAY(Go2AmpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        