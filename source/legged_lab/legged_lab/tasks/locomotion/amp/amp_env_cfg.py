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
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.sensors import RayCasterArrayCfg
from legged_lab.envs import ManagerBasedAmpEnvCfg
from legged_lab.managers import MotionDataTermCfg


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


@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    height_scanner = RayCasterArrayCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterArrayCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class AmpObservationsCfg():
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.05))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Gnoise(mean=0.0, std=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0, std=0.05))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()
    
    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan_ch,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Gnoise(mean=0.0, std=0.05),
            clip=(-1.0, 1.0),
        )
        
        def __post_init__(self):
            self.enable_corruption = True
    
    image: HeightScanCfg = HeightScanCfg()
        
    @configclass
    class AmpCfg(ObsGroup):        
        base_lin_vel_b: ObsTerm = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_b: ObsTerm = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity: ObsTerm = ObsTerm(func=mdp.projected_gravity)        
        base_pos_z: ObsTerm = ObsTerm(func=mdp.base_pos_z)  # TODO: consider terrain height
        dof_pos: ObsTerm = ObsTerm(func=mdp.joint_pos)
        dof_vel: ObsTerm = ObsTerm(func=mdp.joint_vel)
        # key_links_pos_b: ObsTerm = ObsTerm(
        #     func=mdp.key_links_pos_b, 
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"), 
        #         "local_pos_dict": MISSING,
        #     }
        # )
    
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 1
            self.flatten_history_dim = False    # if True, it will flatten each term history first and then concatenate them, 
                                                # which is not we want for AMP observations
                                                # Thus, we set it to False, and address it manually
    # AMP observations group
    amp: AmpCfg = AmpCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1, 1), "y": (-1, 1), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1.0, 1.0),
    #         "velocity_range": (-0.1, 0.1),
    #     },
    # )
    
    reset_robot_joints_rsi = EventTerm(
        func=mdp.ref_state_init_dof,
        mode="reset",
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    alive = RewTerm(
        func=mdp.is_alive, weight=0
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    dof_energy = RewTerm(func=mdp.joint_energy, weight=0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh_roll_link"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(45.0),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # curriculum for velocity command dose not work well now, so we disable them


    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(
    #     func=mdp.lin_vel_cmd_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "lin_vel_x_limit": [-0.5, 1.0],
    #         "lin_vel_y_limit": [-0.3, 0.3],
    #     }
    # )
    # ang_vel_cmd_levels = CurrTerm(
    #     func=mdp.ang_vel_cmd_levels,
    #     params={
    #         "reward_term_name": "track_ang_vel_z_exp",
    #         "ang_vel_z_limit": [-0.2, 0.2],
    #     }
    # )

@configclass
class MotionDataCfg:
    pass

    # # You can add dataset config here
    # dataset = MotionDataTermCfg(
    #     motion_data_dir=MISSING, 
    #     motion_data_weight=MISSING,
    #     dof_names=MISSING,
    #     key_links_mapping=MISSING,
    # )


##
# Environment configuration
##


@configclass
class LocomotionAmpEnvCfg(ManagerBasedAmpEnvCfg):
    """Configuration for the AMP locomotion environment."""

    # scene
    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Motion data
    motion_data: MotionDataTermCfg = MotionDataTermCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
