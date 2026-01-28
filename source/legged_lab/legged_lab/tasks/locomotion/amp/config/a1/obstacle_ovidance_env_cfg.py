import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg, MotionDataCfg
import os
##
# Pre-defined configs
##

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.sensors import RayCasterArrayCfg
from legged_lab.envs import ManagerBasedAmpEnvCfg
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, RayCasterCameraCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.terrains.config.rough import OBSTACLE_TERRAINS_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG
from legged_lab import LEGGED_LAB_ROOT_DIR

KEY_BODY_NAMES = [
    "FL_calf", 
    "FR_calf",
    "RL_calf",
    "RR_calf"
]

ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3

# marker setting（red = on the way, green = arrive）
GOAL_POSITION_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target_far": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_near": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

AMP_NUM_STEPS = 3

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_TERRAINS_CFG,
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
    
    robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.0, 1.0]),
        debug_vis=False,
        update_period=0.02,
        mesh_prim_paths=["/World/ground"],
    )
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment='base',
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-60.0, 60.0], horizontal_res=5
        ),
        debug_vis= False,  # Enable debug visualization for the ray caster sensor
    )
    


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(20,20),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-15,15), pos_y=(-15,15), heading=(-math.pi, math.pi)),
        sample_on_edge_only=True,
        goal_reached_threshold=0.5,
        goal_pose_visualizer_cfg = GOAL_POSITION_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_goal"),  
    )
    

    
    
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg():
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.35, n_max=0.35))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.75, n_max=1.75))
        actions = ObsTerm(func=mdp.last_action)
        ray_caster = ObsTerm(
            func=mdp.ray_caster,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
            clip=(0.2, 5.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        ray_caster = ObsTerm(
            func=mdp.ray_caster,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
            clip=(0.2, 5.0),
        )
        
        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()

        
    @configclass
    class AmpCfg(ObsGroup):        
        # base_lin_vel_b: ObsTerm = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_b: ObsTerm = ObsTerm(func=mdp.base_ang_vel)
        # projected_gravity: ObsTerm = ObsTerm(func=mdp.projected_gravity)
        dof_pos: ObsTerm = ObsTerm(func=mdp.joint_pos)
        dof_vel: ObsTerm = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 3
            self.flatten_history_dim = False    # if True, it will flatten each term history first and then concatenate them, 
                                                # which is not we want for AMP observations
                                                # Thus, we set it to False, and address it manually
    # AMP observations group
    amp: AmpCfg = AmpCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.1, 1.3),
    #         "dynamic_friction_range": (0.1, 0.8),
    #         "restitution_range": (0.0, 0.5),
    #         "num_buckets": 64,
    #         "make_consistent": True,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (-3.0, 3.0),
    #         "operation": "add",
    #     },
    # )


    # randomize_rigid_body_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
    #         "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}, # 0.02
    #     },
    # )
    
    # scale_link_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    # scale_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
    #         "stiffness_distribution_params": (0.85, 1.15),
    #         "damping_distribution_params": (0.85, 1.15),
    #         "operation": "scale",
    #     },
    # )

    
    # scale_joint_parameters = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
    #         "friction_distribution_params": (1.0, 1.0),
    #         "armature_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )
    
    # reset
    reset_base=EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints=EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-1.0, 1.0)}},
    # )
    
    
@configclass
class RewardsCfg():
    """Reward terms for the MDP."""
    staged_navigation_reward = RewTerm(
        func=mdp.staged_navigation_reward,
        weight=0,
    )


    # -- Task
    reach_pos_target_soft = RewTerm(
        func=mdp.reach_pos_target_soft,
        weight=0,
        params={
            "position_target_sigma_soft": 2.0,
        },
    )
    
    reach_pos_target_tight = RewTerm(
        func=mdp.reach_pos_target_tight,
        weight=0,
        params={
            "position_target_sigma_tight": 0.5,
        },
    )
    
    reach_heading_target = RewTerm(
        func=mdp.reach_heading_target,
        weight=0,
        params={
            "heading_target_sigma": 0.1,
            "position_target_sigma_soft": 2.0,
        },
    )
    
    reach_pos_target_times_heading = RewTerm(
        func=mdp.reach_pos_target_times_heading,
        weight=0,
        params={
            "position_target_sigma": 0.5,
        },
    )
    
    velo_dir = RewTerm(
        func=mdp.velo_dir,
        weight=0,
        params={
            "position_target_sigma_tight": 2.0,
        },  
    )
    
    stand_still_pos = RewTerm(
        func=mdp.stand_still_pos,
        weight=0,
        params={
            "position_target_sigma_tight": 0.5,
        },  
    )
    
    nomove = RewTerm(
        func=mdp.nomove,
        weight=0,
        params={
            "position_target_sigma_soft": 2.0,
        },
    )
    
    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0)
    
    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0)
    
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )

    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-100,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    
    
    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*foot.*).*"]),
        },
    )


@configclass
class MotionDataCfg(MotionDataCfg):
    """Configuration for the Atom01 walk motion data."""

    # motion data term
    motion_dataset = MotionDataTerm(
        motion_data_dir=os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData","a1_lab"),
        motion_data_weights={
            
            'canter0':1,
            'canter1':1,
            'canter2':1,
            'left_turn0':1,
            'left_turn1':1,
            # 'pace0':1,
            # 'pace1':1,
            # 'pace2':1,
            'right_turn0':1,
            'right_turn1':1,
            'trot0':1,
            'trot1':1,
            'trot2':1,
        },
    )

@configclass
class AnimationCfg:
    """Animation settings for the MDP."""
    animation = AnimTerm(
        motion_data_term="motion_dataset",
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ], 
        num_steps_to_use=10, 
        random_initialize=True,
        random_fetch=True,
        resample_motion_on_update=False,
        enable_visualization=False,
        velocity_blend_ratio =1.0,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.0},
        time_out=True,
    )
    
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(60.0),
        },
    )
        
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
@configclass
class OAEnvCfg(ManagerBasedAmpEnvCfg):
    # scene
    scene: SceneCfg = SceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Motion data
    motion_data: MotionDataCfg = MotionDataCfg()

    def __post_init__(self):
        # post init of parent
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

        # self.rewards.reach_pos_target_soft.weight = 60
        # self.rewards.reach_pos_target_tight.weight = 1
        # self.rewards.reach_heading_target.weight = 1
        # self.rewards.reach_pos_target_times_heading.weight = 1
        # self.rewards.velo_dir.weight = 0.2
        # self.rewards.stand_still_pos.weight = 1
        # self.rewards.nomove.weight = -1

        # obstacle avoidance rewards
        self.rewards.staged_navigation_reward.weight = 2.0
        self.rewards.alive.weight = 0.5
        
        # base
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -1.0
        
        # joint
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
          
        # feet
        self.rewards.feet_slide.weight = -0.1 # -0.3
        self.rewards.feet_stumble.weight = -0.5  # -1.0
        
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*foot.*).*"],  # exclude ankle links
        )
        self.disable_zero_weight_rewards()
        
        
    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)

            
@configclass
class OAEnvCfg_PLAY(OAEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        
        # self.commands.pose_command.sample_on_edge_only = False
        # self.commands.pose_command.ranges.pos_x = (15, 15)
        # self.commands.pose_command.ranges.pos_y = (15, 15)
        
        self.commands.pose_command.sample_on_edge_only = True
        self.commands.pose_command.ranges.pos_x = (-15, 15)
        self.commands.pose_command.ranges.pos_y = (-15, 15)
        
        # disable randomization for play
        self.observations.policy.enable_corruption = True

