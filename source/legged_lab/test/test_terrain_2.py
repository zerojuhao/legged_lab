import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from legged_lab.tasks.locomotion.amp.config.g1.amp_rough_env_cfg import G1AmpRoughEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv

import matplotlib.pyplot as plt

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

MY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.0,
            stone_width_range=(0.4, 1.0),
            stone_distance_range=(0.1, 0.3),
            holes_depth=-5.0,
            platform_width=2.0,
            border_width=0.25,
        ),
    }
)

def main():
    env_cfg = G1AmpRoughEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # fix the root link of the robot
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True # type: ignore
    env_cfg.actions.joint_pos.use_default_offset = False
    env_cfg.actions.joint_pos.scale = 1.0
    
    # terrain
    env_cfg.scene.terrain.terrain_generator = MY_TERRAINS_CFG
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    
    env_cfg.events.reset_base_rsi.params["pos_rsi"] = False
    
    env = AmpEnv(cfg=env_cfg)
    robot: Articulation = env.scene["robot"]
    
    obs, _ = env.reset()
    
    count = 0
    sim_time = 0.0
    sim_dt = env.step_dt
    print("Simulation time step:", sim_dt)
    

    while simulation_app.is_running():
        
        with torch.inference_mode():

            action = robot.data.default_joint_pos.clone()
            
            obs, _, _, _, _ = env.step(action)
            
            count += 1
            
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()
    
    
    
