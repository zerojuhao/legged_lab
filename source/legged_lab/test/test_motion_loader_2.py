import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# run in headless mode in this test script
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv
from legged_lab.tasks.locomotion.amp.utils_amp.motion_loader import MotionLoader
from legged_lab import LEGGED_LAB_ROOT_DIR

import os
import time

def main():
    
    env_cfg = G1AmpFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.dt = 0.01  # Set simulation time step
    env = AmpEnv(cfg=env_cfg)   # motion loader is initialized inside AmpEnv
    robot:Articulation = env.scene["robot"]
    env_origins = env.scene.env_origins
    
    motion_loader_generator = env.motion_loader.mini_batch_generator(
        num_transitions_per_env=24, 
        num_mini_batches=4, 
        num_epochs=5
    )
    
    count = 0
    start_time =time.time()
    for mini_batch in motion_loader_generator:
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Minibatch {count} loading time: {duration}")
        
        start_time = time.time()
        
        count += 1
            

if __name__ == "__main__":
    main()
    simulation_app.close()  # Close the simulation app when done