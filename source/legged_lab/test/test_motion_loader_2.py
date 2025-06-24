import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

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
    
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True # type: ignore
    
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
        
        # dof_pos = mini_batch[:env.num_envs, 10:10+27]
        # robot.write_joint_position_to_sim(dof_pos)
        
        # base_lin_vel_b = mini_batch[:env.num_envs, 0:3]
        # base_ang_vel_b = mini_batch[:env.num_envs, 3:6]
        # projected_gravity = mini_batch[:env.num_envs, 6:9]
        # base_pos_z = mini_batch[:env.num_envs, 9:10]
        
        # print(f"Minibatch {count} data:")
        # print("Base linear velocity (body frame):", base_lin_vel_b)
        # print("Base angular velocity (body frame):", base_ang_vel_b)
        # print("Projected gravity:", projected_gravity)
        # print("Base position z:", base_pos_z)
        
        # # only render, no physics step
        # env.sim.render()
        
        count += 1
        
    env.close()
            

if __name__ == "__main__":
    main()
    simulation_app.close()  # Close the simulation app when done