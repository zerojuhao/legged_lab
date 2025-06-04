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

from legged_lab.tasks.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from legged_lab.tasks.locomotion.amp.utils_amp.motion_loader import MotionLoader
from legged_lab import LEGGED_LAB_ROOT_DIR

import os

def main():
    
    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.dt = 0.01  # Set simulation time step
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot:Articulation = env.scene["robot"]
    env_origins = env.scene.env_origins

    # motion data
    motion_file_path = os.path.join(
        LEGGED_LAB_ROOT_DIR, "data", "g1", "retargeted_motion.pkl"
    )
    motion_cfg_path = os.path.join(
        LEGGED_LAB_ROOT_DIR, "data", "g1", "retargeted.yaml"
    )
    
    motion_loader = MotionLoader(
        motion_file=motion_file_path,
        cfg_file=motion_cfg_path,
        entity=env.scene["robot"],
        device=args_cli.device
    )
    
    motion_ids = motion_loader.sample_motions(args_cli.num_envs)
    print("Sampled motion IDs:", motion_ids)
    motion_durations = motion_loader.get_motion_duration(motion_ids)

    sim_dt = env.sim.get_physics_dt()
    print("Simulation DT:", sim_dt)
    sim_time = 0.0
    count = 0
    env.reset()
    print("Environment reset complete. Starting simulation...")
    while simulation_app.is_running():
        with torch.inference_mode():
            
            motion_times = sim_time % motion_durations
            
            motion_data = motion_loader.get_motion_state(motion_ids, motion_times)
            
            root_pos = motion_data["root_pos_w"]
            root_quat = motion_data["root_quat"]
            dof_pos = motion_data["dof_pos"]
            
            robot_root_state = robot.data.default_root_state.clone()
            robot_root_state[:, :3] = root_pos + env_origins
            robot_root_state[:, 3:7] = root_quat
            robot.write_root_pose_to_sim(robot_root_state[:, :7])
            robot_dof_pos = robot.data.default_joint_pos.clone()
            robot_dof_pos[:, :len(dof_pos[0])] = dof_pos
            robot.write_joint_position_to_sim(robot_dof_pos)
            
            # only render, no physics step
            env.sim.render()
            
            count += 1
            sim_time += sim_dt
            

if __name__ == "__main__":
    main()
    simulation_app.close()  # Close the simulation app when done