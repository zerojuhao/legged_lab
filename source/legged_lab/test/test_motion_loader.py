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

from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.envs import ManagerBasedAmpEnv
from legged_lab import LEGGED_LAB_ROOT_DIR

import os

def main():
    
    env_cfg = G1AmpFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.dt = 0.01  # Set simulation time step
    env = ManagerBasedAmpEnv(cfg=env_cfg)   # motion loader is initialized inside AmpEnv
    robot:Articulation = env.scene["robot"]
    env_origins = env.scene.env_origins

    # # motion data
    # motion_file_path = os.path.join(
    #     LEGGED_LAB_ROOT_DIR, "data", "g1", "retargeted_motion.pkl"
    # )
    # motion_cfg_path = os.path.join(
    #     LEGGED_LAB_ROOT_DIR, "data", "g1", "retargeted.yaml"
    # )
        
    # marker_cfg:VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FrameVisualizerFromScript",
        markers={
            "red_sphere": sim_utils.SphereCfg(
                radius=0.03, 
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
        }
    )
    marker_vis = VisualizationMarkers(marker_cfg)
    
    motion_dataset = env.motion_data_manager.active_terms[0]
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    
    motion_ids = motion_loader.sample_motions(args_cli.num_envs)
    print("Sampled motion IDs:", motion_ids)
    motion_durations = motion_loader.get_motion_duration(motion_ids)

    print(env.observation_manager.active_terms["amp"])
    
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
            root_vel_w = motion_data["root_vel_w"]
            root_ang_vel_w = motion_data["root_ang_vel_w"]
            
            root_vel_b = motion_data["root_vel_b"]
            root_ang_vel_b = motion_data["root_ang_vel_b"]
            
            key_links_pos_b = motion_data["key_links_pos_b"]    # shape: (N, M, 3), N is number of envs, M is number of key links
            num_key_links = key_links_pos_b.shape[1]
            key_links_pos_w = root_pos.unsqueeze(1) + math_utils.quat_apply(root_quat.unsqueeze(1).expand(-1, num_key_links, -1), 
                                                                            key_links_pos_b)
            key_links_pos_w += env_origins.unsqueeze(1)  # add environment origins
            
            marker_vis.visualize(
                translations=key_links_pos_w.view(-1, 3)
            )
            
            robot_root_state = robot.data.default_root_state.clone()
            robot_root_state[:, :3] = root_pos + env_origins
            robot_root_state[:, 3:7] = root_quat
            robot.write_root_pose_to_sim(robot_root_state[:, :7])
            root_vel = torch.cat([root_vel_w, root_ang_vel_w], dim=-1)
            robot.write_root_velocity_to_sim(root_vel)
            robot_dof_pos = robot.data.default_joint_pos.clone()
            robot_dof_pos[:, :len(dof_pos[0])] = dof_pos
            robot.write_joint_position_to_sim(robot_dof_pos)
            
            # only render, no physics step
            env.sim.render()
            
            # if count % 20 == 0:
            #     print("-" * 80)
            #     print(f"key_links_pos_b: {key_links_pos_b[0, :, :]}")
            #     print(f"root velocity in body frame: {root_vel_b[0, :]}")
            #     print(f"root angular velocity in body frame: {root_ang_vel_b[0, :]}")
            #     print(count)

            count += 1
            sim_time += sim_dt
            

if __name__ == "__main__":
    main()
    simulation_app.close()  # Close the simulation app when done