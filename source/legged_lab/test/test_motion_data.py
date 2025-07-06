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

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv

import matplotlib.pyplot as plt
 
 
def main():
    env_cfg = G1AmpFlatEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # fix the root link of the robot
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True # type: ignore
    env_cfg.actions.joint_pos.use_default_offset = False
    env_cfg.actions.joint_pos.scale = 1.0
    
    env = AmpEnv(cfg=env_cfg)
    env_origins = env.scene.env_origins
    robot: Articulation = env.scene["robot"]
    
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
    
    
    count = 0
    sim_time = 0.0
    sim_dt = env.step_dt
    print("Simulation time step:", sim_dt)
    
    obs, _ = env.reset()

    amp_obs_dict = obs["amp"]
    # print the shape of values in amp_obs_dict
    print("AMP Observations:")
    for key, value in amp_obs_dict.items():
        print(f"{key}: {value.shape}")
    print("-"*50)
        
    motion_ids = env.motion_loader.sample_motions(args_cli.num_envs)
    print("Sampled motion IDs:", motion_ids)
    motion_durations = env.motion_loader.get_motion_duration(motion_ids)
    
    # print joint names
    dof_names = env.scene["robot"].data.joint_names
    print("Joint names:", dof_names)    

    # for plotting 
    sim_length = 250
    # only record the first env
    num_dof = len(dof_names)
    record_dof_pos = torch.zeros((sim_length, num_dof), device=env.sim.device)
    record_dof_vel = torch.zeros((sim_length, num_dof), device=env.sim.device)
    record_dof_pos_ref = torch.zeros((sim_length, num_dof), device=env.sim.device)  
    record_dof_vel_ref = torch.zeros((sim_length, num_dof), device=env.sim.device)
    
    record_dof_effort = torch.zeros((sim_length, num_dof), device=env.sim.device)
    
    record_key_link_pos_b = torch.zeros((sim_length, 4*3), device=env.sim.device)
    record_key_link_pos_b_ref = torch.zeros((sim_length, 4*3), device=env.sim.device)


    while simulation_app.is_running():
        with torch.inference_mode():
            motion_times = sim_time % motion_durations
            motion_data_dict = env.motion_loader.get_motion_state(motion_ids, motion_times)
            
            action = motion_data_dict["dof_pos"]
            
            obs, _, _, _, _ = env.step(action)
            
            amp_obs_dict = obs["amp"]
            amp_obs = torch.cat(list(amp_obs_dict.values()), dim=-1).reshape(args_cli.num_envs, -1)
            # record_dof_pos[count, :] = amp_obs_dict["dof_pos"][0, 1, :] # dim 0: num_envs, dim 1: history length, dim 2: dof
            # record_dof_vel[count, :] = amp_obs_dict["dof_vel"][0, 1, :] # dim 0: num_envs, dim 1: history length, dim 2: dof

            record_dof_pos[count, :] = amp_obs[0, 0:num_dof]
            record_dof_vel[count, :] = amp_obs[0, num_dof:num_dof*2]
            
            # record_dof_pos_ref[count, :] = amp_obs[0, num_dof*2+3*9:num_dof*2+3*9+num_dof] 
            record_dof_pos_ref[count, :] = motion_data_dict["dof_pos"][0, :]  # dim 0: num_envs, dim 1: dof
            record_dof_vel_ref[count, :] = motion_data_dict["dof_vel"][0, :]
            
            record_dof_effort[count, :] = robot.data.applied_torque[0, :]
            
            # record_key_link_pos_b[count, :] = amp_obs_dict["key_links_pos_b"][0, 1, 12:24]
            record_key_link_pos_b[count, :] = amp_obs[0, num_dof*2+ 5*3:num_dof*2 + 9*3] 
            record_key_link_pos_b_ref[count, :] = motion_data_dict["key_links_pos_b"][0, 5:9].flatten()

            
            # visualize key links in the simulation app
            # key_links_pos_b = motion_data_dict["key_links_pos_b"]
            key_links_pos_b = amp_obs_dict["key_links_pos_b"][:, 1, :].reshape(args_cli.num_envs, -1, 3)  # shape: (N, M, 3), N is number of envs, M is number of key links
            num_key_links = key_links_pos_b.shape[1]
            root_pos_w = robot.data.root_pos_w.unsqueeze(1) # shape: (N, 1, 3)
            root_quat_w = robot.data.root_quat_w.unsqueeze(1)  # shape: (N, 1, 4)
            key_links_pos_w = root_pos_w + math_utils.quat_apply(root_quat_w.expand(-1, num_key_links, -1), key_links_pos_b)
            
            marker_vis.visualize(
                translations=key_links_pos_w.view(-1, 3)
            )

            count += 1
            sim_time += sim_dt

            if count >= sim_length:
                break

    env.close()
    
    # Plot the recorded data
    # First figure: DOF positions
    plt.figure(num=1, figsize=(15, 10))
    for i in range(27):
        plt.subplot(9, 3, i + 1)
        plt.plot(record_dof_pos[:, i].cpu().numpy(), label='Dof Pos', color='blue')
        plt.plot(record_dof_pos_ref[:, i].cpu().numpy(), label='Dof Pos Ref', color='orange')
        dof_name = dof_names[i]
        plt.title(f'{dof_name} Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()

    # Second figure: DOF velocities
    plt.figure(num=2, figsize=(15, 10))
    for i in range(27):
        plt.subplot(9, 3, i + 1)
        plt.plot(record_dof_vel[:, i].cpu().numpy(), label='Dof Vel', color='green')
        plt.plot(record_dof_vel_ref[:, i].cpu().numpy(), label='Dof Vel Ref', color='red')
        dof_name = dof_names[i]
        plt.title(f'{dof_name} Velocity')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity')
        plt.legend()

    # Third figure: DOF efforts
    plt.figure(num=3, figsize=(15, 10))
    for i in range(27):
        plt.subplot(9, 3, i + 1)
        plt.plot(record_dof_effort[:, i].cpu().numpy(), label='Dof Effort', color='purple')
        dof_name = dof_names[i]
        plt.title(f'{dof_name} Effort')
        plt.xlabel('Time Step')
        plt.ylabel('Effort')
        plt.legend()

    # Fourth figure: Key link positions in base frame
    plt.figure(num=4, figsize=(15, 10))
    num_key_links = int(record_key_link_pos_b.shape[1] / 3)
    for i in range(num_key_links):
        # x
        plt.subplot(num_key_links, 3, i*3 + 1)
        plt.plot(record_key_link_pos_b[:, i*3].cpu().numpy(), label='Key Link Pos X', color='cyan')
        plt.plot(record_key_link_pos_b_ref[:, i*3].cpu().numpy(), label='Key Link Pos X Ref', color='magenta')
        plt.title(f'Key Link {i} X Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        # y
        plt.subplot(num_key_links, 3, i*3 + 2)
        plt.plot(record_key_link_pos_b[:, i*3 + 1].cpu().numpy(), label='Key Link Pos Y', color='cyan')
        plt.plot(record_key_link_pos_b_ref[:, i*3 + 1].cpu().numpy(), label='Key Link Pos Y Ref', color='magenta')
        plt.title(f'Key Link {i} Y Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        # z
        plt.subplot(num_key_links, 3, i*3 + 3)
        plt.plot(record_key_link_pos_b[:, i*3 + 2].cpu().numpy(), label='Key Link Pos Z', color='cyan')
        plt.plot(record_key_link_pos_b_ref[:, i*3 + 2].cpu().numpy(), label='Key Link Pos Z Ref', color='magenta')
        plt.title(f'Key Link {i} Z Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
    simulation_app.close()