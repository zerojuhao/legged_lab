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

if not args_cli.headless:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import isaaclab.utils.math as math_utils

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv

 
def main():
    env_cfg = G1AmpFlatEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = AmpEnv(cfg=env_cfg)
    
    if not args_cli.headless:
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    
    count = 0
    obs, _ = env.reset()

    amp_obs_dict = obs["amp"]
    # print the shape of values in amp_obs_dict
    for key, value in amp_obs_dict.items():
        print(f"{key}: {value.shape}")
        
    # dof_pos: torch.Size([4, 2, 37])
    # dof_vel: torch.Size([4, 2, 37])
    # base_lin_vel_b: torch.Size([4, 2, 3])
    # base_ang_vel_b: torch.Size([4, 2, 3])
    # base_pos_z: torch.Size([4, 2, 1])
    # key_links_pos_b: torch.Size([4, 2, 48])
    
    while simulation_app.is_running():
        with torch.inference_mode():
                        
            robot:Articulation = env.scene["robot"]
            action = robot.data.default_joint_pos.clone()
            
            obs, _, _, _, _ = env.step(action)
            
            amp_obs_dict = obs["amp"]
            amp_obs = torch.cat(list(amp_obs_dict.values()), dim=-1).reshape(args_cli.num_envs, -1)
            
            # key_links_pos_b_his = amp_obs_dict["key_links_pos_b"].reshape(args_cli.num_envs, 2, -1, 3)  # shape: (N, 2, M, 3), 2 is history length, M is number of key links
            # key_links_pos_b = key_links_pos_b_his[:, 1, :, :]   # shape: (N, M, 3), only use the latest key links positions
            
            key_links_pos_b = amp_obs[:, 27+27:27+27+4*3].reshape(args_cli.num_envs, -1, 3)  # shape: (N, M, 3), N is number of envs, M is number of key links
            num_key_links = key_links_pos_b.shape[1]
            root_pos_w = robot.data.root_pos_w.unsqueeze(1) # shape: (N, 1, 3)
            root_quat_w = robot.data.root_quat_w.unsqueeze(1)  # shape: (N, 1, 4)
            key_links_pos_w = root_pos_w + math_utils.quat_apply(root_quat_w.expand(-1, num_key_links, -1), key_links_pos_b)
            
            draw_interface.clear_lines()
            lines_color = [[1.0, 1.0, 0.0, 1.0]] * env.scene.num_envs
            line_thicknesses = [3.0] * env.scene.num_envs
            for i in range(key_links_pos_w.shape[1]):
                draw_interface.draw_lines(
                    root_pos_w[:, 0, :].tolist(), 
                    key_links_pos_w[:, i, :].tolist(), 
                    lines_color, 
                    line_thicknesses
                )
            
            if count % 50 == 0:
                print("-" * 80)
                print("amp observations shape", amp_obs.shape)
                # print("key links pos history shape", key_links_pos_b_his.shape)
                print("key links pos shape", key_links_pos_b.shape)
                print("key links pos in base frame", key_links_pos_b[0, :, :])
            
            count += 1
    
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()