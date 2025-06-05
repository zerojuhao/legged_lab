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

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv

 
def main():
    env_cfg = G1AmpFlatEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = AmpEnv(cfg=env_cfg)
    
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    
    count = 0
    obs, _ = env.reset()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            
            # if count % 50 == 1:
            #     amp_obs_dict = obs["amp"]
            #     amp_obs = torch.cat(list(amp_obs_dict.values()), dim=-1).reshape(args_cli.num_envs, -1)
            #     print("amp observations shape", amp_obs.shape)
                
            #     feet_pos_in_dict = amp_obs_dict["feet_pos"][0, 1, :].reshape(-1, 3)
            #     feet_pos_in_tensor = amp_obs[0, 43+12:43+24].reshape(-1, 3)
            #     print("feet pos in dict\n", feet_pos_in_dict)
            #     print("feet pos in tensor\n", feet_pos_in_tensor)
            #     print("feet pos in dict == feet pos in tensor: ", torch.allclose(feet_pos_in_dict, feet_pos_in_tensor))

            # action = torch.randn_like(env.action_manager.action)
            # obs, rew, terminated, truncated, info = env.step(action)
            
            robot:Articulation = env.scene["robot"]
            action = robot.data.default_joint_pos.clone()
            
            obs, _, _, _, _ = env.step(action)
            
            amp_obs_dict = obs["amp"]
            amp_obs = torch.cat(list(amp_obs_dict.values()), dim=-1).reshape(args_cli.num_envs, -1)
            
            key_links_pos_b_his = amp_obs_dict["key_links_pos_b"].reshape(args_cli.num_envs, 2, -1, 3)  # shape: (N, 2, M, 3), 2 is history length, M is number of key links
            key_links_pos_b = key_links_pos_b_his[:, 1, :, :]   # shape: (N, M, 3), only use the latest key links positions
            root_pos_w = robot.data.root_pos_w.unsqueeze(1) # shape: (N, 1, 3)
            root_quat_w = robot.data.root_quat_w.unsqueeze(1)  # shape: (N, 1, 4)
            key_links_pos_w = root_pos_w + math_utils.quat_rotate(root_quat_w, key_links_pos_b)
            
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
                print("key links pos history shape", key_links_pos_b_his.shape)
                print("key links pos shape", key_links_pos_b.shape)
            
            count += 1
    
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()