import argparse

from idna import encode
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv
from legged_lab.tasks.locomotion.amp.config.go2.amp_flat_env_cfg import Go2AmpFlatEnvCfg

 
def main():
    env_cfg = Go2AmpFlatEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 1.0
    
    env = AmpEnv(cfg=env_cfg)
    
    count = 0
    obs, _ = env.reset()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            
            if count % 50 == 1:
                amp_obs_dict = obs["amp"]
                amp_obs = torch.cat(list(amp_obs_dict.values()), dim=-1).reshape(args_cli.num_envs, -1)
                print("amp observations shape", amp_obs.shape)
                
                feet_pos_in_dict = amp_obs_dict["feet_pos"][0, 1, :].reshape(-1, 3)
                feet_pos_in_tensor = amp_obs[0, 43+12:43+24].reshape(-1, 3)
                print("feet pos in dict\n", feet_pos_in_dict)
                print("feet pos in tensor\n", feet_pos_in_tensor)
                print("feet pos in dict == feet pos in tensor: ", torch.allclose(feet_pos_in_dict, feet_pos_in_tensor))

            action = torch.randn_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(action)
            
            count += 1
    
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()