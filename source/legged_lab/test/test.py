import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.headless = True  # set headless to True for this script

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

from legged_lab.tasks.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

def main():
    env_cfg = G1RoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    entity: Articulation = env.scene["robot"]
    print(entity.joint_names)
    # joint_ids, joint_names = entity.find_joints(
    #     [".*_shoulder_.*_joint"], preserve_order=False
    # )
    joint_ids, joint_names = entity.find_joints(
        ['right_shoulder_roll_joint', 'left_shoulder_roll_joint'], 
        preserve_order=False
    )
    print("Joint IDs:", joint_ids)
    print("Joint Names:", joint_names)


if __name__ == "__main__":
    main()

    simulation_app.close()  # Close the simulation app when done
