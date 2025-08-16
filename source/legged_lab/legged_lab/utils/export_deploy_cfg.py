import argparse
import yaml

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument(
    "--robot", 
    type=str,
    default="g1_flat",
    help="The robot name to be used.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # set headless to True for this script

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import re
from collections import OrderedDict
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if args_cli.robot == "g1_amp":
    from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg as RobotEnvCfg
    from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv as RobotEnv
elif args_cli.robot == "g1_flat":
    from legged_lab.tasks.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg as RobotEnvCfg
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv as RobotEnv
elif args_cli.robot == "go2_flat":
    from legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg as RobotEnvCfg
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv as RobotEnv
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

def represent_ordereddict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)

def main():
    env_cfg = RobotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = RobotEnv(cfg=env_cfg)
    robot: Articulation = env.scene["robot"]
    
    output_cfg = OrderedDict()
    
    obs_term_names = env.observation_manager.active_terms["policy"]
    obs_term_cfg = env.observation_manager._group_obs_term_cfgs["policy"] # type: ignore
    output_cfg["obs_names"] = obs_term_names
    output_cfg["obs_cfg"] = OrderedDict()
    for term_name, term_cfg in zip(obs_term_names, obs_term_cfg):
        output_cfg["obs_cfg"][term_name] = OrderedDict([
            ("clip", term_cfg.clip if term_cfg.clip else [0.0]),
            ("scale", float(term_cfg.scale) if term_cfg.scale else 1.0),
            ("history_length", term_cfg.history_length)
        ])
    
    action_cfg = env_cfg.actions.joint_pos
    action_term:JointPositionAction = env.action_manager.get_term("joint_pos")
    
    # check joint names
    joint_names = robot.joint_names
    for i, jnt_name in enumerate(joint_names):
        if jnt_name != action_term._joint_names[i]: # type: ignore
            raise ValueError(f"Joint name mismatch: {jnt_name} != {action_term._joint_names[i]}") # type: ignore
    
    output_cfg["joint_names"] = joint_names
    
    output_cfg["action_cfg"] = OrderedDict()
    for i, jnt_name in enumerate(joint_names):
        if action_cfg.clip is not None:
            clip = action_term._clip[0, i, :].cpu().tolist()  # type: ignore
            print(f"Joint {jnt_name} clip: {clip}")
        else:
            clip = [0.0] # https://github.com/ros2/rclcpp/issues/1955
        
        if isinstance(action_cfg.scale, (float, int)):
            scale = float(action_cfg.scale)
        elif isinstance(action_cfg.scale, dict):
            scale = action_term._scale[0, i].item() # type: ignore
        else:
            scale = 1.0
        
        found = False
        kp = 0.0
        kd = 0.0
        for key, value in env_cfg.scene.robot.actuators.items():
            for expr in value.joint_names_expr:
                if re.fullmatch(expr, jnt_name):
                    found = True

                    if isinstance(value.stiffness, float):
                        kp = value.stiffness
                    elif isinstance(value.stiffness, dict):
                        for k, v in value.stiffness.items():
                            if re.fullmatch(k, jnt_name):
                                kp = v
                                break
                    else:
                        raise ValueError(f"Unsupported stiffness type for joint {jnt_name}: {type(value.stiffness)}")
                    
                    if isinstance(value.damping, float):
                        kd = value.damping
                    elif isinstance(value.damping, dict):
                        for k, v in value.damping.items():
                            if re.fullmatch(k, jnt_name):
                                kd = v
                                break
                    else:
                        raise ValueError(f"Unsupported damping type for joint {jnt_name}: {type(value.damping)}")
                    
                    # found, so break the loop
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Joint {jnt_name} not found in Robot Cfg (ArticulationCfg)'s actuators.")
        
        default_pos = robot.data.default_joint_pos[0, i].item()
        
        output_cfg["action_cfg"][jnt_name] = OrderedDict([
            ("clip", clip),
            ("scale", scale),
            ("kp", kp),
            ("kd", kd),
            ("default_pos", default_pos),
        ])
    
    # export the configuration to a YAML file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f"deploy_config_{args_cli.robot}.yaml")
    with open(output_file, 'w') as f:
        yaml.dump(output_cfg, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Configuration exported to {output_file}")
    
    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()



