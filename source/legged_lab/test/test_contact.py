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
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg, ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv
from legged_lab.assets.unitree import G1_27DOF_CFG
 
def main():
    env_cfg = G1AmpFlatEnvCfg()
    env_cfg.scene.robot = G1_27DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True # type:ignore
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True # type:ignore
    env_cfg.actions.joint_pos.use_default_offset = False
    env_cfg.actions.joint_pos.scale = 1.0
    
    env_cfg.motion_loader.motion_weights = {
        "08_09_poses": 1.0,
    }
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env_cfg.scene.contact_forces.debug_vis = True

    env = AmpEnv(cfg=env_cfg)

    sensor_cfg = SceneEntityCfg("contact_forces", body_names=[
        ".*_wrist_yaw_link",
    ])
    sensor_cfg.resolve(env.scene)
    print(sensor_cfg.body_names)
    print(sensor_cfg.body_ids)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    lab_joint_names = env.scene["robot"].joint_names
    modified_joint = {
        "left_shoulder_roll_joint": -0.12,
        "left_elbow_joint": 1.3,
    }
    modified_joint_ids = [
        lab_joint_names.index(name) for name in modified_joint.keys()
    ]

    count = 0
    obs, _ = env.reset()
    
    while simulation_app.is_running():
        with torch.inference_mode():
                        
            robot:Articulation = env.scene["robot"]
            action = robot.data.default_joint_pos.clone()

            for joint_id, joint_value in zip(modified_joint_ids, modified_joint.values()):
                action[:, joint_id] = joint_value
            
            obs, _, _, _, _ = env.step(action)
            
            net_contact_forces = contact_sensor.data.net_forces_w_history
            # Shape is (N, T, B, 3), where N is the number of sensors (num_envs), 
            # T is the configured history length
            # B is the number of bodies in each sensor.
            
            fs = torch.norm(net_contact_forces[0, :, sensor_cfg.body_ids], dim=-1)
            
            if count % 10 == 0:
                print(net_contact_forces.shape)
                print(fs)
            
            count += 1
    
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()