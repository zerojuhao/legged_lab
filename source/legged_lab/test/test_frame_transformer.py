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

from legged_lab.tasks.locomotion.amp.config.g1.amp_flat_env_cfg import G1AmpFlatEnvCfg
from legged_lab.tasks.locomotion.amp.amp_env import AmpEnv

def main():
    env_cfg = G1AmpFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env_cfg.scene.frame_transformer.debug_vis = True
    env_cfg.scene.frame_transformer.visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    env = AmpEnv(cfg=env_cfg)
    
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    frame_transformer: FrameTransformer = env.scene["frame_transformer"]
    ft: FrameTransformer = env.scene.sensors["frame_transformer"]
    
    # 检查frame_transformer与ft的地址
    print("FrameTransformer address:", hex(id(frame_transformer)))
    print("FrameTransformer from scene address:", hex(id(ft)))
    
    print(frame_transformer)
    
    sim_dt = env.sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    while simulation_app.is_running():
        
        if count % 200 == 0:
            sim_time = 0.0
            count = 0
            
            # reset robot
            env.reset()
            print("-"*80)
            print("[INFO] Resetting environment")
            
            print("Target frames:")
            print(frame_transformer.data.target_frame_names)
            
        source_pos = frame_transformer.data.source_pos_w
        target_pos = frame_transformer.data.target_pos_w
        
        draw_interface.clear_lines()
        lines_color = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
        line_thicknesses = [3.0] * source_pos.shape[0]
        
        for i in range(target_pos.shape[1]):
            draw_interface.draw_lines(
                source_pos.tolist(), 
                target_pos[:, i, :].tolist(), 
                lines_color, 
                line_thicknesses
            )

        robot:Articulation = env.scene["robot"]
        joint_pos_target = robot.data.default_joint_pos

        env.step(joint_pos_target)
        count += 1
        sim_time += sim_dt
        
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()  # Close the simulation app gracefully


