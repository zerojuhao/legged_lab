import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

import isaaclab.utils.math as math_utils

from legged_lab.tasks.locomotion.amp.utils_amp.motion_loader import vel_forward_diff, ang_vel_from_quat_diff

import matplotlib.pyplot as plt

def design_scene():
    
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    
    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # Rigid Object
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Objects/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), 
                                                  ang_vel=(5, 6, 3),),
    )
    cone_object = RigidObject(cfg=cone_cfg)
    
    return cone_object
    

if __name__ == "__main__":
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    cone_object = design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    sim_dt = sim.get_physics_dt()
    
    sim_time = 0.0
    count = 0
    
    root_state = cone_object.data.default_root_state.clone()
    
    print(root_state.shape)
    
    cone_object.write_root_pose_to_sim(root_state[:, :7])
    cone_object.write_root_velocity_to_sim(root_state[:, 7:])
    
    root_quat = []
    root_ang_vel_w = []
    root_ang_vel_b = []
    
    # Simulate physics
    while simulation_app.is_running():
        
        if sim_time > 1.0:
            print(f"[INFO]: Simulation completed in {count} steps, total time: {sim_time:.2f} seconds.")
            break

        root_quat.append(cone_object.data.root_quat_w.clone())
        root_ang_vel_w.append(cone_object.data.root_ang_vel_w.clone())
        root_ang_vel_b.append(cone_object.data.root_ang_vel_b.clone())

        cone_object.write_data_to_sim()
        sim.step()
        
        sim_time += sim_dt
        count += 1
        
        # update buffer
        cone_object.update(sim_dt)
    
    root_quat = torch.cat(root_quat, dim=0)
    root_ang_vel_w = torch.cat(root_ang_vel_w, dim=0)
    root_ang_vel_b = torch.cat(root_ang_vel_b, dim=0)

    root_ang_vel_w_diff = ang_vel_from_quat_diff(root_quat, sim_dt, in_frame="world")
    root_ang_vel_b_diff = ang_vel_from_quat_diff(root_quat, sim_dt, in_frame="body")

    # plotting the results
    plt.figure(figsize=(30, 15))
    plt.subplot(2, 2, 1)
    plt.plot(root_ang_vel_w[:, 0].cpu().numpy(), label='Angular Velocity X')
    plt.plot(root_ang_vel_w[:, 1].cpu().numpy(), label='Angular Velocity Y')
    plt.plot(root_ang_vel_w[:, 2].cpu().numpy(), label='Angular Velocity Z')
    plt.title('Angular Velocity in World Frame')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(root_ang_vel_w_diff[:, 0].cpu().numpy(), label='Angular Velocity X (Diff)')
    plt.plot(root_ang_vel_w_diff[:, 1].cpu().numpy(), label='Angular Velocity Y (Diff)')
    plt.plot(root_ang_vel_w_diff[:, 2].cpu().numpy(), label='Angular Velocity Z (Diff)')
    plt.title('Angular Velocity Difference from Quaternion in World Frame')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(root_ang_vel_b[:, 0].cpu().numpy(), label='Angular Velocity X')
    plt.plot(root_ang_vel_b[:, 1].cpu().numpy(), label='Angular Velocity Y')
    plt.plot(root_ang_vel_b[:, 2].cpu().numpy(), label='Angular Velocity Z')
    plt.title('Angular Velocity in Body Frame')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(root_ang_vel_b_diff[:, 0].cpu().numpy(), label='Angular Velocity X (Diff)')
    plt.plot(root_ang_vel_b_diff[:, 1].cpu().numpy(), label='Angular Velocity Y (Diff)')
    plt.plot(root_ang_vel_b_diff[:, 2].cpu().numpy(), label='Angular Velocity Z (Diff)')
    plt.title('Angular Velocity Difference from Quaternion in Body Frame')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Close the simulation app
    simulation_app.close()
    