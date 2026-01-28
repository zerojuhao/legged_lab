import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visulization of retargeted data.")
parser.add_argument(
    "--robot", 
    type=str,
    default="a1",
    help="The robot name to be used.",
    choices=["g1", "a1", "atom01", "atom01_long_base_link", "atom02"],
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # set headless to True for this script

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np

import isaacsim.core.utils.prims as prim_utils # type: ignore 

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
# load robot cfg as single_retarget does
if args_cli.robot == "g1":
    from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
elif args_cli.robot == "atom01":
    from legged_lab.assets.roboparty import ATOM01_CFG as ROBOT_CFG
elif args_cli.robot == "atom01_long_base_link":
    from legged_lab.assets.roboparty import ATOM01_LONG_BASE_LINK_CFG as ROBOT_CFG
elif args_cli.robot == "atom02":
    from legged_lab.assets.roboparty import ATOM02_CFG as ROBOT_CFG
elif args_cli.robot == "a1":
    from legged_lab.assets.unitree import UNITREE_A1_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

if __name__ == "__main__":
    # # Ground-plane
    # cfg = sim_utils.GroundPlaneCfg()
    # cfg.func("/World/defaultGroundPlane", cfg)
    
    # # Lights
    # cfg = sim_utils.LightCfg()
    # cfg.func("/World/defaultLight", cfg)
    
    # origin = np.array([0.0, 0.0, 0.0])
    # prim_utils.create_prim(
    #     prim_path="/World/Origin",
    #     prim_type="Xform",
    #     position=origin,
    # )
    
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    
    # Create the scene
    origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin)
    robot = Articulation(ROBOT_CFG.replace(prim_path="/World/Origin1/Robot"))
    
    sim.reset()
    
    lab_joint_names = robot.joint_names
    print("Legged Lab joint names:")
    # print(lab_joint_names)
    for jnt in lab_joint_names:
        print("-", jnt)

    body_names = robot.body_names
    body_masses = robot.data.default_mass[0]
    for name, mass in zip(body_names, body_masses):
        print(f"Body: {name}, Mass: {mass} kg")