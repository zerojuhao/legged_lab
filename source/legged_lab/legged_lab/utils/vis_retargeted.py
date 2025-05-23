import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visulization of retargeted data.")
parser.add_argument(
    "--robot", 
    type=str,
    default="g1",
    help="The robot name to be used.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np
import torch
import joblib
import yaml

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
if args_cli.robot == "g1":
    from isaaclab_assets import G1_MINIMAL_CFG as ROBOT_CFG  # isort: skip
elif args_cli.robot == "h1":
    from isaaclab_assets import H1_MINIMAL_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin)
    robot = Articulation(ROBOT_CFG.replace(prim_path="/World/Origin1/Robot"))

    return robot, origin


class RetargetedMotionLoader:
    def __init__(self, robot_name: str, motion_file: str, lab_joint_names: list[str]) -> None:
        
        self.robot_name = robot_name
        
        """Initialize the motion loader."""
        motion_dir = os.path.join(LEGGED_LAB_ROOT_DIR, "data", robot_name)
        self.motion_file = os.path.join(motion_dir, motion_file)
        if not os.path.exists(self.motion_file):
            raise FileNotFoundError(f"Motion file {self.motion_file} does not exist, please check the file.")
        # yaml config file contains some retargeted data config
        self.config_file = os.path.join(motion_dir, "retargeted.yaml")
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} does not exist, please check the file.")
        
        # load the motion data
        self.motion_data = joblib.load(self.motion_file)
        self.motion_names = list(self.motion_data.keys())
        print(f"[INFO]: Motion names: {self.motion_names}")
        self.current_motion_idx = 0
        self.current_motion_name = self.motion_names[self.current_motion_idx]
        
        
        # load the config file
        self.retargeted_cfg = yaml.safe_load(open(self.config_file, "r"))
        self.retargeted_joint_names = self.retargeted_cfg["joint_names"]
        self._get_joint_mapping(lab_joint_names, self.retargeted_joint_names)
        self.lab_joint_names = lab_joint_names
        print(self.lab_joint_names.index("left_one_joint"), self.lab_joint_names.index("left_two_joint"))
        print(self.lab_joint_names.index("right_one_joint"), self.lab_joint_names.index("right_two_joint"))

    def _get_joint_mapping(self, lab_joint_names: list[str], retargeted_joint_names: list[str]) -> dict:
        """Get the joint index mapping from lab joint names to retargeted joint names, and vice versa."""
        # first check if there are any joint name not in both lists
        for joint_name in lab_joint_names:
            if joint_name not in retargeted_joint_names:
                raise ValueError(f"Joint name {joint_name} not found in retargeted joint names.")
        for joint_name in retargeted_joint_names:
            if joint_name not in lab_joint_names:
                raise ValueError(f"Joint name {joint_name} not found in lab joint names.")
        
        self.retargeted_to_lab_mapping = [retargeted_joint_names.index(joint_name) for joint_name in lab_joint_names]
        self.lab_to_retargeted_mapping = [lab_joint_names.index(joint_name) for joint_name in retargeted_joint_names]

    def get_motion_data(self, current_frame: int, device):
        """Get the motion data for the current frame.

        Args:
            current_frame (int): The frame index of the motion data.
        """
        m_data = self.motion_data[self.current_motion_name]
        root_pos = m_data['root_trans_offset'][current_frame]
        root_quat = m_data['root_rot'][current_frame][[3, 0, 1, 2]] # w, x, y, z
        dof_pos = m_data['dof'][current_frame][self.retargeted_to_lab_mapping]
        
        if self.robot_name == "g1":
            # add offset to the left_one_joint and right_one_joint
            dof_pos[self.lab_joint_names.index("left_one_joint")] += np.deg2rad(68)
            dof_pos[self.lab_joint_names.index("left_two_joint")] += np.deg2rad(45)
            dof_pos[self.lab_joint_names.index("right_one_joint")] -= np.deg2rad(68)
            dof_pos[self.lab_joint_names.index("right_two_joint")] -= np.deg2rad(45)
        
        return torch.tensor(root_pos, device=device), \
                torch.tensor(root_quat, device=device), \
                torch.tensor(dof_pos, device=device)
    
    def get_motion_length(self):
        """Get the length of the motion data."""
        return len(self.motion_data[self.current_motion_name]['dof'])
    
    def next_motion(self):
        """Get the next motion data."""
        self.current_motion_idx += 1
        if self.current_motion_idx >= len(self.motion_names):
            self.current_motion_idx = 0
        self.current_motion_name = self.motion_names[self.current_motion_idx]
        print(f"[INFO]: Current motion name: {self.current_motion_name}")
        

def run_simulator(sim: sim_utils.SimulationContext, 
                  robot: Articulation, 
                  origin: torch.Tensor) -> None:

    re_motion_loader = RetargetedMotionLoader(args_cli.robot, "0-ACCAD_Male1Walking_c3d_Walk B10 - Walk turn left 45_poses.pkl", robot.data.joint_names)
    
    """Run the simulation loop"""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Simulate physics
    while simulation_app.is_running():

        current_time = int(sim_time / sim_dt) % re_motion_loader.get_motion_length()
        # get the motion data
        root_pos, root_quat, dof_pos = re_motion_loader.get_motion_data(current_time, sim.device)
        robot_state = robot.data.default_root_state.clone()
        robot_state[:, :3] = origin + root_pos
        robot_state[:, 3:7] = root_quat
        robot.write_root_pose_to_sim(robot_state[:, :7])
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, :] = dof_pos
        robot.write_joint_position_to_sim(joint_pos)
        
        # only render, no physics
        sim.render()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)

def main():
    """Main function."""
    
    MOTION_DT = 1.0 / 30.0  # 30 fps

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=MOTION_DT))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    robot, origin = design_scene()
    origin = torch.tensor(origin, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, robot, origin)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()