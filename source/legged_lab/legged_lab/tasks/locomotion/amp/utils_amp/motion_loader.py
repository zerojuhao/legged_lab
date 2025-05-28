import os
import numpy as np
import yaml
from typing import Literal

import torch
import joblib

import isaaclab.utils.math as math_utils


@torch.jit.script
def vel_forward_diff(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the forward differences of the input data

    Args:
        data (torch.Tensor): The input data tensor of shape (N, dim).
        dt (float): The time step duration.
    """
    N = data.shape[0]
    if N < 2:
        raise RuntimeError(f"Input data has only {N} frames, cannot compute velocity.")
    vel = torch.zeros_like(data)
    vel[:-1] = (data[1:] - data[:-1]) / dt
    vel[-1] = vel[-2]  # use the last value as the same as the second last value
    return vel


@torch.jit.script
def ang_vel_from_quat_diff(quat: torch.Tensor, dt: float, in_frame:str = "body") -> torch.Tensor:
    """Compute the angular velocity from quaternion differences.

    Args:
        quat (torch.Tensor): The input quaternion tensor of shape (N, 4), 
                            representing the rotation from world to body frame.
        dt (float): The time step duration.
        in_frame (str): The frame in which the angular velocity is expressed, either "body" or "world".
    """
    if in_frame not in ["body", "world"]:
        raise ValueError(f"Invalid in_frame value: {in_frame}. Must be 'body' or 'world'.")
    
    N = quat.shape[0]
    if N < 2:
        raise RuntimeError(f"Input quaternion has only {N} frames, cannot compute angular velocity.")
    
    ang_vel = torch.zeros((N, 3), dtype=torch.float32, device=quat.device)
    for i in range(N-1):
        q1 = quat[i].unsqueeze(0)  # from world frame to body, shape (1, 4)
        q2 = quat[i + 1].unsqueeze(0)  # from world frame to body (at next time), shape (1, 4)

        diff_quat = math_utils.quat_mul(math_utils.quat_conjugate(q1), q2)
        diff_angle_axis = math_utils.axis_angle_from_quat(diff_quat)
        if in_frame == "world":
            diff_angle_axis = math_utils.quat_rotate(q1, diff_angle_axis)
        ang_vel[i, :] = diff_angle_axis.squeeze() / dt  # convert to angular velocity

    ang_vel[-1, :] = ang_vel[-2, :]  # use the last value as the same as the second last value
    
    return ang_vel


class MotionLoader:
    """Handles loading and preparing retargeted motion data for a robot. 

    The retargeting process is performed using the PHC repository (https://github.com/ZhengyiLuo/PHC).
    
    Args: 
        motion_file (str): Path to the motion data file.
        cfg_file (str): Path to the configuration file for the motion data. This is a YAML file that
            contains the necessary parameters for loading the motion data.
        device (str): Device to load the motion data onto, default is "cuda:0".
    """
    def __init__(self, motion_file, cfg_file, device: str = "cuda:0") -> None:
        self.device = device
        
        assert os.path.exists(motion_file), f"[MotionLoader] Motion file {motion_file} does not exist, please check the file."
        assert os.path.exists(cfg_file), f"[MotionLoader] Config file {cfg_file} does not exist, please check the file."

        self.motion_file = motion_file
        self.cfg_file = cfg_file
        
        # load config file
        with open(cfg_file, "r") as f:
            self.cfg: dict = yaml.safe_load(f)
            
        self._load_config()
        self._build_joint_mapping()
        
        self._load_motion_data()
        
    def _load_config(self):
        """Load the configuration from the YAML file."""
        
        # joint names
        self.retargeted_joint_names: list[str] = self.cfg["retargeted_joint_names"]
        self.lab_joint_names: list[str] = self.cfg["lab_joint_names"]

        # set the joint offsets
        self.joint_offsets_dict: dict[str, float] = self.cfg["joint_offsets"]
        self.joint_offsets: np.ndarray = np.zeros(len(self.lab_joint_names), dtype=np.float32)
        for joint_name, offset in self.joint_offsets_dict.items():
            if joint_name in self.lab_joint_names:
                idx = self.lab_joint_names.index(joint_name)
                self.joint_offsets[idx] = np.deg2rad(offset)
            else:
                raise ValueError(f"Joint name {joint_name} in joint_offsets not found in lab joint names: {self.lab_joint_names}")
        
        
        
    def _build_joint_mapping(self):
        """Get the joint index mapping from lab joint names to retargeted joint names, and vice versa."""

        # first check if there are any joint name not in both lists
        for joint_name in self.lab_joint_names:
            if joint_name not in self.retargeted_joint_names:
                raise ValueError(f"Joint name {joint_name} not found in retargeted joint names.")
        for joint_name in self.retargeted_joint_names:
            if joint_name not in self.lab_joint_names:
                raise ValueError(f"Joint name {joint_name} not found in lab joint names.")
        
        # create the mapping from retargeted joint names to lab joint names
        self.retargeted_to_lab_mapping = [self.retargeted_joint_names.index(joint_name) for joint_name in self.lab_joint_names]
        # create the mapping from lab joint names to retargeted joint names
        self.lab_to_retargeted_mapping = [self.lab_joint_names.index(joint_name) for joint_name in self.retargeted_joint_names]
    
    
    def _load_motion_data(self):
        self.motion_data_dict = joblib.load(self.motion_file)
        
        self.motion_weights_dict = self.cfg.get("motion_weights", None)
        assert self.motion_weights_dict is not None, f"[MotionLoader] Motion weights not found in config file {self.cfg_file}."
        for key in self.motion_weights_dict:
            if key not in self.motion_data_dict:
                raise ValueError(f"[MotionLoader] Motion {key} not found in motion data file {self.motion_file}.")
        
        # only load those motions that are in the motion_weights
        self.motion_names = list(self.motion_weights_dict.keys())
        print(f"[MotionLoader] Load motions: {self.motion_names}")
        
        # get motion weights and normalize them
        self.motion_weights = np.array(list(self.motion_weights_dict.values()), dtype=np.float32)
        self.motion_weights /= np.sum(self.motion_weights)
        
        self.motion_data = []
        self.motion_duration = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        
        for motion_name in self.motion_names:
            print(f"[MotionLoader] Loading motion: {motion_name}.")
            
            motion_raw_data = self.motion_data_dict[motion_name]
            """
            root_trans_offset
            pose_aa
            dof
            root_rot
            smpl_joints
            fps
            """
            motion_fps = motion_raw_data["fps"]
            dt = 1.0 / motion_fps
            num_frames = len(motion_raw_data["dof"])
            duration = dt * (num_frames - 1)
            
            self.motion_duration.append(duration)
            self.motion_fps.append(motion_fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)

            # convert to torch tensors, and rename dict keys
            motion_data = {
                "root_pos": 1
            }
            
    def _process_motion_data(self, motion_raw_data):
        """Process the raw motion data into a format suitable for training.
        
        Args:
            motion_raw_data (dict): The raw motion data dictionary containing keys:
                - root_trans_offset
                - pose_aa
                - root_rot
                - dof
                - smpl_joints
                - fps
                root_trans_offset, root_rot, dof, etc. has shape (num_frames, dim).
        """
        fps = motion_raw_data["fps"]
        dt = 1.0 / fps
        num_frames = len(motion_raw_data["dof"])
        if num_frames < 2:
            raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")
        
        # root position in world frame, shape (num_frames, 3)
        root_pos_w = torch.tensor(motion_raw_data["root_trans_offset"], dtype=torch.float32, device=self.device)  
        
        # root rotation (quaternion) from world frame to body frame, shape (num_frames, 4)
        root_quat = torch.tensor(motion_raw_data["root_rot"], dtype=torch.float32, device=self.device)
        root_quat = math_utils.convert_quat(root_quat, "wxyz") # convert to w, x, y, z format
        root_quat = math_utils.quat_unique(root_quat)  # ensure unique quaternions
        root_quat = math_utils.normalize(root_quat)  # ensure quaternion is normalized

        # root velocity in world frame, shape (num_frames, 3)
        root_vel_w = vel_forward_diff(root_pos_w, dt)
        
        # root velocity in body frame
        root_vel_b = math_utils.quat_rotate_inverse(root_quat, root_vel_w)
        
        # root angular velocity in body frame, shape (num_frames, 3)
        root_ang_vel_b = ang_vel_from_quat_diff(root_quat, dt, in_frame="body")

        # joint position, shape (num_frames, num_joints)
        dof_pos = motion_raw_data["dof"][:, self.retargeted_to_lab_mapping]
        # apply joint offsets
        dof_pos += self.joint_offsets
        dof_pos = torch.tensor(dof_pos, dtype=torch.float32, device=self.device)
        
        # joint velocity, shape (num_frames, num_joints)
        dof_vel = vel_forward_diff(dof_pos, dt)
        
        # key end effectors position
        # TODO
        
        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_b": root_vel_b,
            "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel
        }
        




