import os
import numpy as np
import yaml
from typing import Literal

import torch
import joblib

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import FrameTransformer

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

    The retargeting process is performed using [motion_retarget](https://github.com/zitongbai/motion_retarget)
    
    The example of usage can be found in `source/legged_lab/test/test_motion_loader.py`
    
    Args: 
        motion_file (str): Path to the motion data file.
        cfg_file (str): Path to the configuration file for the motion data. This is a YAML file that
            contains the necessary parameters for loading the motion data.
        device (str): Device to load the motion data onto, default is "cuda:0".
    """
    def __init__(self, motion_file, cfg_file, env: ManagerBasedEnv, device: str = "cuda:0") -> None:
        self.device = device
        self.env = env

        assert os.path.exists(motion_file), f"[MotionLoader] Motion file {motion_file} does not exist, please check the file."
        assert os.path.exists(cfg_file), f"[MotionLoader] Config file {cfg_file} does not exist, please check the file."

        self.motion_file = motion_file
        self.cfg_file = cfg_file
        
        # load config file
        with open(cfg_file, "r") as f:
            self.cfg: dict = yaml.safe_load(f)
        
        self._load_motion_data()
        self.motion_ids = torch.arange(len(self.motion_names), dtype=torch.long, device=self.device)

    def _build_joint_mapping(self):
        """Get the joint index mapping from lab joint names to retargeted joint names, and vice versa."""
        robot:Articulation = self.env.scene["robot"]
        self.lab_joint_names = robot.joint_names
        if "joint_mapping" in self.cfg:
            # if joint names in retarget motion data are different from the lab joint names,
            # we need to load the joint mapping from the config file
            raise NotImplementedError("Joint mapping from config file is not implemented yet.")
        else:
            # else, we assume the joint names in the retargeted motion data are the same as the lab joint names, 
            # and we only need to rearrange them to match the order in the lab joint names
            try:
                # in lab joint order, with value as the index in retargeted joint names
                self.retargeted_to_lab_mapping, _ = string_utils.resolve_matching_names(
                    keys=self.lab_joint_names,
                    list_of_strings= self.retargeted_joint_names,
                    preserve_order=True
                )
            except ValueError as e:
                print(f"[MotionLoader] Error in resolving joint names: {e}")
                raise ValueError(f"[MotionLoader] Joint names in retargeted motion data {self.retargeted_joint_names} do not match the lab joint names {self.lab_joint_names}.")

    def _build_key_links_mapping(self):
        frame_transformer: FrameTransformer = self.env.scene.sensors["frame_transformer"]
        self.lab_key_links_names = frame_transformer.data.target_frame_names
        
        if "key_links_mapping" in self.cfg:
            # if key links names in retarget motion data are different from the lab key links names,
            # we need to load the key links mapping from the config file
            raise NotImplementedError("Key links mapping from config file is not implemented yet.")
        else:
            # else, we assume the key links names in the retargeted motion data are the same as the lab key links names, 
            # and we only need to rearrange them to match the order in the lab key links names
            try:
                self.retargeted_key_links_mapping, _ = string_utils.resolve_matching_names(
                    keys=self.lab_key_links_names, 
                    list_of_strings=self.retargeted_link_names,
                    preserve_order=True
                )
            except ValueError as e:
                print(f"[MotionLoader] Error in resolving key links names: {e}")
                raise ValueError(f"[MotionLoader] Key links names in retargeted motion data {self.retargeted_link_names} do not match the lab key links names {self.lab_key_links_names}.")

    def _load_joint_offsets(self):
        """Load the joint offsets from the configuration file."""
        self.joint_offsets = np.zeros(len(self.lab_joint_names), dtype=np.float32)
        self.joint_offsets_dict = self.cfg.get("joint_offsets", None)
        if self.joint_offsets_dict is None:
            return
        assert isinstance(self.joint_offsets_dict, dict), f"[MotionLoader] Joint offsets should be a dictionary, but got {type(self.joint_offsets_dict)}."
        for joint_name, offset in self.joint_offsets_dict.items():
            if joint_name in self.lab_joint_names:
                idx = self.lab_joint_names.index(joint_name)
                self.joint_offsets[idx] = np.deg2rad(offset)
            else:
                raise ValueError(f"[MotionLoader] Joint name {joint_name} in joint_offsets not found in lab joint names: {self.lab_joint_names}")
        
    def _load_motion_data(self):
        """Load the motion data from the motion file and process it."""
        # load pkl file, it is a dictionary with keys:
        # - `retarget_data`: a dictionary with keys as motion names and values as the raw motion data.
        # - `dof_names`: a list of joint names in the retargeted motion data.
        # - `joint_names_robot`: a list of joint names in the robot model. Note that the joint here refers to the cartesian joint, not the dof joint.
        # - `joint_names_smpl`: a list of joint names in the SMPL model. Note that the joint here refers to the cartesian joint, not the dof joint.
        motion_dict = joblib.load(self.motion_file)
        self.motion_data_dict = motion_dict["retarget_data"]
        self.retargeted_joint_names = motion_dict["dof_names"]
        self.retargeted_link_names = motion_dict["joint_names_robot"]
        
        self._build_joint_mapping()
        self._build_key_links_mapping()
        self._load_joint_offsets()

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
        
        self.motion_data = []   # list to store processed motion data (a dictionary), each element corresponds to a motion
        self.motion_duration = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        
        for motion_name in self.motion_names:
            print(f"[MotionLoader] Loading motion: {motion_name}.")
            
            motion_raw_data = self.motion_data_dict[motion_name]
            
            motion_processed_data = self._process_motion_data(motion_raw_data)
            self.motion_data.append(motion_processed_data)

            motion_fps = motion_raw_data["fps"]
            dt = 1.0 / motion_fps
            num_frames = len(motion_raw_data["dof_pos"])
            duration = dt * (num_frames - 1)
            
            self.motion_duration.append(duration)
            self.motion_fps.append(motion_fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
        
        self.motion_fps = np.array(self.motion_fps, dtype=np.float32)
        self.motion_dt = np.array(self.motion_dt, dtype=np.float32)
        self.motion_duration = np.array(self.motion_duration, dtype=np.float32)
        self.motion_num_frames = np.array(self.motion_num_frames, dtype=np.int32)
        
        print(f"[MotionLoader] Loaded {len(self.motion_names)} motions with total duration: {self.get_total_duration()} seconds.")
            
    def _process_motion_data(self, motion_raw_data):
        """Process the raw motion data into a format suitable for training.
        
        Args:
            motion_raw_data (dict): The raw motion data dictionary containing keys:
            - `root_pos`: the root position of the robot in the world frame, shape (N, 3).
            - `root_rot`: the root quaternion (x, y, z, w) of the robot in the world frame, shape (N, 4).
            - `pose_aa`: the angle-axis of each joint in the `Humanoid_Batch` model for the robot.
            - `dof_pos`: the optimized joint angles of the robot that is best fit to the motion, shape (N, num_dof). It is in the order of the mujoco model. 
            - `joint_pos_smpl`: the joint positions of the SMPL model, shape (N, num_joints, 3).
        """
        fps = motion_raw_data["fps"]
        dt = 1.0 / fps
        num_frames = len(motion_raw_data["dof_pos"])
        if num_frames < 2:
            raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")
        
        # root position in world frame, shape (num_frames, 3)
        root_pos_w = torch.tensor(motion_raw_data["root_pos"], dtype=torch.float32, device=self.device, requires_grad=False)

        # root rotation (quaternion) from world frame to body frame, shape (num_frames, 4)
        root_quat = torch.tensor(motion_raw_data["root_rot"], dtype=torch.float32, device=self.device, requires_grad=False)
        root_quat = math_utils.convert_quat(root_quat, "wxyz") # convert to w, x, y, z format
        root_quat = math_utils.quat_unique(root_quat)  # ensure unique quaternions
        root_quat = math_utils.normalize(root_quat)  # ensure quaternion is normalized

        # root velocity in world frame, shape (num_frames, 3)
        root_vel_w = vel_forward_diff(root_pos_w, dt)
        
        # root velocity in body frame
        root_vel_b = math_utils.quat_rotate_inverse(root_quat, root_vel_w)
        
        # root angular velocity in world frame, shape (num_frames, 3)
        root_ang_vel_w = ang_vel_from_quat_diff(root_quat, dt, in_frame="world")
        
        # root angular velocity in body frame, shape (num_frames, 3)
        root_ang_vel_b = ang_vel_from_quat_diff(root_quat, dt, in_frame="body")

        # joint position, shape (num_frames, num_joints)
        dof_pos = motion_raw_data["dof_pos"][:, self.retargeted_to_lab_mapping]
        # apply joint offsets
        dof_pos += self.joint_offsets
        dof_pos = torch.tensor(dof_pos, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # joint velocity, shape (num_frames, num_joints)
        dof_vel = vel_forward_diff(dof_pos, dt)
        
        # key links position TODO: use `joint_pos_robot` or `joint_pos_smpl`?
        key_links_pos_w = motion_raw_data["joint_pos_robot"][:, self.retargeted_key_links_mapping, :]
        key_links_pos_w = torch.tensor(key_links_pos_w, dtype=torch.float32, device=self.device, requires_grad=False)
        
        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            # "root_vel_b": root_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            # "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_links_pos_w": key_links_pos_w,
        }
        
    def get_total_duration(self) -> float:
        """Get the total duration of all motions."""
        return np.sum(self.motion_duration)

    def get_num_motions(self) -> int:
        """Get the number of motions."""
        return len(self.motion_names)

    def get_motion(self, motion_id: int) -> dict:
        """Get the motion data for a specific motion ID.

        Args:
            motion_id (int): The ID of the motion to retrieve.
        
        Returns:
            dict: The motion data dictionary containing the processed motion data.
        """
        if motion_id < 0 or motion_id >= len(self.motion_data):
            raise IndexError(f"[MotionLoader] Motion ID {motion_id} out of range, must be between 0 and {len(self.motion_data) - 1}.")
        return self.motion_data[motion_id]

    def get_motion_duration(self, motion_ids: np.ndarray) -> np.ndarray:
        """Get the duration of a specific motion.

        Args:
            motion_id (np.ndarray): An array of motion IDs for which to get the duration.

        Returns:
            float: The duration of the motion in seconds.
        """
        return self.motion_duration[motion_ids]

    def sample_motions(self, n: int) -> np.ndarray:
        """Sample a batch of motion IDs.

        Args:
            n (int): The number of motion IDs to sample.

        Returns:
            np.ndarray: An array of sampled motion IDs.
        """
        m = self.get_num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True, p=self.motion_weights)
        return motion_ids
    
    def sample_time(self, motion_ids, truncate_time=None):

        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)
        
        motion_durations = self.motion_duration[motion_ids]
        if truncate_time is not None:
            assert truncate_time > 0, f"[MotionLoader] Truncate time must be positive, but got {truncate_time}."
            motion_durations = np.maximum(motion_durations - truncate_time, np.zeros_like(motion_durations))

        # Sample time for each motion
        sample_times = phase * motion_durations
        return sample_times
    
    def get_motion_state(self, motion_ids, motion_times):
        
        n = len(motion_ids)
        
        root_pos_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
        root_pos_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
        root_quat_0 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
        root_quat_1 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
        root_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
        root_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
        root_ang_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
        root_ang_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
        dof_pos_0 = torch.empty([n, len(self.lab_joint_names)], dtype=torch.float32, device=self.device)
        dof_pos_1 = torch.empty([n, len(self.lab_joint_names)], dtype=torch.float32, device=self.device)
        dof_vel_0 = torch.empty([n, len(self.lab_joint_names)], dtype=torch.float32, device=self.device)
        dof_vel_1 = torch.empty([n, len(self.lab_joint_names)], dtype=torch.float32, device=self.device)
        key_links_pos_w_0 = torch.empty([n, len(self.lab_key_links_names), 3], dtype=torch.float32, device=self.device)
        key_links_pos_w_1 = torch.empty([n, len(self.lab_key_links_names), 3], dtype=torch.float32, device=self.device)

        motion_durations = self.motion_duration[motion_ids]
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]
        
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_durations, num_frames, dt)
        
        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            ids = torch.from_numpy(ids[0]).to(self.device).long()
            motion = self.get_motion(uid)
            
            root_pos_w_0[ids, :] = motion["root_pos_w"][frame_idx0[ids], :]
            root_pos_w_1[ids, :] = motion["root_pos_w"][frame_idx1[ids], :]
            
            root_quat_0[ids, :] = motion["root_quat"][frame_idx0[ids], :]
            root_quat_1[ids, :] = motion["root_quat"][frame_idx1[ids], :]
            
            root_vel_w_0[ids, :] = motion["root_vel_w"][frame_idx0[ids], :]
            root_vel_w_1[ids, :] = motion["root_vel_w"][frame_idx1[ids], :]
            
            root_ang_vel_w_0[ids, :] = motion["root_ang_vel_w"][frame_idx0[ids], :]
            root_ang_vel_w_1[ids, :] = motion["root_ang_vel_w"][frame_idx1[ids], :]
            
            dof_pos_0[ids, :] = motion["dof_pos"][frame_idx0[ids], :]
            dof_pos_1[ids, :] = motion["dof_pos"][frame_idx1[ids], :]
            
            dof_vel_0[ids, :] = motion["dof_vel"][frame_idx0[ids], :]
            dof_vel_1[ids, :] = motion["dof_vel"][frame_idx1[ids], :]
            
            key_links_pos_w_0[ids, :, :] = motion["key_links_pos_w"][frame_idx0[ids], :, :]
            key_links_pos_w_1[ids, :, :] = motion["key_links_pos_w"][frame_idx1[ids], :, :]
            
        # interpolate the values
        blend = blend.unsqueeze(-1)  # make it (n, 1) for broadcasting
        
        root_pos_w = root_pos_w_0 * (1.0 - blend)+ root_pos_w_1 * blend
        root_quat = torch.zeros_like(root_quat_0)
        for i in range(n):
            # `quat_slerp` does not support batch operation
            root_quat[i, :] = math_utils.quat_slerp(root_quat_0[i, :], root_quat_1[i, :], blend[i])
        root_vel_w = root_vel_w_0 * (1.0 - blend) + root_vel_w_1 * blend
        root_vel_b = math_utils.quat_rotate_inverse(root_quat, root_vel_w)
        root_ang_vel_w = root_ang_vel_w_0 * (1.0 - blend) + root_ang_vel_w_1 * blend
        root_ang_vel_b = math_utils.quat_rotate_inverse(root_quat, root_ang_vel_w)
        dof_pos = dof_pos_0 * (1.0 - blend) + dof_pos_1 * blend
        dof_vel = dof_vel_0 * (1.0 - blend) + dof_vel_1 * blend
        key_links_pos_w = key_links_pos_w_0 * (1.0 - blend.unsqueeze(1)) + key_links_pos_w_1 * blend.unsqueeze(1)
        key_links_pos_b = math_utils.quat_rotate_inverse(root_quat.unsqueeze(1), key_links_pos_w - root_pos_w.unsqueeze(1))
        
        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            "root_vel_b": root_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_links_pos_b": key_links_pos_b
        }

    def _calc_frame_blend(self, time, duration, num_frames, dt):
        
        assert len(time) == len(duration) == len(num_frames) == len(dt), \
            f"[MotionLoader] Length mismatch: time {len(time)}, duration {len(duration)}, num_frames {len(num_frames)}, dt {len(dt)}"
        
        phase = time / duration
        phase = np.clip(phase, 0.0, 1.0)
        
        frame_idx0 = (phase * (num_frames - 1)).astype(int)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        frame_idx0 = torch.from_numpy(frame_idx0).to(self.device).long()
        frame_idx1 = torch.from_numpy(frame_idx1).to(self.device).long()
        blend = torch.from_numpy(blend).to(self.device).float()

        return frame_idx0, frame_idx1, blend