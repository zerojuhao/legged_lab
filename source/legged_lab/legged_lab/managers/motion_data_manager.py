from __future__ import annotations

import os
import numpy as np
import joblib
import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation

from isaaclab.managers import ManagerBase, ManagerTermBase
from .motion_data_term_cfg import MotionDataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from legged_lab.utils.math import vel_forward_diff, ang_vel_from_quat_diff, quat_slerp, linear_interpolate, calc_frame_blend


class MotionDataTerm(ManagerTermBase):
    
    cfg: MotionDataTermCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: MotionDataTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert os.path.exists(cfg.motion_data_dir), \
            f"Motion data directory {cfg.motion_data_dir} does not exist."
            
        self._load_motion_data()
        
    def _load_motion_data(self):
        # list the motion data files in the directory
        motion_files = [f for f in os.listdir(self.cfg.motion_data_dir) if f.endswith('.pkl')]
        if not motion_files:
            raise ValueError(f"No motion data files with .pkl extension found in {self.cfg.motion_data_dir}.")
        
        self.motion_weights_dict = self.cfg.motion_data_weight
        
        # load the first motion data file to get some info
        motion_raw_data = joblib.load(
            os.path.join(self.cfg.motion_data_dir, motion_files[0])
        )
        if not isinstance(motion_raw_data, dict):
            raise ValueError(f"Motion data file {motion_files[0]} does not contain a valid dictionary.")
        self.retargeted_link_names = motion_raw_data["link_body_list"]
        self.retargeted_joint_names = self.cfg.dof_names # TODO: exported in motion data files
        
        self._build_joint_mapping()
        self._build_key_links_mapping()
        
        self.motion_data = []   # list to store processed motion data (a dictionary), each element corresponds to a motion
        self.motion_duration = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        
        # only load the motion data files that are in the motion weights dict
        for motion_name in self.motion_weights_dict.keys():
            motion_file = f"{motion_name}.pkl"
            if motion_file not in motion_files:
                raise ValueError(f"Motion name {motion_name} defined in motion weights not found in motion data directory {self.cfg.motion_data_dir}. Available files: {motion_files}")

            motion_path = os.path.join(self.cfg.motion_data_dir, motion_file)
            print(f"[Motion Data Manager] Loading motion data from {motion_path}...")
            motion_raw_data = joblib.load(motion_path)
            if not isinstance(motion_raw_data, dict):
                raise ValueError(f"Motion data file {motion_file} does not contain a valid dictionary.")
            
            # check link_body_list, it is optinal
            link_body_list = motion_raw_data.get("link_body_list", None)
            if link_body_list is None:
                raise ValueError(f"Motion data file {motion_file} does not contain 'link_body_list'.")
            if link_body_list != self.retargeted_link_names:
                raise ValueError(f"Link body list in {motion_file} does not match the expected link body list.")
            
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
            
        self.motion_fps = torch.tensor(self.motion_fps, dtype=torch.float32, device=self.device)
        self.motion_dt = torch.tensor(self.motion_dt, dtype=torch.float32, device=self.device)
        self.motion_duration = torch.tensor(self.motion_duration, dtype=torch.float32, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.int32, device=self.device)
            
        self.motion_names = list(self.motion_weights_dict.keys())
        # get motion weights and normalize them
        self.motion_weights = torch.tensor(list(self.motion_weights_dict.values()), dtype=torch.float32, device=self.device)
        self.motion_weights /= torch.sum(self.motion_weights)
        # set motion ids
        self.motion_ids = torch.arange(len(self.motion_names), dtype=torch.long, device=self.device)
        # print motion names and weights
        for id, name, weight in zip(self.motion_ids, self.motion_names, self.motion_weights):
            print(f"Motion ID: {id.item()}, name: {name}, weight: {weight.item():.4f}")

    def _build_joint_mapping(self):
        """Get the joint index mapping from lab joint names to retargeted joint names, and vice versa."""

        robot: Articulation = self._env.scene["robot"]
        self.lab_joint_names = robot.data.joint_names
            
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
        """
        Build the mapping from lab key links names to retargeted key links names.
        """

        # key links names in retarget motion data may be different from the lab key links names,
        # we need to load the key links mapping from the config
        key_links_mapping = self.cfg.key_links_mapping
        name_lab_idx_retargeted = {}
        for retargeted_name, lab_name in key_links_mapping.items():
            if retargeted_name not in self.retargeted_link_names:
                raise ValueError(f"[MotionLoader] Retargeted link name {retargeted_name} not found in retargeted motion data {self.retargeted_link_names}.")
            name_lab_idx_retargeted[lab_name] = self.retargeted_link_names.index(retargeted_name)
        robot:Articulation = self._env.scene["robot"]
        self.lab_body_names = robot.data.body_names
        indices, names, values = string_utils.resolve_matching_names_values(
            data=name_lab_idx_retargeted, 
            list_of_strings=self.lab_body_names, 
            preserve_order=False
        )
        # in lab body order, with value as the index in retargeted link names
        self.retargeted_key_links_mapping = values
        
    def _process_motion_data(self, motion_raw_data) -> dict[str, torch.Tensor]:
        """
        Process the raw motion data
        
        Args:
            motion_raw_data (dict): The raw motion data dictionary.
        
        Returns:
            dict: Processed motion data
        """
        fps = motion_raw_data["fps"]
        dt = 1.0 / fps
        num_frames = len(motion_raw_data["root_pos"])
        if num_frames < 2:
            raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")
        
        # root position in world frame, shape (num_frames, 3)
        root_pos_w = torch.from_numpy(motion_raw_data["root_pos"]).to(self.device).float()
        root_pos_w.requires_grad_(False)
        # root rotation (quaternion) from world frame to body frame, shape (num_frames, 4)
        root_quat = torch.from_numpy(motion_raw_data["root_rot"]).to(self.device).float()
        root_quat.requires_grad_(False)
        root_quat = math_utils.convert_quat(root_quat, "wxyz") # convert to w, x, y, z format
        root_quat = math_utils.quat_unique(root_quat)  # ensure unique quaternions
        root_quat = math_utils.normalize(root_quat)  # ensure quaternion is normalized
        
        # root velocity in world frame, shape (num_frames, 3)
        root_vel_w = vel_forward_diff(root_pos_w, dt)
        
        # root angular velocity in world frame, shape (num_frames, 3)
        root_ang_vel_w = ang_vel_from_quat_diff(root_quat, dt, in_frame="world")
        
        # joint position, shape (num_frames, num_joints)
        dof_pos = motion_raw_data["dof_pos"][:, self.retargeted_to_lab_mapping]
        dof_pos = torch.from_numpy(dof_pos).to(self.device).float()
        dof_pos.requires_grad_(False)
        
        # joint velocity, shape (num_frames, num_joints)
        dof_vel = vel_forward_diff(dof_pos, dt)
        
        # key links position in body frame, shape (num_frames, num_key_links, 3)
        key_links_pos_b = motion_raw_data["local_body_pos"][:, self.retargeted_key_links_mapping, :]
        key_links_pos_b = torch.from_numpy(key_links_pos_b).to(self.device).float()
        key_links_pos_b.requires_grad_(False)
        
        # we need to interpolate in later usage, so pos in world frame is needed
        key_links_pos_w = root_pos_w.unsqueeze(1) + math_utils.quat_apply(
            root_quat.unsqueeze(1).expand(-1, len(self.retargeted_key_links_mapping), -1),
            key_links_pos_b
        )
        
        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            "root_ang_vel_w": root_ang_vel_w,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_links_pos_w": key_links_pos_w,
        }
        
    def _print_motion_info(self):
        """Print the information of the loaded motions."""
        print("="* 80)
        print(f"[MotionLoader] Loaded {len(self.motion_names)} motions with total duration: {self.get_total_duration()} seconds.")
        print("-"* 80)
        for i, motion_name in enumerate(self.motion_names):
            motion_id = self.motion_ids[i].item()
            weight = self.motion_weights[i].item()
            duration = self.motion_duration[i]
            fps = self.motion_fps[i]
            num_frames = self.motion_num_frames[i]
            base_lin_vel_avg = torch.mean(torch.norm(self.motion_data[i]["root_vel_w"], dim=1)).item()
            base_lin_vel_max = torch.max(torch.norm(self.motion_data[i]["root_vel_w"], dim=1)).item()
            print(f"  - ID {motion_id} - {motion_name}: weight = {weight:.2f}, duration={duration:.2f}s, fps={fps}, num_frames={num_frames}, base_lin_vel_avg={base_lin_vel_avg:.2f}m/s, base_lin_vel_max={base_lin_vel_max:.2f}m/s")
        print("-"* 80)
        # print joint mapping
        print(f"Lab Joint Names: {self.lab_joint_names}")
        print(f"Retargeted Joint Names: {self.retargeted_joint_names}")
        print(f"Retargeted to Lab Joint Mapping: {self.retargeted_to_lab_mapping}")
        print("-"* 80)
        # print key links mapping
        print(f"Lab Body Names: {self.lab_body_names}")
        print(f"Retargeted Link Names: {self.retargeted_link_names}")
        print(f"Retargeted Key Links Mapping: {self.retargeted_key_links_mapping}")
        key_links = [self.retargeted_link_names[i] for i in self.retargeted_key_links_mapping]
        print(f"Key links: {key_links}")
        print("="* 80)
        
    # -------------------------------------------------
    # Some helper functions
    # -------------------------------------------------
        
    def get_total_duration(self) -> float:
        """Get the total duration of all motions."""
        return torch.sum(self.motion_duration).item()

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

    def get_motion_duration(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the duration of a specific motion.

        Args:
            motion_id (np.ndarray): An array of motion IDs for which to get the duration.

        Returns:
            float: The duration of the motion in seconds.
        """
        return self.motion_duration[motion_ids]

    def sample_motions(self, n: int) -> torch.Tensor:
        """Sample a batch of motion IDs.

        Args:
            n (int): The number of motion IDs to sample.

        Returns:
            torch.Tensor: A tensor of sampled motion IDs, shape (n,).
        """
        motion_ids = torch.multinomial(self.motion_weights, num_samples=n, replacement=True)
        return motion_ids
    
    def sample_times(self, motion_ids:torch.Tensor, truncate_time=None):

        phase = torch.rand(motion_ids.shape, device=self.device)
        motion_durations = self.motion_duration[motion_ids]
        
        if truncate_time is not None:
            assert truncate_time > 0, f"[MotionLoader] Truncate time must be positive, but got {truncate_time}."
            motion_durations = torch.clamp(motion_durations - truncate_time, min=0.0)

        # Sample time for each motion
        sample_times = phase * motion_durations
        return sample_times
        
    def _allocate_temp_tensors(self, n):
        """Allocate temporary tensors for motion state computation."""
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
        key_links_pos_w_0 = torch.empty([n, len(self.retargeted_key_links_mapping), 3], dtype=torch.float32, device=self.device)
        key_links_pos_w_1 = torch.empty([n, len(self.retargeted_key_links_mapping), 3], dtype=torch.float32, device=self.device)

        return (root_pos_w_0, root_pos_w_1,
                root_quat_0, root_quat_1,
                root_vel_w_0, root_vel_w_1,
                root_ang_vel_w_0, root_ang_vel_w_1,
                dof_pos_0, dof_pos_1,
                dof_vel_0, dof_vel_1,
                key_links_pos_w_0, key_links_pos_w_1)
        
    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:

        n = motion_ids.shape[0]
        
        if not hasattr(self, "root_pos_w_0") or self.root_pos_w_0.shape[0] != n:
            # allocate new tensors if the number of motion_ids has changed
            (self.root_pos_w_0, self.root_pos_w_1,
             self.root_quat_0, self.root_quat_1,
             self.root_vel_w_0, self.root_vel_w_1,
             self.root_ang_vel_w_0, self.root_ang_vel_w_1,
             self.dof_pos_0, self.dof_pos_1,
             self.dof_vel_0, self.dof_vel_1,
             self.key_links_pos_w_0, self.key_links_pos_w_1) = self._allocate_temp_tensors(n)
        
        motion_durations = self.motion_duration[motion_ids]
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]
        
        frame_idx0, frame_idx1, blend = calc_frame_blend(motion_times, motion_durations, num_frames, dt)
        
        unique_ids, inverse_indices = torch.unique(motion_ids, return_inverse=True)
            
        for i, uid in enumerate(unique_ids):
            mask = inverse_indices == i
            ids = torch.where(mask)[0]
            motion = self.get_motion(uid.item())

            self.root_pos_w_0[ids, :] = motion["root_pos_w"][frame_idx0[ids], :]
            self.root_pos_w_1[ids, :] = motion["root_pos_w"][frame_idx1[ids], :]

            self.root_quat_0[ids, :] = motion["root_quat"][frame_idx0[ids], :]
            self.root_quat_1[ids, :] = motion["root_quat"][frame_idx1[ids], :]

            self.root_vel_w_0[ids, :] = motion["root_vel_w"][frame_idx0[ids], :]
            self.root_vel_w_1[ids, :] = motion["root_vel_w"][frame_idx1[ids], :]

            self.root_ang_vel_w_0[ids, :] = motion["root_ang_vel_w"][frame_idx0[ids], :]
            self.root_ang_vel_w_1[ids, :] = motion["root_ang_vel_w"][frame_idx1[ids], :]

            self.dof_pos_0[ids, :] = motion["dof_pos"][frame_idx0[ids], :]
            self.dof_pos_1[ids, :] = motion["dof_pos"][frame_idx1[ids], :]

            self.dof_vel_0[ids, :] = motion["dof_vel"][frame_idx0[ids], :]
            self.dof_vel_1[ids, :] = motion["dof_vel"][frame_idx1[ids], :]

            self.key_links_pos_w_0[ids, :, :] = motion["key_links_pos_w"][frame_idx0[ids], :, :]
            self.key_links_pos_w_1[ids, :, :] = motion["key_links_pos_w"][frame_idx1[ids], :, :]

        # interpolate the values

        root_quat = quat_slerp(q0=self.root_quat_0, q1=self.root_quat_1, blend=blend)

        blend = blend.unsqueeze(-1)  # make it (n, 1) for broadcasting
        root_pos_w = linear_interpolate(self.root_pos_w_0, self.root_pos_w_1, blend)
        root_vel_w = linear_interpolate(self.root_vel_w_0, self.root_vel_w_1, blend)
        root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
        root_ang_vel_w = linear_interpolate(self.root_ang_vel_w_0, self.root_ang_vel_w_1, blend)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
        dof_pos = linear_interpolate(self.dof_pos_0, self.dof_pos_1, blend)
        dof_vel = linear_interpolate(self.dof_vel_0, self.dof_vel_1, blend)
        key_links_pos_w = linear_interpolate(self.key_links_pos_w_0, self.key_links_pos_w_1, blend.unsqueeze(1))
        num_key_links = key_links_pos_w.shape[1]
        key_links_pos_b = math_utils.quat_apply_inverse(root_quat.unsqueeze(1).expand(-1, num_key_links, -1), 
                                                         key_links_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_links, -1))
        
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
        
    def mini_batch_generator(self, num_transitions_per_env, num_mini_batches, num_epochs=8):
        
        dt = self._env.cfg.sim.dt * self._env.cfg.decimation # TODO: use self._env.cfg.sim.dt or self._env.sim.get_physics_dt() ?
        num_envs = self._env.scene.num_envs

        batch_size = num_transitions_per_env * num_envs
        mini_batch_size = batch_size // num_mini_batches
        
        if batch_size % num_mini_batches != 0:
            raise ValueError(f"Epoch batch size {batch_size} is not divisible by number of mini-batches {num_mini_batches}.")

        num_steps = self.cfg.num_steps
        if num_steps < 2:
            raise ValueError(f"[MotionDataTerm] Number of steps must be at least 2, but got {num_steps}.")
        motion_ids = self.sample_motions(batch_size)
        motion_start_times = self.sample_times(motion_ids, truncate_time=dt*(num_steps-1))
        motion_times = [motion_start_times + i * dt for i in range(num_steps)]
        
        # get the observation terms to extract from the motion state
        amp_obs_terms = self._env.observation_manager.active_terms["amp"]
        extract_funcs = []
        for term in amp_obs_terms:
            func_name = f"_extract_{term}"
            if hasattr(self, func_name):
                extract_funcs.append(getattr(self, func_name))
            else:
                raise ValueError(f"[MotionDataTerm] Observation term '{term}' is not supported, please check the observation terms in the config file.")
        
        motion_state_dicts = [self.get_motion_state(motion_ids, t) for t in motion_times]
        # motion_next_state_dict = self.get_motion_state(motion_ids, motion_next_times)
        
        motion_states = []
        for ms_dict in motion_state_dicts:
            for func in extract_funcs:
                motion_states.append(func(ms_dict))
        motion_states_tensor = torch.cat(motion_states, dim=1).to(self.device) # (N, num_steps*D), where D is the total dimension of the motion state
        
        for epoch in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mini_batch_idx = indices[start:end]
                
                yield motion_states_tensor[mini_batch_idx]
        
    def _extract_dof_pos(self, motion_state: dict) -> torch.Tensor:
        """Extract the dof position from the motion state."""
        return motion_state["dof_pos"]  # (N, num_dof)
    
    def _extract_dof_vel(self, motion_state: dict) -> torch.Tensor:
        """Extract the dof velocity from the motion state."""
        return motion_state["dof_vel"]  # (N, num_dof)
    
    def _extract_base_lin_vel_b(self, motion_state: dict) -> torch.Tensor:
        """Extract the base linear velocity in body frame from the motion state."""
        return motion_state["root_vel_b"]   # (N, 3)
    
    def _extract_base_ang_vel_b(self, motion_state: dict) -> torch.Tensor:
        """Extract the base angular velocity in body frame from the motion state."""
        return motion_state["root_ang_vel_b"]   # (N, 3)
    
    def _extract_base_pos_z(self, motion_state: dict) -> torch.Tensor:
        return motion_state["root_pos_w"][:, 2].unsqueeze(-1)  # (N, 1)

    def _extract_key_links_pos_b(self, motion_state: dict) -> torch.Tensor:
        """Extract the key links position in body frame from the motion state."""
        return motion_state["key_links_pos_b"].flatten(start_dim=1)  # (N, M*3), where M is the number of key links, and 3 is the x, y, z position in body frame
    
    def _extract_projected_gravity(self, motion_state: dict) -> torch.Tensor:
        """Extract the projected gravity from the motion state."""
        gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 3)
        root_quat = motion_state["root_quat"]  # (N, 4)
        gravity_vec_w = gravity_vec_w.expand(root_quat.shape[0], -1)  # (N, 3)
        projected_gravity = math_utils.quat_apply_inverse(root_quat, gravity_vec_w)
        return projected_gravity
        

class MotionDataManager(ManagerBase):
    """Manager for motion data.
    
    This manager is responsible for loading and managing motion data terms.
    Each motion data term is responsible for managing a group of data.
    """
    
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        
        # check that cfg is not None
        if cfg is None:
            raise ValueError("MotionDataManager requires a valid configuration object.")
        
        self._terms: dict[str, MotionDataTerm] = {}
        self._term_cfgs: dict[str, MotionDataTermCfg] = {}
        
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for motion data manager."""
        msg = f"<MotionDataManager> contains {len(self._terms)} active terms.\n"
        
        # create table for term information
        table = PrettyTable()
        table.title = "Motion Data Manager Terms"
        table.field_names = ["Index", "Motion Dataset", "Total Duration"]
        # set alignment of table columns
        table.align["Motion Dataset"] = "l"
        table.align["Total Duration"] = "r"
        # add info on each term
        for index, (term_name, term) in enumerate(self._terms.items()):
            table.add_row([index, term_name, term.get_total_duration()])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
    
    def get_generators(self) -> dict[str, callable]:
        """Get the generators for the motion data manager."""
        generators = {}
        for term_name, term in self._terms.items():
            generators[term_name] = term.mini_batch_generator
        return generators
    
    def get_term(self, term_name: str) -> MotionDataTerm:
        """Get the motion data term by name."""
        if term_name not in self._terms:
            raise KeyError(f"Motion data term '{term_name}' not found.")
        return self._terms[term_name]

    def get_term_weights(self) -> dict[str, float]:
        """Get the weights of the motion data terms."""
        term_weights = {}
        for term_name, term in self._terms.items():
            term_weights[term_name] = term.cfg.weight
        return term_weights

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, MotionDataTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MotionDataTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = MotionDataTerm(term_cfg, self._env)
            # add class to dict
            self._terms[term_name] = term
            self._term_cfgs[term_name] = term_cfg


