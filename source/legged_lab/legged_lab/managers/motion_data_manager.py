from __future__ import annotations

import os
import enum
import joblib
import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject

from isaaclab.managers import ManagerBase, ManagerTermBase
from .motion_data_term_cfg import MotionDataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from legged_lab.utils.math import vel_forward_diff, ang_vel_from_quat_diff, quat_slerp, linear_interpolate, calc_frame_blend


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1


class MotionDataTerm(ManagerTermBase):
    
    cfg: MotionDataTermCfg
    _env: ManagerBasedEnv
    env: ManagerBasedRLEnv

    def __init__(self, cfg: MotionDataTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert os.path.exists(cfg.motion_data_dir), \
            f"Motion data directory {cfg.motion_data_dir} does not exist."
            
        self._load_motion_data()
        
        # Initialize sample buffer
        self._sample_buffer_motion_ids = None
        self._sample_buffer_times = None
        self._sample_buffer_cursor = 0
        self._build_sample_buffer()
        
    def _load_motion_data(self):
        # list the motion data files in the directory
        motion_files = [f for f in os.listdir(self.cfg.motion_data_dir) if f.endswith('.pkl')]
        if not motion_files:
            raise ValueError(f"No motion data files with .pkl extension found in {self.cfg.motion_data_dir}.")
        
        self.motion_weights_dict = self.cfg.motion_data_weights

        self.motion_durations = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        self.motion_weights = []
        self.motion_loop_modes = []
        
        self.root_pos_w = []
        self.root_quat = []
        self.root_vel_w = []
        self.root_ang_vel_w = []
        self.dof_pos = []
        self.dof_vel = []
        self.key_body_pos_w = []

        # only load the motion data files that are in the motion weights dict
        for motion_name, motion_weight in self.motion_weights_dict.items():
            # check if the motion file name is valid
            motion_file = f"{motion_name}.pkl"
            if motion_file not in motion_files:
                raise ValueError(f"Motion name {motion_name} defined in motion weights not found in motion data directory {self.cfg.motion_data_dir}. Available files: {motion_files}")

            # load the motion data file
            motion_path = os.path.join(self.cfg.motion_data_dir, motion_file)
            print(f"[Motion Data Manager] Loading motion data from {motion_path}...")
            motion_raw_data = joblib.load(motion_path)
            if not isinstance(motion_raw_data, dict):
                raise ValueError(f"Motion data file {motion_file} does not contain a valid dictionary.")
            
            # Some info about the motion
            fps = motion_raw_data["fps"]
            dt = 1.0 / fps
            num_frames = len(motion_raw_data["root_pos"])
            if num_frames < 2:
                raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")
            duration = dt * (num_frames - 1)
            loop_mode = motion_raw_data["loop_mode"]
            
            self.motion_durations.append(duration)
            self.motion_fps.append(fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
            self.motion_loop_modes.append(loop_mode)
            self.motion_weights.append(motion_weight)
            
            # Get the motion data
            
            # root position in world frame, shape (num_frames, 3)
            root_pos_w = torch.from_numpy(motion_raw_data["root_pos"]).to(self.device).float()
            root_pos_w.requires_grad_(False)
            # root rotation (quaternion) from world frame to body frame, shape (num_frames, 4), in (w, x, y, z) format
            root_quat = torch.from_numpy(motion_raw_data["root_rot"]).to(self.device).float()
            root_quat.requires_grad_(False)
            
            # root velocity in world frame, shape (num_frames, 3)
            root_vel_w = vel_forward_diff(root_pos_w, dt)
            root_vel_w.requires_grad_(False)
            
            # root angular velocity in world frame, shape (num_frames, 3)
            root_ang_vel_w = ang_vel_from_quat_diff(root_quat, dt, in_frame="world")
            root_ang_vel_w.requires_grad_(False)
            
            # dof position, shape (num_frames, num_dofs)
            dof_pos = torch.from_numpy(motion_raw_data["dof_pos"]).to(self.device).float()
            dof_pos.requires_grad_(False)
            
            # dof velocity, shape (num_frames, num_dofs)
            dof_vel = vel_forward_diff(dof_pos, dt)
            dof_vel.requires_grad_(False)
            
            # key body position in world frame, shape (num_frames, num_key_bodies, 3)
            key_body_pos_w = torch.from_numpy(motion_raw_data["key_body_pos"]).to(self.device).float()
            key_body_pos_w.requires_grad_(False)
            
            self.root_pos_w.append(root_pos_w)
            self.root_quat.append(root_quat)
            self.root_vel_w.append(root_vel_w)
            self.root_ang_vel_w.append(root_ang_vel_w)
            self.dof_pos.append(dof_pos)
            self.dof_vel.append(dof_vel)
            self.key_body_pos_w.append(key_body_pos_w)
        
        self.motion_fps = torch.tensor(self.motion_fps, dtype=torch.float32, device=self.device)
        self.motion_dt = torch.tensor(self.motion_dt, dtype=torch.float32, device=self.device)
        self.motion_durations = torch.tensor(self.motion_durations, dtype=torch.float32, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.int32, device=self.device)
        self.motion_loop_modes = torch.tensor(self.motion_loop_modes, dtype=torch.int32, device=self.device)
        # Get the normalized motion weights
        self.motion_weights = torch.tensor(self.motion_weights, dtype=torch.float32, device=self.device)
        
        # Adaptive weighting: Scale by sqrt(duration)
        # This balances between "Motion Sampling" (equal probability per clip) 
        # and "Frame Sampling" (equal probability per frame).
        # It prevents short motions from being over-sampled frame-wise, 
        # while ensuring they are not drowned out by long motions.
        self.motion_weights = self.motion_weights * torch.sqrt(self.motion_durations)
        self.motion_weights = self.motion_weights / torch.sum(self.motion_weights)
        
        # Some other infomation
        self.num_dofs = self.dof_pos[0].shape[1]
        self.num_key_bodies = self.key_body_pos_w[0].shape[1]
        
        # Concatenate all motion data along the first dimension
        self.root_pos_w = torch.cat(self.root_pos_w, dim=0)
        self.root_quat = torch.cat(self.root_quat, dim=0)
        self.root_vel_w = torch.cat(self.root_vel_w, dim=0)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w, dim=0)
        self.dof_pos = torch.cat(self.dof_pos, dim=0)
        self.dof_vel = torch.cat(self.dof_vel, dim=0)
        self.key_body_pos_w = torch.cat(self.key_body_pos_w, dim=0)
        
        num_motions = self.get_num_motions()
        self.motion_ids = torch.arange(num_motions, dtype=torch.long, device=self.device)
        
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)
        
        # Compute average velocity for each motion (in body frame)
        # This is used for command-conditioned sampling
        self._compute_motion_avg_velocities()
        
        return
    
    def _compute_motion_avg_velocities(self):
        """Compute average linear and angular velocity for each motion clip.
        
        The velocity is computed in the body frame for command matching.
        """
        num_motions = self.get_num_motions()
        
        # Shape: (num_motions, 3) for (vx, vy, vz)
        self.motion_avg_lin_vel_b = torch.zeros(num_motions, 3, device=self.device)
        # Shape: (num_motions,) for yaw rate (angular velocity around z-axis)
        self.motion_avg_ang_vel_z = torch.zeros(num_motions, device=self.device)
        
        for i in range(num_motions):
            start_idx = self.motion_start_indices[i].item()
            end_idx = start_idx + self.motion_num_frames[i].item()
            
            # Get velocity in world frame
            root_vel_w = self.root_vel_w[start_idx:end_idx]  # (num_frames, 3)
            root_ang_vel_w = self.root_ang_vel_w[start_idx:end_idx]  # (num_frames, 3)
            root_quat = self.root_quat[start_idx:end_idx]  # (num_frames, 4)
            
            # Transform linear velocity to body frame
            root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
            
            # Compute average
            self.motion_avg_lin_vel_b[i] = root_vel_b.mean(dim=0)
            self.motion_avg_ang_vel_z[i] = root_ang_vel_w[:, 2].mean()  # z-component
         
    # Some helper functions
    
    def get_num_motions(self) -> int:
        """Get the number of motions loaded."""
        return self.motion_num_frames.shape[0]
    
    def get_total_duration(self) -> float:
        """Get the total duration of all motions."""
        return torch.sum(self.motion_durations).item()

    def get_motion_durations(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the duration of a specific motion.

        Args:
            motion_id (torch.Tensor): A tensor of motion IDs for which to get the duration.

        Returns:
            float: The duration of the motion in seconds.
        """
        return self.motion_durations[motion_ids]
        
    def get_motion_loop_modes(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the loop mode of a specific motion.

        Args:
            motion_id (torch.Tensor): A tensor of motion IDs for which to get the loop mode.

        Returns:
            int: The loop mode of the motion.
        """
        return self.motion_loop_modes[motion_ids]

    def _build_sample_buffer(self):
        """Pre-compute a shuffled buffer of (motion_id, time) pairs based on weights."""
        # Use total frames as the baseline buffer size, but distribution follows weights
        buffer_size = torch.sum(self.motion_num_frames).item()
        
        # Calculate how many samples each motion should have in the buffer based on weights
        # counts = buffer_size * weights
        counts = (buffer_size * self.motion_weights).long()
        
        # Adjust counts to match buffer_size exactly (distribute remainder)
        diff = buffer_size - counts.sum().item()
        if diff > 0:
            # Add remainder to the motion with highest weight
            counts[torch.argmax(self.motion_weights)] += diff
            
        # Allocate buffers
        self._sample_buffer_motion_ids = torch.empty(buffer_size, dtype=torch.long, device=self.device)
        self._sample_buffer_times = torch.empty(buffer_size, dtype=torch.float32, device=self.device)
        
        start_idx = 0
        num_motions = self.get_num_motions()
        
        for i in range(num_motions):
            count = counts[i].item()
            if count <= 0:
                continue
                
            n_frames = self.motion_num_frames[i].item()
            dt = self.motion_dt[i].item()
            duration = self.motion_durations[i].item()
            
            # Fill Motion IDs
            self._sample_buffer_motion_ids[start_idx : start_idx + count] = i
            
            # Fill Times
            # Randomly sample times for this motion 'count' times
            # This implements over-sampling for high-weight short motions
            # and under-sampling for low-weight long motions
            
            # Generate random phases [0, 1]
            phases = torch.rand(count, device=self.device)
            
            # Convert to time, ensuring we don't exceed duration
            # We use a small epsilon to avoid index out of bounds at the exact end
            times = phases * (duration - 1e-6)
            
            self._sample_buffer_times[start_idx : start_idx + count] = times
            
            start_idx += count
            
        # Shuffle the entire buffer
        perm = torch.randperm(buffer_size, device=self.device)
        self._sample_buffer_motion_ids = self._sample_buffer_motion_ids[perm]
        self._sample_buffer_times = self._sample_buffer_times[perm]
        
        self._sample_buffer_cursor = 0
        # print(f"[MotionDataTerm] Built sample buffer with {total_frames} frames.")
        
    def sample_motions(self, n: int) -> torch.Tensor:
        """Sample a batch of motion IDs.

        Args:
            n (int): The number of motion IDs to sample.

        Returns:
            torch.Tensor: A tensor of sampled motion IDs, shape (n,).
        """
        ids_list = []
        times_list = []
        remaining = n
        
        while remaining > 0:
            if self._sample_buffer_motion_ids is None or self._sample_buffer_cursor >= self._sample_buffer_motion_ids.shape[0]:
                self._build_sample_buffer()
            
            available = self._sample_buffer_motion_ids.shape[0] - self._sample_buffer_cursor
            take = min(remaining, available)
            start = self._sample_buffer_cursor
            end = start + take
            
            ids_list.append(self._sample_buffer_motion_ids[start:end])
            times_list.append(self._sample_buffer_times[start:end])
            
            self._sample_buffer_cursor = end
            remaining -= take
            
        motion_ids = torch.cat(ids_list)
        self._last_sampled_times = torch.cat(times_list)
        
        return motion_ids
        
    def sample_times(self, motion_ids: torch.Tensor, truncate_time_start: float = None, truncate_time_end: float = None):
        """Sample time within the duration of the given motions.
        
        Args:
            motion_ids (torch.Tensor): A tensor of motion IDs, shape (batch_size,).
            truncate_time_start (float | None): If provided, the sampled time will be truncated
                from the start, i.e., sampled in [truncate_time_start, duration]. Default is None.
            truncate_time_end (float | None): If provided, the sampled time will be truncated
                from the end, i.e., sampled in [0, duration - truncate_time_end]. Default is None.
                
        Returns:
            torch.Tensor: A tensor of sampled times, shape (batch_size,).
        """
        motion_durations = self.motion_durations[motion_ids]
        
        # Calculate valid time range
        time_start = torch.zeros_like(motion_durations)
        time_end = motion_durations.clone()
        
        if truncate_time_start is not None:
            assert truncate_time_start >= 0, f"[MotionLoader] truncate_time_start must be non-negative, but got {truncate_time_start}."
            time_start = torch.clamp(time_start + truncate_time_start, min=0.0, max=motion_durations)
        
        if truncate_time_end is not None:
            assert truncate_time_end >= 0, f"[MotionLoader] truncate_time_end must be non-negative, but got {truncate_time_end}."
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)
        
        # Check if valid range exists
        valid_range = time_end - time_start
        if torch.any(valid_range <= 0.0):
            print("[Warning] Some motions have invalid time range after truncation (start >= end).")
            valid_range = torch.clamp(valid_range, min=1e-6)  # Prevent division by zero
        
        # Sample time within the valid range
        phase = torch.rand(motion_ids.shape, device=self.device)
        sample_times = time_start + phase * valid_range
        
        return sample_times

    def fa(
        self,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """支持 2D (n, 3) 或 3D (n, m, 3) 张量"""
        esp = 1e-3
        x = velocity[..., 0].abs()
        y = velocity[..., 1].abs()
        z = velocity[..., 2].abs()
        value = y*z / (x + esp) + x*z / (y + esp) + x*y / (z + esp)
        return value
    
    def fb(
        self,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """支持 2D (n, 3) 或 3D (n, m, 3) 张量"""
        esp = 1e-3
        x = velocity[..., 0].abs()
        y = velocity[..., 1].abs()
        z = velocity[..., 2].abs()
        value = y*z / (x + esp) * x*z / (y + esp) * x*y / (z + esp)
        return value

    def cosine_distance(
        self,
        commands: torch.Tensor,
        motion_vel: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """计算余弦距离矩阵
        
        Args:
            commands: (n, 3)
            motion_vel: (m, 3)
        Returns:
            (n, m) 余弦距离矩阵
        """
        # 矩阵乘法计算所有点积

        dot_product = torch.matmul(commands, motion_vel.t())  # (n, m)
        commands_norm = torch.norm(commands, dim=1, keepdim=True)  # (n, 1)
        motion_vel_norm = torch.norm(motion_vel, dim=1, keepdim=True)
        norms_product = torch.matmul(commands_norm, motion_vel_norm.t())  # (n, m)
        cosine_similarity = dot_product / (norms_product + eps)  # (n, m)
        cosine_distance = 1.0 - cosine_similarity  # (n, m)
        return cosine_distance
    
    def vel_diff(
        self,
        commands: torch.Tensor,
        motion_vel: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        velocity_blend_ratio: float = 1.0,
    ) -> torch.Tensor:
        """计算速度差异矩阵
        
        Args:
            commands: 目标速度命令 (n, 3)
            motion_vel: 动作数据的平均速度 (m, 3)
            env_ids: 环境ID
            velocity_blend_ratio: 当前速度和目标命令的混合比例
                0.0 = 完全使用当前速度
                1.0 = 完全使用目标命令
                0.5 = 当前速度和目标命令的平均
        """
        asset: RigidObject = self._env.scene["robot"]
        root_lin_vel = asset.data.root_lin_vel_b[env_ids, :2]
        root_ang_vel = asset.data.root_ang_vel_b[env_ids, 2]
        root_vel = torch.cat([root_lin_vel, root_ang_vel.unsqueeze(1)], dim=1)
        # print("root_vel:", root_vel)
        # 根据blend ratio混合当前速度和目标命令
        # velocity_blend_ratio = 1.0 时，完全使用commands
        # velocity_blend_ratio = 0.0 时，完全使用当前速度
        # velocity_blend_ratio = 0.5 时，取平均
        blended_target = (1.0 - velocity_blend_ratio) * root_vel + velocity_blend_ratio * commands
        # 计算方向mask：只考虑同方向分量
        # commands/motion_vel: (n, 3)/(m, 3)
        cmd_sign = torch.sign(commands)  # (n, 3)
        motion_sign = torch.sign(motion_vel)  # (m, 3)
        cmd_sign_exp = cmd_sign.unsqueeze(1)  # (n, 1, 3)
        motion_sign_exp = motion_sign.unsqueeze(0)  # (1, m, 3)
        match_mask = (cmd_sign_exp == motion_sign_exp).all(dim=2)  # (n, m)

        # 构造mask: (n, m, 3)，同方向分量为True
        direction_mask = (cmd_sign_exp == motion_sign_exp)  # (n, m, 3)

        # 计算差异
        diff_all = torch.norm(blended_target.unsqueeze(1) - motion_vel.unsqueeze(0), dim=2)  # (n, m)
        # 只考虑同方向分量
        diff_dir = torch.norm((blended_target.unsqueeze(1) - motion_vel.unsqueeze(0)) * direction_mask.float(), dim=2)

        # 匹配项用同方向分量差异，非匹配项用全向差异
        diff = torch.where(match_mask, diff_dir, diff_all)
        return diff
        

    def sample_motions_conditioned(
        self, 
        commands: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        min_prob: float = 0.001,
        velocity_blend_ratio: float | None = 1.0,
    ) -> torch.Tensor:
        """根据速度命令条件采样motion IDs
        
        Args:
            commands: 目标速度命令
            env_ids: 环境ID
            min_prob: 每个motion的最小采样概率
            velocity_blend_ratio: 当前速度和目标命令的混合比例 (0.0-1.0)
        """
        if env_ids is None:
            return
        esp = 1e-3
        num_motions = self.get_num_motions()
        cmd_vx = commands[:, 0]  # (n,)
        cmd_vy = commands[:, 1]  # (n,)
        cmd_ang_z = commands[:, 2].unsqueeze(1)  # (n,)
    
        # Motion velocity components: (num_motions, 3)
        motion_vx = self.motion_avg_lin_vel_b[:, 0]  # (num_motions,)
        motion_vy = self.motion_avg_lin_vel_b[:, 1]  # (num_motions,)
        motion_ang_z = self.motion_avg_ang_vel_z     # (num_motions,)
        
        # 过滤掉微小速度，避免噪声影响
        cmd_vx = torch.where(torch.abs(cmd_vx) < 0.2, 0, cmd_vx)
        cmd_vy = torch.where(torch.abs(cmd_vy) < 0.2, 0, cmd_vy)
        cmd_ang_z = torch.where(torch.abs(cmd_ang_z) < 0.2, 0, cmd_ang_z)
        commands = torch.where(torch.abs(commands) < 0.2, 0, commands)
    
        motion_vx = torch.where(torch.abs(motion_vx) < 0.2, 0, motion_vx)
        motion_vy = torch.where(torch.abs(motion_vy) < 0.2, 0, motion_vy)
        motion_ang_z = torch.where(torch.abs(motion_ang_z) < 0.2, 0, motion_ang_z)

        # motion_vy = motion_vy * 2

        motion_vel = torch.stack([motion_vx, motion_vy, motion_ang_z], dim=1)  # (num_motions, 3)
        
        # ========== 1. 方向一致性匹配过滤 ========== 
        # 只有与命令速度方向完全一致的数据集视为匹配
        cmd_sign = torch.sign(commands)  # (n, 3)
        motion_sign = torch.sign(motion_vel)  # (m, 3)
        cmd_sign_exp = cmd_sign.unsqueeze(1)  # (n, 1, 3)
        motion_sign_exp = motion_sign.unsqueeze(0)  # (1, m, 3)
        # 仅方向完全一致视为匹配
        match_mask = (cmd_sign_exp == motion_sign_exp).all(dim=2)  # (n, m)

        # ========== 2. 基于速度距离分配概率 ========== 
        vel_dist = self.vel_diff(commands, motion_vel, env_ids, velocity_blend_ratio)  # (n, m)
        scale = 3.0  # 可调参数
        similarities = torch.exp(-vel_dist * scale)  # 距离越小，相似度越高
        # print("similarities before mask:", similarities)
        # 概率分配：
        # 匹配项直接用相似度，非匹配项也用相似度但缩小权重
        mismatch_scale = 0.001
        similarities = torch.where(
            match_mask,
            similarities,
            similarities * mismatch_scale
        )

        # 概率归一化
        similarities_sum = similarities.sum(dim=1, keepdim=True).clamp(min=1e-8)
        probs = similarities / similarities_sum

        # 严格兜底：所有 motion 概率至少为 min_prob，总和为1
        num_motions = probs.shape[1]
        min_total = min_prob * num_motions
        mask = probs < min_prob
        remain = 1.0 - min_total
        probs_remain_sum = torch.sum(torch.where(mask, torch.zeros_like(probs), probs), dim=1, keepdim=True)
        all_mask = mask.all(dim=1)
        probs_scaled = torch.where(
            mask,
            min_prob,
            probs * (remain / (probs_remain_sum + 1e-8))
        )
        uniform_probs = torch.ones_like(probs) / num_motions
        probs = torch.where(
            all_mask.unsqueeze(1),
            uniform_probs,
            probs_scaled
        )

        # 处理 NaN 和 Inf
        probs = torch.where(torch.isnan(probs), torch.ones_like(probs) / num_motions, probs)
        probs = torch.where(torch.isinf(probs), torch.ones_like(probs) / num_motions, probs)

        # Sample motion IDs based on probabilities
        motion_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (n,)

        # Debug output
        # print("========== 综合相似度计算 ==========")
        # print(f"env_ids: {env_ids}")
        # print(f"root lin velocity: {self._env.scene['robot'].data.root_lin_vel_b[env_ids]}")
        # print(f"root ang velocity: {self._env.scene['robot'].data.root_ang_vel_b[env_ids]}")
        # print(f"commands: {commands}")
        # print(f"motion_vel: {motion_vel}")
        # print(f"match_mask: {match_mask}")
        # print(f"vel_dist: {vel_dist}")
        # print(f"similarities: {similarities}")
        # print(f"probs: {probs}")
        # print(f"motion_ids: {motion_ids}")
        
        return motion_ids

        
        
    def calc_motion_phase(self, motion_ids, times):
        motion_durations = self.motion_durations[motion_ids]
        loop_modes = self.motion_loop_modes[motion_ids]
        phase = calc_phase(times, motion_durations, loop_modes)
        return phase
    
    def _calc_frame_blend(self, motion_ids: torch.Tensor, times: torch.Tensor):
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]
        motion_start_indices = self.motion_start_indices[motion_ids]
        
        phase = self.calc_motion_phase(motion_ids, times)
        
        frame_idx0 = (phase * (num_frames - 1).float()).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1).float() - frame_idx0.float()
        
        frame_idx0 = frame_idx0 + motion_start_indices
        frame_idx1 = frame_idx1 + motion_start_indices
        
        return frame_idx0, frame_idx1, blend
        
    
    # def _allocate_temp_tensors(self, n):
    #     """Allocate temporary tensors for motion state computation."""
    #     root_pos_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_pos_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_quat_0 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
    #     root_quat_1 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
    #     root_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_ang_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_ang_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     dof_pos_0 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_pos_1 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_vel_0 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_vel_1 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     key_body_pos_w_0 = torch.empty([n, self.num_key_bodies, 3], dtype=torch.float32, device=self.device)
    #     key_body_pos_w_1 = torch.empty([n, self.num_key_bodies, 3], dtype=torch.float32, device=self.device)

    #     return (root_pos_w_0, root_pos_w_1,
    #             root_quat_0, root_quat_1,
    #             root_vel_w_0, root_vel_w_1,
    #             root_ang_vel_w_0, root_ang_vel_w_1,
    #             dof_pos_0, dof_pos_1,
    #             dof_vel_0, dof_vel_1,
    #             key_body_pos_w_0, key_body_pos_w_1)
    
    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
    
        root_pos_w_0 = self.root_pos_w[frame_idx0]
        root_pos_w_1 = self.root_pos_w[frame_idx1]
        root_quat_0 = self.root_quat[frame_idx0]
        root_quat_1 = self.root_quat[frame_idx1]
        root_vel_w_0 = self.root_vel_w[frame_idx0]
        root_vel_w_1 = self.root_vel_w[frame_idx1]
        root_ang_vel_w_0 = self.root_ang_vel_w[frame_idx0]
        root_ang_vel_w_1 = self.root_ang_vel_w[frame_idx1]
        dof_pos_0 = self.dof_pos[frame_idx0]
        dof_pos_1 = self.dof_pos[frame_idx1]
        dof_vel_0 = self.dof_vel[frame_idx0]
        dof_vel_1 = self.dof_vel[frame_idx1]
        key_body_pos_w_0 = self.key_body_pos_w[frame_idx0]
        key_body_pos_w_1 = self.key_body_pos_w[frame_idx1]
        
        # interpolate the values

        root_quat = quat_slerp(q0=root_quat_0, q1=root_quat_1, blend=blend)

        blend = blend.unsqueeze(-1)  # make it (n, 1) for broadcasting
        root_pos_w = torch.lerp(root_pos_w_0, root_pos_w_1, blend)
        root_vel_w = torch.lerp(root_vel_w_0, root_vel_w_1, blend)
        root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
        root_ang_vel_w = torch.lerp(root_ang_vel_w_0, root_ang_vel_w_1, blend)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
        dof_pos = torch.lerp(dof_pos_0, dof_pos_1, blend)
        dof_vel = torch.lerp(dof_vel_0, dof_vel_1, blend)
        key_body_pos_w = torch.lerp(key_body_pos_w_0, key_body_pos_w_1, blend.unsqueeze(1))
        key_body_pos_b = math_utils.quat_apply_inverse(
            root_quat.unsqueeze(1).expand(-1, self.num_key_bodies, -1),
            key_body_pos_w - root_pos_w.unsqueeze(1)
        )

        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            "root_vel_b": root_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_body_pos_b": key_body_pos_b,
        }
        
        
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


@torch.jit.script
def calc_phase(times: torch.Tensor, motion_duration: torch.Tensor, loop_mode: torch.Tensor) -> torch.Tensor:
    phase = times / motion_duration
        
    loop_wrap_mask = (loop_mode == int(LoopMode.WRAP.value))
    phase_wrap = phase[loop_wrap_mask]
    phase_wrap = phase_wrap - torch.floor(phase_wrap)
    phase[loop_wrap_mask] = phase_wrap
        
    phase = torch.clip(phase, 0.0, 1.0)

    return phase

        
        