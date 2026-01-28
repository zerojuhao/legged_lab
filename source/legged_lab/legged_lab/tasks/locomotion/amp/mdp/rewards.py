from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_conjugate, quat_apply
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
    
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # return torch.exp(-lin_vel_error / std**2)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # return torch.exp(-ang_vel_error / std**2)
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def feet_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2



def knee_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 只对超过 threshold 的空中时间给予正奖励（防止负值惩罚）
    positive_air = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(positive_air * first_contact.float(), dim=1)
    # no reward for zero command
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    return torch.sum(diff, dim=1)


def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    每只脚单独计算：
    脚接触地面时，z 方向加速度大 → 惩罚
    """

    asset = env.scene["robot"]

    # 1️⃣ 取所有 body 的线加速度 (world)
    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # 2️⃣ 取“脚”的 z 方向线加速度
    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    # 3️⃣ 取脚的接触状态
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    # 4️⃣ 每只脚：加速度平方 × 接触状态
    acc_penalty = (foot_acc_z ** 2) * in_contact.float()

    # 防止数值爆炸（非常重要）
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    # 5️⃣ 所有脚加起来
    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    # 仅当速度命令较小（小于 1.5）时才启用该奖励
    cmd = env.command_manager.get_command(command_name)
    
    # 使用 xy 分量的速度范数作为速度大小判断
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def low_speed_sway_penalty(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear and angular velocities when command velocity is below threshold.
    
    This function penalizes the robot for moving (both linear and angular) when the command
    speed is very small, encouraging the robot to remain still during low-speed commands.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get command velocity
    command = env.command_manager.get_command(command_name)
    command_speed = torch.norm(command[:, :2], dim=1)
    
    # Penalize linear velocity in xy plane
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    
    # Penalize angular velocity
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    
    # Total velocity penalty
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    
    # Apply penalty only when command speed is below threshold
    return vel_penalty * (command_speed < command_threshold).float()

def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # 只在低于目标高度时惩罚，高于目标高度无惩罚
    height_diff = adjusted_target_height - asset.data.root_pos_w[:, 2]
    penalty = torch.square(torch.clamp(height_diff, min=0.0))
    return penalty


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis angular velocities of specified bodies using L2 squared kernel.

    NOTE: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their angular velocities contribute to the penalty.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the penalty for xy angular velocities
    ang_vel_xy = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=[1, 2])


def body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation of x-axis of specified bodies from horizontal using L2 squared kernel.

    This penalizes the z-component of the body's x-axis in world coordinates, encouraging the x-axis to remain level.

    NOTE: Only the bodies configured in :attr:`asset_cfg.body_ids` will contribute to the penalty.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get body quaternions
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # shape: (num_envs, num_bodies, 4)
    # local x-axis vector
    x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).unsqueeze(0).expand(body_quat.shape[0], body_quat.shape[1], -1)
    # transform to world coordinates
    x_axis_world = math_utils.quat_apply(body_quat, x_axis_local)  # shape: (num_envs, num_bodies, 3)
    # penalize the z-component (deviation from horizontal)
    return torch.sum(torch.square(x_axis_world[:, :, 2]), dim=1)



##########################
# Agile But Safe Rewards #
##########################

def reach_pos_target_soft(
    env: ManagerBasedRLEnv, 
    position_target_sigma_soft: float = 2.0, 
    command_name: str = "pose_command"
    ) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    distance = torch.norm(position_targets, dim=1)
    return (1 /(1 + torch.square(distance / position_target_sigma_soft)))

def reach_pos_target_tight(
    env: ManagerBasedRLEnv, 
    position_target_sigma_tight: float = 0.5, 
    command_name: str = "pose_command"
    ) -> torch.Tensor:
    
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    distance = torch.norm(position_targets, dim=1)    
    return (1 /(1 + torch.square(distance / position_target_sigma_tight)))

def reach_heading_target(
    env: ManagerBasedRLEnv, 
    heading_target_sigma: float = 0.1, 
    position_target_sigma_soft: float = 2.0, 
    command_name: str = "pose_command"
    ) -> torch.Tensor:
    
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    
    distance = torch.norm(position_targets, dim=1)
    near_goal = (distance < position_target_sigma_soft)
    angle_difference = torch.abs(command[:, 2])
    heading_rew = 1 /(1 + torch.square(angle_difference / heading_target_sigma))
    return heading_rew * near_goal


def reach_pos_target_times_heading(
    env: ManagerBasedRLEnv,
    position_target_sigma: float = 0.5,
    command_name: str = "pose_command"
    ) -> torch.Tensor:
    
    # Compute distance between robot and target positions
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    distance = torch.norm(position_targets, dim=1)    
    # Compute heading angle of the robot
    angle_difference = torch.abs(command[:, 2])  # 0 radians represents positive x-axis direction
    
    # Apply a penalty if the robot deviates from the positive x-axis direction
    heading_penalty = torch.abs(torch.cos(angle_difference)) # avoid negative rewards
    
    # Compute the reward based on distance and heading penalty
    distance_reward = (1 / (1 + torch.square(distance / position_target_sigma)))
    
    # Combine distance reward and heading penalty
    combined_reward = distance_reward * heading_penalty * torch.exp(- angle_difference.abs())

    return combined_reward


def velo_dir(
    env: ManagerBasedRLEnv,
    position_target_sigma_tight: float = 0.5,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward velocity when moving roughly toward the target.

    - command is expected in base frame: (x, y, heading)
    - gives small continuous reward proportional to forward speed when moving toward target,
      plus a fixed bonus when very close to the target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # get command (base frame) and robot base velocities
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    distance = torch.norm(position_targets, dim=1)

    # in base frame forward is +x axis, so good direction means dir_unit.x not strongly negative
    good_dir = command[:, 2].abs() < 0.25

    # root forward velocity in base frame
    forward_vel = asset.data.root_lin_vel_b[:, 0]
    # forward = forward_vel > 0.0
    forward_reward = forward_vel.clip(min=0.0) * good_dir * (distance > position_target_sigma_tight) + 1.0 * (distance < position_target_sigma_tight)


    return forward_reward 


def stand_still_pos(
    env: ManagerBasedRLEnv,
    position_target_sigma_tight: float = 0.5,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from a nominal standing pose when commanded to stay near the target.

    - builds a small stand_bias (matching original structure: every 3rd joint pattern)
    - returns the L1 deviation summed over joints, masked by being close to the goal.
    """
    # get articulation joint positions and defaults
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos

    # distance to target (command in base frame)
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    distance = torch.norm(position_targets, dim=1)

    # L1 deviation from desired standing pose, applied only when close to target
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    
    return deviation * (distance < position_target_sigma_tight)


def nomove(
    env: ManagerBasedRLEnv,
    position_target_sigma_soft: float = 2.0,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize agents that stand still while facing away from the target and far from it."""
    # robot base velocities and angular velocity
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_xy = asset.data.root_lin_vel_b[:, :2]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]

    # static = low linear and angular velocity
    static = torch.logical_and(torch.norm(lin_vel_xy, dim=-1) < 0.1, torch.abs(ang_vel_z) < 0.1)

    # direction to target (base frame) and bad_dir test
    command = env.command_manager.get_command(command_name)
    position_targets = command[:, :2]
    distance = torch.norm(position_targets, dim=1)
    bad_dir = command[:, 2].abs() > 0.25

    # apply penalty only when far from target
    return static * bad_dir * (distance > position_target_sigma_soft)


###########################
# Walk These Ways Rewards #
###########################
def reward_jump(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "pose_command",
    base_height_target: float = 0.3,
) -> torch.Tensor:
    """奖励跳跃高度接近目标值"""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_height = asset.data.root_pos_w[:, 2]
    jump_height_target = env.command_manager.get_command(command_name)[:, 3] + base_height_target
    reward = -torch.square(body_height - jump_height_target)

    return reward

                
def tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_force_sigma: float = 50.0,
) -> torch.Tensor:
    """
    奖励脚部接触力与期望接触状态的匹配
    """
    contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]
    # 获取足端受力
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    desired_contact_states = env.desired_contact_states
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(4):
        reward += - (1 - desired_contact_states[:, i]) * (
                    1 - torch.exp(-1 * foot_forces[:, i] ** 2 / gait_force_sigma))

    return reward / 4 # 对所有脚取均值


def tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gait_vel_sigma: float = 10.0,
) -> torch.Tensor:
    """
    奖励脚部速度与期望接触状态的匹配
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取足端速度
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    desired_contact_states = env.desired_contact_states

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(4):
        reward += - ( desired_contact_states[:, i] * (
                    1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / gait_vel_sigma)))

    return reward / 4 # 对所有脚取均值


def feet_clearance_cmd_linear(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "pose_command",
) -> torch.Tensor:
    """
    奖励脚在摆动相时达到命令指定的高度（线性相位）
    """
    # phases: [num_feet]，用于区分步态周期
    phases = 1 - torch.abs(1.0 - torch.clip((env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取脚的高度
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].view(env.num_envs, -1)
    # 获取目标高度（命令第10维），并加上脚半径偏置

    target_height = env.command_manager.get_command(command_name)[:, 9].unsqueeze(1) * phases + 0.02
    
    # 只对非接触脚计算奖励
    desired_contact_states = env.desired_contact_states
    rew_foot_clearance = torch.square(target_height - foot_height) * (1 - desired_contact_states)
    reward = torch.sum(rew_foot_clearance, dim=1)
    # print("foot_height:", foot_height)
    # print("target_height:", target_height)
    
    return reward

def reward_raibert_heuristic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "pose_command",
    
) -> torch.Tensor:
    """
    Raibert步态启发式奖励：鼓励足端位置接近启发式目标
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # 足端位置转到身体坐标系
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)

    for i in range(4):
        # 将足端位置从世界坐标系转换到身体坐标系
        footsteps_in_body_frame[:, i, :] = quat_apply(
            quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # 期望步态宽度和长度
    commands = env.command_manager.get_command(command_name)
    
    # 使用命令中的步态参数
    desired_stance_width = commands[:, 12:13]
    desired_stance_length = commands[:, 13:14]

    desired_ys_nom = torch.cat([
        desired_stance_width / 2, -desired_stance_width / 2,
        desired_stance_width / 2, -desired_stance_width / 2
    ], dim=1)

    desired_xs_nom = torch.cat([
        desired_stance_length / 2, desired_stance_length / 2,
        -desired_stance_length / 2, -desired_stance_length / 2
    ], dim=1)

    # Raibert offsets
    phases = torch.abs(1.0 - (env.foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = commands[:, 4]

    y_vel_des = desired_stance_length / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    desired_ys_offset[:, 2:4] *= -1    
    desired_xs_offset = phases * (0.5 / frequencies.unsqueeze(1))

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset

    desired_footsteps_body_frame = torch.cat(
        (desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2
    )

    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

    reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
   
    return reward



def staged_navigation_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    distance_threshold: float = 0.5,   #  距离目标的阈值
    near_goal_threshold: float = 2.0,  # 接近目标的距离阈值
    obstacle_threshold: float = 0.8,  # 前方障碍物的距离阈值
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    ray_caster: RayCaster = env.scene[sensor_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :2] # 目标位置
    des_heading = torch.atan2(command[:, 1], command[:, 0]).abs() # 目标朝向
    des_heading = (des_heading - 0.15).clamp(min=0.0)  # 朝向误差阈值处理
    distance = torch.norm(des_pos, dim=1) # 机器人到目标位置的距离
    
    vx = asset.data.root_lin_vel_b[:, 0] # 机器人在base frame下的前向速度
    vy = asset.data.root_lin_vel_b[:, 1] # 机器人在base frame下的侧向速度
       
    # 当前移动方向与期望朝向误差（规范化到 [-pi, pi]）
    move_dir_angle = torch.atan2(vy, vx)
    move_heading_error = move_dir_angle.abs()
    
    # 雷达前方最近障碍物距离（已在外部被 clamp）
    origin = ray_caster.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    hits = ray_caster.data.ray_hits_w  # [num_envs, num_rays, 3]
    distances = torch.norm(hits - origin, dim=-1).clamp(min=0.2, max=5.0)  # [num_envs, num_rays]
    front_min_dist = torch.min(distances, dim=1).values  # [num_envs]

    # 雷达前方距离
    num_rays = ray_caster.data.ray_hits_w.shape[1]
    angles = torch.linspace(-torch.pi/3, torch.pi/3, num_rays, device=env.device)
    front_angle = torch.pi/10  # 18°
    front_mask = (angles.abs() <= front_angle)
    distances_ray = torch.norm(hits - origin, dim=-1).clamp_max(4)
    front_min_dist_mask = distances_ray[:, front_mask].min(dim=1).values

    # 1) 朝向匹配奖励：误差越小奖励越高
    heading_reward = torch.exp(- (des_heading)*3) * torch.exp(- move_heading_error*3)  # 当误差小于0.5rad时，基本全量奖励，误差增大时指数衰减

    # 2) 沿期望朝向的速度（越朝向目标前进越好），只奖励正向分量

    progress_reward = vx.clamp(min=0.0) * torch.exp(-move_heading_error) * torch.exp(-des_heading) # 当误差小于0.5rad时，基本全量奖励前向速度，误差增大时指数衰减
    # 3) 障碍物清除奖励：鼓励与障碍物保持距离
    safe_min = 0.5
    denom = max(obstacle_threshold - safe_min, 1e-6)
    obs_clearance = torch.clamp(front_min_dist - safe_min, min=0.0, max=obstacle_threshold - safe_min) / denom  # 0..1 when front_min_dist within [safe_min, obstacle_threshold]
    # 保持变量名兼容下游使用
    
    obs_approach_raw = obs_clearance + torch.where(front_min_dist_mask < safe_min, torch.exp(-vx), 0)

    # 距离目标的奖励：距离越小越好，使用 near_goal_threshold 归一化尺度
    dist_reward = torch.exp(- distance)

    # 分阶段加权：远离目标优先前进与保持清除，接近目标优先朝向精确并靠近目标
    is_far = distance > near_goal_threshold
    is_near = torch.logical_and(distance <= near_goal_threshold, distance > distance_threshold)
    is_at_goal = distance <= distance_threshold
    
    far_reward = 0.70 * progress_reward + 0.30 * heading_reward + 0.01 * obs_approach_raw + 1.0 * dist_reward
    near_reward = 0.10 * progress_reward + 0.90 * heading_reward + 0.01 * obs_approach_raw + 1.0 * dist_reward
    goal_reward = 0.01 * progress_reward + 0.01 * heading_reward + 0.01 * obs_approach_raw + 2.0 * torch.exp(-torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)) + 1.0 * dist_reward

    
    reward = torch.zeros_like(distance)
    reward = torch.where(is_far, far_reward, reward)
    reward = torch.where(is_near, near_reward, reward)
    reward = torch.where(is_at_goal, goal_reward, reward)

    reward = torch.clamp(reward, min=-1.0, max=2.0)

    return reward