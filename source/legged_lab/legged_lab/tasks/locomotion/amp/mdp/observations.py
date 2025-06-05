from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    

def feet_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Compute the position of the feet in the base frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get the pose of the root in world frame
    base_pos_w = asset.data.root_pos_w      # shape: (num_instances, 3).
    base_quat_w = asset.data.root_quat_w    # shape: (num_instances, 4), w, x, y, z order.
    # get the pose of the feet in world frame
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # get the pose of the feet in base frame
    feet_pos_b = math_utils.quat_rotate_inverse(base_quat_w.unsqueeze(1), feet_pos_w - base_pos_w.unsqueeze(1))
    return feet_pos_b.reshape(base_pos_w.shape[0], -1)

def key_links_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frame_transformer")):
    """Get the position of the key links in the base frame.

    Args:
        env (ManagerBasedEnv): The environment instance.
        asset_cfg (SceneEntityCfg, optional): The configuration for the asset. Defaults to SceneEntityCfg("frame_transformer").

    Returns:
        torch.Tensor: Position of the key links relative to source frame. Shape is (N, M*3), 
                    where N is the number of environments, and M is the number of target frames.
    """
    
    sensor: FrameTransformer = env.scene.sensors[asset_cfg.name]
    num_envs = env.scene.num_envs
    return sensor.data.target_pos_source.reshape(num_envs, -1)

