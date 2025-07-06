from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    

def key_links_pos_b(env: ManagerBasedEnv, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                    local_pos_dict: dict[str, float] = {}) -> torch.Tensor:
    """Get the position of the key links in the base frame.

    Args:
        env (ManagerBasedEnv): The environment instance.
        asset_cfg (SceneEntityCfg, optional): The configuration for the asset. Defaults to SceneEntityCfg("robot").
        local_pos (dict[str, float], optional): Position of the key links in their parent frame.

    Returns:
        torch.Tensor: Position of the key links in the base frame. Shape is (N, M*3), 
                    where N is the number of environments, and M is the number of key links.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    body_names = asset.data.body_names

    indices, names, local_pos = string_utils.resolve_matching_names_values(
        data=local_pos_dict,
        list_of_strings=body_names,
        preserve_order=False
    )
    local_pos = torch.tensor(
        local_pos,
        dtype=torch.float32,
        device=asset.data.root_pos_w.device
    ).unsqueeze(0)  # shape: (1, M, 3)
    local_pos = local_pos.expand(asset.data.root_pos_w.shape[0], -1, -1)  # shape: (num_instances, M, 3)
    
    # get the pose of the root in world frame
    base_pos_w = asset.data.root_pos_w      # shape: (num_instances, 3).
    base_quat_w = asset.data.root_quat_w    # shape: (num_instances, 4), w, x, y, z order.
    # get the positions of the key links in world frame
    parent_pos_w = asset.data.body_pos_w[:, indices, :] # shape: (num_instances, M, 3)
    parent_quat = asset.data.body_quat_w[:, indices, :] # shape: (num_instances, M, 4)
    key_links_pos_w = parent_pos_w + math_utils.quat_apply(parent_quat, local_pos)
    # get the positions of the key links in base frame
    # expand base_quat_w and base_pos_w to match key_links_pos_w dimensions
    num_key_links = key_links_pos_w.shape[1]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_key_links, -1)  # shape: (num_instances, M, 4)
    base_pos_w_expanded = base_pos_w.unsqueeze(1).expand(-1, num_key_links, -1)    # shape: (num_instances, M, 3)
    key_links_pos_b = math_utils.quat_apply_inverse(base_quat_w_expanded, key_links_pos_w - base_pos_w_expanded)
    return key_links_pos_b.reshape(base_pos_w.shape[0], -1)
