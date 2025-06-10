import torch
from typing import Optional
import numpy as np


def quat_slerp(
    q0: torch.Tensor,
    *,
    q1: Optional[torch.Tensor] = None,
    blend: Optional[torch.Tensor] = None,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Interpolation between consecutive rotations (Spherical Linear Interpolation).

    Args:
        q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        blend: Interpolation coefficient between 0 (q0) and 1 (q1). Shape is (N,) or (N, M).
        start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
        end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

    Returns:
        Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
    """
    if start is not None and end is not None:
        return quat_slerp(q0=q0[start], q1=q0[end], blend=blend)
    if q0.ndim >= 2:
        blend = blend.unsqueeze(-1) # type: ignore
    if q0.ndim >= 3:
        blend = blend.unsqueeze(-1) # type: ignore

    qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
    cos_half_theta = (
        q0[..., qw] * q1[..., qw]
        + q0[..., qx] * q1[..., qx]
        + q0[..., qy] * q1[..., qy]
        + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.clone() # type: ignore
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
    new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
    new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
    new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q

if __name__ == "__main__":
    import isaaclab.utils.math as math_utils
    import time
    
    N = 1024
    q0 = torch.rand(N, 4, device="cuda:0")
    q1 = torch.rand(N, 4, device="cuda:0")
    blend = torch.rand(N, device="cuda:0")
    
    q0 = math_utils.quat_unique(q0)
    q0 = math_utils.normalize(q0)
    q1 = math_utils.quat_unique(q1)
    q1 = math_utils.normalize(q1)
    
    start_time = time.time()
    
    q = quat_slerp(
        q0=q0,
        q1=q1,
        blend=blend,
    )
    
    end_time = time.time()
    
    print(f"Time taken for quaternion slerp: {end_time - start_time:.6f} seconds")
    print(q.shape)