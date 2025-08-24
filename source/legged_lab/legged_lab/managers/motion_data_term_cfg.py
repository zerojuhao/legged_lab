from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass

@configclass 
class MotionDataTermCfg:
    """
    Configuration for the motion data term in the motion data manager.
    """
    
    weight: float = 1.0
    """Weight of this term in the motion data manager."""
    
    motion_data_dir: str = MISSING
    """Directory containing motion data files.
    
    Only supports reading .pkl files from this directory.
    """
    
    motion_data_weight: dict[str, float] = MISSING
    """Weights for the motion data in this term."""
    
    dof_names: list[str] = MISSING
    """Names of the dof in the motion data. The order should match the order in the motion data files."""

    key_links_mapping: dict[str, str] = MISSING
    """Mapping from moton data key links to lab key links.
    
    It is common that the key links names are different in the motion data and the lab
    - the keys are the names of the links in the motion dataset
    - the values are the names of the links in lab 
    """
    
    num_steps: int = 2
    """Number of steps given by mini-batch generator."""
