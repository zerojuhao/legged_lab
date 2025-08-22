from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class ManagerBasedAmpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a AMP environment with the manager-based workflow."""
    
    motion_data: object = MISSING
    """Motion data configuration for the AMP environment.
    
    Please refer to the :class:`legged_lab.managers.MotionDataManager` class for more details.
    """