from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training
    """
    
    replay_buffer_size: int = 1000
    """Size of the replay buffer for storing AMP observations"""
    
    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""
    
    amp_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""
    
    amp_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""
    
    amp_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""
    
    amp_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        amp_reward_scale: float = 1.0
        """Scale for the AMP reward in the training"""
        
        task_reward_lerp: float = 0.0
        """Linear interpolation factor for the task reward in the AMP training."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""

    motion_dataset: str = MISSING
    """Name of the motion data term to use in AMP training."""
