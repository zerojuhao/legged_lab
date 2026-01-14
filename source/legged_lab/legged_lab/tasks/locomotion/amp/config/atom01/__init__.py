import gymnasium as gym

from . import agents

from legged_lab.envs import ManagerBasedAmpEnv

gym.register(
    id="LeggedLab-Isaac-AMP-Flat-Atom01-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_flat_env_cfg:Atom01AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Atom01FlatRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Flat-Atom01-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_flat_env_cfg:Atom01AmpFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Atom01FlatRslRlOnPolicyRunnerAmpCfg",
    },
)