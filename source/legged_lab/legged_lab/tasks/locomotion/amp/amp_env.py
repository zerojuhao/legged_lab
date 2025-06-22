from __future__ import annotations

import torch
from typing import Any
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
from isaaclab.managers import ActionManager, ObservationManager, RecorderManager, CommandManager, CurriculumManager, RewardManager, TerminationManager

from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

from legged_lab.tasks.locomotion.amp.utils_amp import MotionLoader

class AmpObservationManager(ObservationManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg=cfg, env=env)
    
    def compute_group(self, group_name: str, update_buffer: bool = True) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.
        
        This function is almost identical to its parent class, except an addtional flag to
        indicate whether to update the buffer or not. This is useful for AMP learning, where
        we need to compute the observations for the group without updating the buffer.
        """
        # check if group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if term_cfg.noise:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                if update_buffer:
                    self._group_obs_term_history_buffer[group_name][term_name].append(obs)
                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = self._group_obs_term_history_buffer[group_name][term_name].buffer.reshape(
                        self._env.num_envs, -1
                    )
                else:
                    group_obs[term_name] = self._group_obs_term_history_buffer[group_name][term_name].buffer
            else:
                group_obs[term_name] = obs

        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=-1)
        else:
            return group_obs

class AmpEnv(ManagerBasedRLEnv):
    """AMP Environment for locomotion tasks.

    This class inherits from the `ManagerBasedRLEnv` class and is used to create an environment for
    training and testing reinforcement learning agents on locomotion tasks using the AMP.
    
    In the original ManagerBasedRLEnv's `step` method, observations are lost if the environments are 
    reset. But in AMP we should record the observations before resetting the environments.
    This class overrides the `step` method to ensure that observations are retained even when
    environments are reset.
    """
    
    cfg: LocomotionAmpEnvCfg
    
    def __init__(self, cfg: LocomotionAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        
        # load the motion loader
        self.motion_loader: MotionLoader = MotionLoader(
            cfg=cfg.motion_loader,
            env=self,
            device=self.device
        )

    def load_managers(self):
        """Load the managers for the environment.

        This function is almost identical to the parent class, except that it uses the custom
        `AmpObservationManager` class for the observation manager. This is necessary to ensure that
        the observations are retained even when environments are reset.
        """
        # prepare the managers
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)
        
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = AmpObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
            
        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
            

    def _get_amp_observations(self) -> torch.Tensor:
        """Get the AMP observations.

        This function retrieves the AMP observations from the observation manager and returns them.

        Returns:
            The AMP observations as a tensor.
        """
        amp_obs_dict = self.observation_manager.compute_group("amp", update_buffer=False)
        return amp_obs_dict
        
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # -- update AMP observations
        amp_obs = self._get_amp_observations()

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()
        if len(reset_env_ids) > 0:
            for term in amp_obs.keys():
                # reset the observations for the reset envs
                self.obs_buf["amp"][term][reset_env_ids] = amp_obs[term][reset_env_ids]
        
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras



