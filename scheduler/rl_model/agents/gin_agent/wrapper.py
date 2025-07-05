from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi


class GinAgentWrapper(gym.Wrapper):
    observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float32)
    action_space = gym.spaces.Discrete(MAX_OBS_SIZE)

    prev_obs: EnvObservation
    initial_obs: EnvObservation

    def __init__(self, env: gym.Env[np.ndarray, int]):
        super().__init__(env)
        self.mapper = GinAgentMapper(MAX_OBS_SIZE)
        self.heuristic=False
        self.phase = "ACTIVE_ONLY"
        self.validation_interval = 1000
        self.global_step=0

    def set_global_step(self, step: int):
        self.global_step = step

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        assert isinstance(obs, EnvObservation)
        mapped_obs = self.map_observation(obs)
        self.prev_obs = obs
        self.initial_obs = obs
        return mapped_obs, info

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:

        mapped_action = self.map_action(action)
        obs, _, terminated, truncated, info = super().step(mapped_action)
        assert isinstance(obs, EnvObservation)
        mapped_obs = self.map_observation(obs)

        makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / obs.makespan()
        energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / obs.energy_consumption()
        reward = energy_reward
        # Sparse reward: Idle energy + makespan (only at episode end)
        # if terminated or truncated:
        #     idle_energy = obs.idle_energy()
        #     sparse_reward = - (idle_energy) / 1e8  # 0.5 is tunable
        #     info.update({
        #         "energy/idle": idle_energy,
        #         "energy/total": obs.total_energy_consumption()
        #     })

        #
        # else:
        #     sparse_reward = 0.0
        # Combine rewards (or use sparse_reward alone if preferred)



        current_idle_energy=-(obs.idle_energy()-self.prev_obs.idle_energy()/ obs.idle_energy())
        #
        # # print(f"current_idle_energy is {current_idle_energy}")
        # #
        # print(f"energy_reward is {energy_reward}")
        #
        # if self.phase == "ACTIVE_ONLY":
        #     print(f'Active only phase, global step is {self.global_step}')
        #     if self.global_step % self.validation_interval == 0:
        #         if self.validate_vs_heuristic(obs):
        #             self.phase = "FULL_OPTIMIZATION"
        #
        # else:  # FULL_OPTIMIZATION phase
        #     print(f"-----------Converged to heuristic-------------- ")
        #     print(f"energy_reward is {energy_reward}")
        #     #
        #     print(f"makespan_reward is {makespan_reward}")
        #     reward =energy_reward+makespan_reward

        reward= energy_reward + makespan_reward
        if terminated or truncated:
            idle_energy = obs.idle_energy()
            idle_reward = - idle_energy/ 1e8
            info.update({
                "energy/idle": idle_energy,
                "energy/total": obs.total_energy_consumption(),

            })
            print(f'energy_reward is {energy_reward}')
            print(f'makespan_reward is {makespan_reward}')
            print(f'idle_reward is {idle_reward}')

            reward= energy_reward + makespan_reward+idle_reward


        self.prev_obs = obs
        return mapped_obs, reward, terminated, truncated, info
    def validate_vs_heuristic(self,obs):
        # Run 10 episodes with heuristic
        # Compare makespan/energy to agent
        return obs.energy_consumption() >= 0.95 * self.prev_obs.energy_consumption()
    def map_action(self, action: int) -> EnvAction:
        vm_count = len(self.prev_obs.vm_observations)
        return EnvAction(task_id=int(action // vm_count), vm_id=int(action % vm_count))

    def map_observation(self, observation: EnvObservation) -> np.ndarray:
        # Task observations
        task_state_scheduled = np.array([task.assigned_vm_id is not None for task in observation.task_observations])
        task_state_ready = np.array([task.is_ready for task in observation.task_observations])
        task_length = np.array([task.length for task in observation.task_observations])

        # VM observations
        vm_speed = np.array([vm.cpu_speed_mips for vm in observation.vm_observations])
        vm_energy_rate = np.array([active_energy_consumption_per_mi(vm) for vm in observation.vm_observations])
        vm_idle = np.array([vm.host_power_idle_watt for vm in observation.vm_observations])

        vm_completion_time = np.array([vm.completion_time for vm in observation.vm_observations])

        # Task-Task observations
        task_dependencies = np.array(observation.task_dependencies).T

        # Task-VM observations
        compatibilities = np.array(observation.compatibilities).T

        # Task completion times
        task_completion_time = observation.task_completion_time()
        assert task_completion_time is not None

        return self.mapper.map(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,
            task_completion_time=task_completion_time,
            vm_speed=vm_speed,vm_idle=vm_idle,
            vm_energy_rate=vm_energy_rate,
            vm_completion_time=vm_completion_time,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )
