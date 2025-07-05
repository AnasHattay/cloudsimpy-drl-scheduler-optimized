import torch
from icecream import ic

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment


def main():
    mapper = GinAgentMapper(MAX_OBS_SIZE)
    agent = GinAgent(torch.device("cpu"))
    env = GinAgentWrapper(
        CloudSchedulingGymEnvironment(
            dataset_args=DatasetArgs(
                host_count=1,
                vm_count=2,
                workflow_count=1,
                gnp_min_n=5,
                gnp_max_n=5,
                max_memory_gb=10,
                min_cpu_speed=500,
                max_cpu_speed=5000,
                min_task_length=500,
                max_task_length=100_000,
                task_arrival="static",
                dag_method="gnp",
            )
        )
    )
    obs, info = env.reset(seed=0)
    tensor_obs = torch.Tensor(obs)
    ic(mapper.unmap(tensor_obs))
    while True:
        action, *_ = agent.get_action_and_value(tensor_obs.reshape(1, -1))
        ic(action)
        obs, reward, terminated, truncated, info = env.step(action)
        tensor_obs = torch.Tensor(obs)
        ic(mapper.unmap(tensor_obs), reward)
        if terminated or truncated:
            ic(info)
            break


if __name__ == "__main__":
    main()
