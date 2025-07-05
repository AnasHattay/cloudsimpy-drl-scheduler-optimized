import torch
import numpy as np

from scheduler.dataset_generator.core.models import Dataset, Workflow
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto
from scheduler.viz_results.algorithms.base import BaseScheduler


class GinAgentScheduler(BaseScheduler):
    vm_completion_time: np.ndarray | None = None

    def __init__(self, model_path: str):
        self.model_path = model_path

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        agent = GinAgent(device=torch.device("cpu"))
        agent.load_state_dict(torch.load(str(self.model_path), weights_only=True))

        if self.vm_completion_time is None:
            self.vm_completion_time = np.zeros(len(vms))

        # Create a fake dataset to be used by the RL environment
        workflow_ds: dict[int, Workflow] = {}
        for task in tasks:
            if task.workflow_id not in workflow_ds:
                workflow_ds[task.workflow_id] = Workflow(id=task.workflow_id, tasks=[], arrival_time=0)
            workflow_ds[task.workflow_id].tasks.append(task.to_task())
        vm_ds = [vm.to_vm() for vm in vms]
        host_ds = [vm.to_host() for vm in vms]

        env = GinAgentWrapper(
            CloudSchedulingGymEnvironment(
                dataset=Dataset(
                    workflows=list(workflow_ds.values()),
                    vms=vm_ds,
                    hosts=host_ds,
                )
            )
        )

        # Run environment
        obs, info = env.reset(seed=0)
        while True:
            tensor_obs = torch.Tensor(obs).reshape(1, -1)
            action, *_ = agent.get_action_and_value(tensor_obs)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                break

        assert len(tasks) == len(info["assignments"])
        return info["assignments"]
