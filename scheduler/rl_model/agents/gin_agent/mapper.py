from dataclasses import dataclass

import numpy as np
import torch


class GinAgentMapper:
    def __init__(self, obs_size: int):
        self.obs_size = obs_size

    def map(
        self,
        task_state_scheduled: np.ndarray,
        task_state_ready: np.ndarray,
        task_length: np.ndarray,
        task_completion_time: np.ndarray,
        vm_speed: np.ndarray,
        vm_energy_rate: np.ndarray,vm_idle: np.ndarray,
        vm_completion_time: np.ndarray,
        task_dependencies: np.ndarray,
        compatibilities: np.ndarray,
    ) -> np.ndarray:
        num_tasks = task_completion_time.shape[0]
        num_vms = vm_completion_time.shape[0]
        num_task_deps = task_dependencies.shape[1]
        num_compatibilities = compatibilities.shape[1]

        arr = np.concatenate(
            [
                np.array([num_tasks, num_vms, num_task_deps, num_compatibilities], dtype=np.int32),  # Header
                np.array(task_state_scheduled, dtype=np.int32),  # num_tasks
                np.array(task_state_ready, dtype=np.int32),  # num_tasks
                np.array(task_length, dtype=np.float64),  # num_tasks
                np.array(task_completion_time, dtype=np.float64),  # num_tasks
                np.array(vm_speed, dtype=np.float64),  # num_vms
                np.array(vm_energy_rate, dtype=np.float64),  # num_vms
                np.array(vm_idle, dtype=np.float64),  # num_vms
                np.array(vm_completion_time, dtype=np.float64),  # num_vms
                np.array(task_dependencies.flatten(), dtype=np.int32),  # num_task_deps*2
                np.array(compatibilities.flatten(), dtype=np.int32),  # num_compatibilities*2
            ]
        )

        assert len(arr) <= self.obs_size, "Observation size does not fit the buffer, please adjust the size of mapper"
        arr = np.pad(arr, (0, self.obs_size - len(arr)), "constant")

        return arr

    def unmap(self, tensor: torch.Tensor) -> "GinAgentObsTensor":
        assert len(tensor) == self.obs_size, "Tensor size is not of expected size"

        num_tasks = int(tensor[0].long().item())
        num_vms = int(tensor[1].long().item())
        num_task_deps = int(tensor[2].long().item())
        num_compatibilities = int(tensor[3].long().item())
        tensor = tensor[4:]

        task_state_scheduled = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_state_ready = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_length = tensor[:num_tasks]
        tensor = tensor[num_tasks:]
        task_completion_time = tensor[:num_tasks]
        tensor = tensor[num_tasks:]

        vm_speed = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_energy_rate = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_idle = tensor[:num_vms]
        tensor = tensor[num_vms:]

        vm_completion_time = tensor[:num_vms]
        tensor = tensor[num_vms:]

        task_dependencies = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
        tensor = tensor[num_task_deps * 2 :]
        compatibilities = tensor[: num_compatibilities * 2].reshape(2, num_compatibilities).long()
        tensor = tensor[num_compatibilities * 2 :]

        assert not tensor.any(), "There are non-zero elements in the padding"

        return GinAgentObsTensor(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,
            task_completion_time=task_completion_time,
            vm_speed=vm_speed,
            vm_energy_rate=vm_energy_rate,
            vm_idle=vm_idle,
            vm_completion_time=vm_completion_time,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )


@dataclass
class GinAgentObsTensor:
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_length: torch.Tensor
    task_completion_time: torch.Tensor
    vm_speed: torch.Tensor
    vm_energy_rate: torch.Tensor
    vm_idle: torch.Tensor
    vm_completion_time: torch.Tensor
    task_dependencies: torch.Tensor
    compatibilities: torch.Tensor
