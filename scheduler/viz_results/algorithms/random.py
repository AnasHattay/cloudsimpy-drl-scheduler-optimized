import random
from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import is_suitable
from scheduler.viz_results.algorithms.base_ready_queue import BaseReadyQueueScheduler


class RandomScheduler(BaseReadyQueueScheduler):
    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        """Choose the next task (with no preference)."""
        return ready_tasks[random.randint(0, len(ready_tasks) - 1)]

    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        """Schedule the task on the next VM in the list."""
        while True:
            vm_index = random.randint(0, len(vms) - 1)
            if is_suitable(vms[vm_index], task):
                return vms[vm_index]
