from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import is_suitable
from scheduler.viz_results.algorithms.base_ready_queue import BaseReadyQueueScheduler


class MinMinScheduler(BaseReadyQueueScheduler):
    """
    Implementation of the MinMin scheduling algorithm.

    MinMin is a simple scheduling algorithm that schedules the task with the smallest length
    on the VM that will complete the task the fastest.
    """

    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        """Choose the task with the smallest length."""
        smallest_task = None
        smallest_task_length = float("inf")
        for task in ready_tasks:
            if task.length < smallest_task_length:
                smallest_task_length = task.length
                smallest_task = task
        assert smallest_task is not None

        return smallest_task

    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        """Schedule the task on the VM that will complete the task the fastest."""
        assert self.est_vm_completion_times is not None
        assert self.est_task_min_start_times is not None

        # Select the best VM by comparing the completion times
        best_vm = None
        best_vm_completion_time = float("inf")
        for vm in vms:
            if not is_suitable(vm, task):
                continue

            completion_time = (
                max(self.est_vm_completion_times[self.vid(vm)], self.est_task_min_start_times[self.tid(task)])
                + task.length / vm.cpu_speed_mips
            )
            if best_vm_completion_time > completion_time:
                best_vm = vm
                best_vm_completion_time = completion_time

        if best_vm is None:
            raise Exception("No VM found for task")

        return best_vm
