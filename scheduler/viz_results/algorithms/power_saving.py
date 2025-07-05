from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable
from scheduler.viz_results.algorithms.base_ready_queue import BaseReadyQueueScheduler


class PowerSavingScheduler(BaseReadyQueueScheduler):
    """
    Implementation of the Power Saving scheduling algorithm.

    Power Saving is a simple scheduling algorithm that schedules the task on the VM with the lowest power consumption per processed MIP.
    This is guaranteed to be the most energy-efficient scheduling algorithm, but it may not be the fastest.
    """

    def select_task_and_vm(self, ready_tasks: list[TaskDto], vms: list[VmDto]) -> tuple[TaskDto, VmDto]:
        best_task = None
        best_vm = None
        best_energy_cons = float("inf")
        for task in ready_tasks:
            for vm in vms:
                if not is_suitable(vm, task):
                    continue
                energy_cons = active_energy_consumption_per_mi(vm) * task.length
                if energy_cons < best_energy_cons:
                    best_energy_cons = energy_cons
                    best_task = task
                    best_vm = vm
        assert best_task is not None
        assert best_vm is not None

        return best_task, best_vm

    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        return ready_tasks[0]

    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        return min(vms, key=active_energy_consumption_per_mi)
