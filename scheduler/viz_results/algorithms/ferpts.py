from scheduler.rl_model.core.types import VmAssignmentDto, TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable
from scheduler.rl_model.core.utils.task_mapper import TaskMapper
from scheduler.viz_results.algorithms.base import BaseScheduler


class FerptsScheduler(BaseScheduler):
    """Implementation of the FERPTS (Fast and Energy-Aware Resource Provisioning and Task Scheduling) algorithm."""

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        mapper = TaskMapper(tasks)
        mapped_tasks = mapper.map_tasks()

        vm_ready_times = {vm.id: 0.0 for vm in vms}
        task_completion_times = {}
        assignments = []

        for m_task in mapped_tasks:
            best_vm, min_cost = None, float("inf")

            for vm in vms:
                if not is_suitable(vm, m_task):
                    continue

                ready_time = vm_ready_times[vm.id]
                parent_completion_times = [
                    task_completion_times[parent_task.id]
                    for parent_task in mapped_tasks
                    if m_task.id in parent_task.child_ids
                ]
                start_time = max(ready_time, max(parent_completion_times, default=0.0))
                finish_time = start_time + (m_task.length / vm.cpu_speed_mips)

                # Calculate energy cost based on runtime and power usage
                energy_cost = m_task.length * active_energy_consumption_per_mi(vm)

                # FERPTS minimizes both runtime and energy cost
                combined_cost = finish_time + energy_cost

                if combined_cost < min_cost:
                    best_vm = vm
                    min_cost = combined_cost

            if best_vm is None:
                raise Exception(f"Task {m_task.id} could not be scheduled on any VM.")

            vm_ready_times[best_vm.id] = min_cost
            task_completion_times[m_task.id] = min_cost
            if m_task.id != mapper.dummy_start_task_id() and m_task.id != mapper.dummy_end_task_id():
                u_workflow_id, u_task_id = mapper.unmap_id(m_task.id)
                assignments.append(VmAssignmentDto(vm_id=best_vm.id, workflow_id=u_workflow_id, task_id=u_task_id))

        return assignments
