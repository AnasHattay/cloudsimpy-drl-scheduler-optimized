from scheduler.rl_model.core.types import VmAssignmentDto, TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import is_suitable
from scheduler.rl_model.core.utils.task_mapper import TaskMapper
from scheduler.viz_results.algorithms.base import BaseScheduler


class HeftScheduler(BaseScheduler):
    """Implementation of the HEFT (Heterogeneous Earliest Finish Time) algorithm."""

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        mapper = TaskMapper(tasks)
        mapped_tasks = mapper.map_tasks()

        # Compute task priorities based on upward rank
        task_rank = self.compute_task_priorities(mapped_tasks, vms)
        sorted_m_tasks = sorted(mapped_tasks, key=lambda t: task_rank[t.id], reverse=True)

        vm_ready_times = {vm.id: 0.0 for vm in vms}
        task_completion_times = {}
        assignments = []
        for m_task in sorted_m_tasks:
            best_vm, earliest_finish_time = None, float("inf")

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
                if finish_time < earliest_finish_time:
                    best_vm = vm
                    earliest_finish_time = finish_time

            if best_vm is None:
                raise Exception(f"Task {m_task.id} could not be scheduled on any VM.")

            vm_ready_times[best_vm.id] = earliest_finish_time
            task_completion_times[m_task.id] = earliest_finish_time
            if m_task.id != mapper.dummy_start_task_id() and m_task.id != mapper.dummy_end_task_id():
                u_workflow_id, u_task_id = mapper.unmap_id(m_task.id)
                assignments.append(VmAssignmentDto(vm_id=best_vm.id, workflow_id=u_workflow_id, task_id=u_task_id))

        return assignments

    @staticmethod
    def compute_task_priorities(tasks: list[TaskDto], vms: list[VmDto]) -> dict:
        """Compute task priorities based on upward rank."""

        average_vm_speed = sum(vm.cpu_speed_mips for vm in vms) / len(vms)
        task_rank = {}

        def compute_upward_rank(task: TaskDto):
            if task.id in task_rank:
                return task_rank[task.id]

            child_ranks = [compute_upward_rank(child_task) for child_task in tasks if child_task.id in task.child_ids]
            child_rank_max = max(child_ranks, default=0)
            task_rank[task.id] = (task.length / average_vm_speed) + child_rank_max
            return task_rank[task.id]

        for _task in tasks:
            compute_upward_rank(_task)

        return task_rank
