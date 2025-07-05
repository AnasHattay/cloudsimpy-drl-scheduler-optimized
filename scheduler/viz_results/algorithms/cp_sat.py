from collections import defaultdict

from scheduler.dataset_generator.core.models import Workflow, Vm, Task
from scheduler.dataset_generator.solvers.cp_sat_solver import solve_cp_sat
from scheduler.rl_model.core.types import VmAssignmentDto, TaskDto, VmDto
from scheduler.viz_results.algorithms.base import BaseScheduler


class CpSatScheduler(BaseScheduler):
    """
    Implementation of the CP-SAT scheduling algorithm.

    CP-SAT is a scheduling algorithm that uses constraint programming to solve the scheduling problem.
    """

    _is_optimal: bool | None = None
    _makespan: float | None = None

    def __init__(self, timeout: int = 10) -> None:
        super().__init__()
        self.timeout = timeout

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        self._is_optimal = None

        grouped_task_objs: defaultdict[int, list[Task]] = defaultdict(list)
        for task in tasks:
            grouped_task_objs[task.workflow_id].append(
                Task(
                    id=task.id,
                    workflow_id=task.workflow_id,
                    length=task.length,
                    req_memory_mb=task.req_memory_mb,
                    child_ids=task.child_ids,
                )
            )

        workflow_objs: list[Workflow] = []
        for workflow_id, task_objs in grouped_task_objs.items():
            workflow_objs.append(
                Workflow(
                    id=workflow_id,
                    tasks=task_objs,
                    arrival_time=0,  # Static scheduling (all tasks arrived t=0)
                )
            )

        vm_objs: list[Vm] = []
        for vm in vms:
            vm_objs.append(
                Vm(
                    id=vm.id,
                    host_id=0,  # Doesn't matter
                    cpu_speed_mips=int(vm.cpu_speed_mips),
                    memory_mb=vm.memory_mb,
                    disk_mb=500,  # Doesn't matter
                    bandwidth_mbps=500,  # Doesn't matter
                    vmm="Xen",  # Doesn't matter
                )
            )

        # We have to sort since we lose start time and the assignments are supposed to be in temporal order
        self._is_optimal, assignment_objs = solve_cp_sat(workflow_objs, vm_objs, timeout=self.timeout)
        assignment_objs = list(sorted(assignment_objs, key=lambda t: t.start_time))
        self._makespan = max([t.end_time for t in assignment_objs])

        assignments: list[VmAssignmentDto] = []
        for assignment_obj in assignment_objs:
            assignments.append(
                VmAssignmentDto(
                    vm_id=assignment_obj.vm_id,
                    workflow_id=assignment_obj.workflow_id,
                    task_id=assignment_obj.task_id,
                )
            )

        return assignments

    def is_optimal(self) -> bool:
        if self._is_optimal is None:
            raise Exception("Schedule the tasks first")
        return self._is_optimal
