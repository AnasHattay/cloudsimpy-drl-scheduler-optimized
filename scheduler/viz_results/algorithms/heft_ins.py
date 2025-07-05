from collections import defaultdict

import networkx as nx
from heft import heft
import numpy as np

from scheduler.rl_model.core.types import VmAssignmentDto, TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import is_suitable
from scheduler.viz_results.algorithms.base import BaseScheduler

ScheduleType = dict[int, list[heft.ScheduleEvent]]


class InsertionHeftScheduler(BaseScheduler):
    """
    Implementation of the HEFT scheduling algorithm.

    HEFT is a scheduling algorithm that uses a combination of task-level and workflow-level scheduling.
    Following implementation uses library: https://github.com/mackncheesiest/heft
    """

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        # Group tasks by workflow
        grouped_tasks: defaultdict[int, list[TaskDto]] = defaultdict(list)
        for task in tasks:
            grouped_tasks[task.workflow_id].append(task)

        # Schedule each workflow
        schedule: ScheduleType | None = None
        assignments: list[tuple[float, VmAssignmentDto]] = []
        assigning_task_start_id = 0
        for workflow_id, task_list in grouped_tasks.items():
            schedule = self.schedule_workflow(task_list, vms, schedule)
            # Convert the schedule to a list of assignments
            for vm_id, events in schedule.items():
                for event in events:
                    actual_task_id = event.task - assigning_task_start_id
                    # We only care about tasks from the current workflow (events has old workflows + dummy tasks)
                    if 0 <= actual_task_id < len(task_list):
                        assignments.append((event.start, VmAssignmentDto(vm_id, workflow_id, actual_task_id)))
            assigning_task_start_id += len(task_list) + 1

        # Sort assignments by start time (make sure the order is correct)
        assignments.sort(key=lambda x: x[0])
        return [assignment for _, assignment in assignments]

    @staticmethod
    def schedule_workflow(tasks: list[TaskDto], vms: list[VmDto], schedule: ScheduleType | None) -> ScheduleType:
        total_tasks = len(tasks) + 1  # Add a dummy task to represent the end of the workflow
        total_vms = len(vms)
        dummy_task_id = total_tasks - 1

        # Computational cost between tasks and vms (+ dummy task)
        comp_matrix = np.zeros((total_tasks, total_vms))
        for i in range(total_tasks - 1):  # Exclude the dummy task
            for j in range(total_vms):
                if not is_suitable(vms[j], tasks[i]):
                    comp_matrix[i, j] = np.inf
                else:
                    comp_matrix[i, j] = tasks[i].length / vms[j].cpu_speed_mips
        # Communication cost between tasks - 0 if tasks are on the same VM, 1 otherwise
        comm_matrix = 1 - np.eye(total_vms)
        # Communication startup for VMs - 0 for all VMs
        comm_startup = np.zeros(total_vms)

        dag: nx.DiGraph = nx.DiGraph()
        dag.add_node(dummy_task_id)  # Add a dummy node to represent the end of the workflow

        for task in tasks:
            dag.add_node(task.id)
            if not task.child_ids:
                dag.add_edge(task.id, dummy_task_id, weight=1)
                continue
            for child_id in task.child_ids:
                dag.add_edge(task.id, child_id, weight=1)

        schedule, _, _ = heft.schedule_dag(dag, comp_matrix, comm_matrix, comm_startup, proc_schedules=schedule)
        assert schedule is not None, "HEFT scheduling failed"
        return schedule
