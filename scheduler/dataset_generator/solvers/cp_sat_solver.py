import collections

from typing import Any
from ortools.sat.python import cp_model

from scheduler.dataset_generator.core.models import Workflow, Vm, VmAssignment, Task


def solve_cp_sat(
    workflows: list[Workflow], vms: list[Vm], timeout: int | None = None
) -> tuple[bool, list[VmAssignment]]:
    """
    Solve the VM assignment problem using CP-SAT solver.

    Returns a tuple of (isOptimal, assignments).
    """

    def is_assignable(task: Task, vm: Vm):
        return vm.memory_mb >= task.req_memory_mb

    def runtime(task_length: int, vm_speed: int):
        return int(task_length / vm_speed)

    model = cp_model.CpModel()

    # Horizon is the maximum possible end time of any task.
    horizon = sum(
        runtime(task.length, min(vm.cpu_speed_mips for vm in vms if is_assignable(task, vm)))
        for workflow in workflows
        for task in workflow.tasks
    )

    task_start_vars: dict[tuple[int, int], Any] = {}
    task_end_vars: dict[tuple[int, int], Any] = {}
    vm_assigned_intervals: dict[int, list[Any]] = collections.defaultdict(list)
    task_assignment_vars: dict[tuple[int, int], dict[int, Any]] = collections.defaultdict(dict)
    for workflow in workflows:
        for task in workflow.tasks:
            task_key = (workflow.id, task.id)

            # Start time and end time. (0 <= start <= horizon and 0 <= end <= horizon)
            task_start_var = model.new_int_var(0, horizon, f"start|{task.id}")
            end_var = model.new_int_var(0, horizon, f"end|{task.id}")
            task_start_vars[task_key] = task_start_var
            task_end_vars[task_key] = end_var

            assignable_vms = [vm for vm in vms if is_assignable(task, vm)]

            # Duration a task takes to complete. (length/min_speed <= duration <= length/max_speed)
            min_speed = min(vm.cpu_speed_mips for vm in assignable_vms)
            max_speed = max(vm.cpu_speed_mips for vm in assignable_vms)
            min_duration = runtime(task.length, max_speed)
            max_duration = runtime(task.length, min_speed)
            duration_var = model.new_int_var(min_duration, max_duration, f"duration|{task.id}")

            # Constraint between start, end and duration. (start + duration = end)
            model.new_interval_var(task_start_var, duration_var, end_var, f"interval|{task.id}")

            for vm in assignable_vms:
                # 0 <= l_start <= horizon and 0 <= l_end <= horizon
                vm_start_var = model.new_int_var(0, horizon, f"start|{task.id}|{vm.id}")
                vm_end_var = model.new_int_var(0, horizon, f"end|{task.id}|{vm.id}")

                # l_start + l_duration = l_end only if assigned
                vm_duration = runtime(task.length, vm.cpu_speed_mips)
                assigned_var = model.new_bool_var(f"assigned|{task.id}|{vm.id}")
                vm_interval = model.new_optional_interval_var(
                    vm_start_var, vm_duration, vm_end_var, assigned_var, f"interval|{task.id}|{vm.id}"
                )
                vm_assigned_intervals[vm.id].append(vm_interval)

                # (start, duration, end) = (l_start, l_duration, l_end) if assigned
                model.add(task_start_var == vm_start_var).only_enforce_if(assigned_var)
                model.add(duration_var == vm_duration).only_enforce_if(assigned_var)
                model.add(end_var == vm_end_var).only_enforce_if(assigned_var)
                task_assignment_vars[task_key][vm.id] = assigned_var

            # Exactly one VM assignment (No preemption)
            model.add_exactly_one([task_assignment_vars[task_key][vm.id] for vm in assignable_vms])

    # Precedence constraints. (end of task <= start of child task)
    for workflow in workflows:
        for task in workflow.tasks:
            task_end_var = task_end_vars[(workflow.id, task.id)]
            for child_id in task.child_ids:
                child_start_var = task_start_vars[(workflow.id, child_id)]
                model.add(task_end_var <= child_start_var)

    # Machine constraints
    for vm_intervals in vm_assigned_intervals.values():
        if len(vm_intervals) > 1:
            model.add_no_overlap(vm_intervals)

    # Makespan objective
    workflow_end_vars = []
    for workflow in workflows:
        end_task_id_set = set(task.id for task in workflow.tasks if not task.child_ids)
        for end_task_id in end_task_id_set:
            workflow_end_vars.append(task_end_vars[(workflow.id, end_task_id)])
    makespan_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan_var, workflow_end_vars)
    model.minimize(makespan_var)

    solver = cp_model.CpSolver()
    # Reproducibility: https://groups.google.com/g/or-tools-discuss/c/lPb1FzhTMt0
    solver.parameters.interleave_search = True
    solver.parameters.num_search_workers = 16
    solver.parameters.share_binary_clauses = False
    if timeout is not None:
        solver.parameters.max_time_in_seconds = timeout

    status = solver.Solve(model)
    is_optimal = status == cp_model.OPTIMAL

    assigned_tasks: list[VmAssignment] = []
    for workflow in workflows:
        for task in workflow.tasks:
            for vm in vms:
                if not is_assignable(task, vm):
                    continue
                if not solver.Value(task_assignment_vars[(workflow.id, task.id)][vm.id]):
                    continue
                start_time = solver.Value(task_start_vars[(workflow.id, task.id)])
                end_time = solver.Value(task_end_vars[(workflow.id, task.id)])
                assigned_tasks.append(VmAssignment(workflow.id, task.id, vm.id, start_time, end_time))
                break
            else:
                raise Exception("Task not assigned to any VM.")

    return is_optimal, assigned_tasks
