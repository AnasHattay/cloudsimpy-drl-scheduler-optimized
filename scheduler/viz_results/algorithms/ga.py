from typing import Any

import numpy as np
from icecream import ic
from pygad import pygad

from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto
from scheduler.rl_model.core.utils.helpers import is_suitable
from scheduler.viz_results.algorithms.base import BaseScheduler


class GAScheduler(BaseScheduler):
    tasks: list[TaskDto]
    vms: list[VmDto]
    task_vm_time_cost_idx: np.ndarray
    dependencies_idx: dict[int, list[int]]

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        self.tasks = tasks
        self.vms = vms

        self.task_vm_time_cost_idx = np.zeros((len(tasks), len(vms)))
        for task_index, task in enumerate(tasks):
            for vm_index, vm in enumerate(vms):
                self.task_vm_time_cost_idx[task_index, vm_index] = (
                    (task.length / vm.cpu_speed_mips) if is_suitable(vm, task) else np.inf
                )

        self.dependencies_idx: dict[int, list[int]] = {}
        for child_index, child_task in enumerate(tasks):
            self.dependencies_idx[child_index] = []
            for parent_index, parent_task in enumerate(tasks):
                if child_task.id in parent_task.child_ids:
                    self.dependencies_idx[child_index].append(parent_index)

        ga_instance = pygad.GA(
            num_generations=100,
            num_parents_mating=400,
            sol_per_pop=500,
            num_genes=len(tasks),
            gene_type=int,
            init_range_low=0,
            init_range_high=len(vms) - 1,
            parent_selection_type="tournament",
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            fitness_func=lambda ga, sol, idx: self.fitness(sol),
            # on_generation=lambda x: ic(x.best_solution()[1]),
        )

        ga_instance.run()
        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        ic(best_solution, best_solution_fitness)

        vm_assignments = []
        for task_index, vm_index in enumerate(best_solution):
            vm_assignments.append(
                VmAssignmentDto(
                    vm_id=vms[vm_index].id,
                    workflow_id=tasks[task_index].workflow_id,
                    task_id=tasks[task_index].id,
                )
            )
        return vm_assignments

    def fitness(self, solution: Any) -> float:
        task_to_vm_mapping = solution.astype(int)

        # Initialize VM completion times and task completion times
        vm_completion_times_idx = np.zeros(len(self.vms))
        task_completion_times_idx = np.zeros(len(self.tasks))

        # Ensure all parent tasks are completed before scheduling this task
        for task_index, vm_index in enumerate(task_to_vm_mapping):
            dep_indices = self.dependencies_idx[task_index]
            parent_completion_time = task_completion_times_idx[dep_indices].max() if len(dep_indices) > 0 else 0

            # Task start time is the later of the parent's completion and the VM's availability
            start_time = max(vm_completion_times_idx[vm_index], parent_completion_time)
            execution_time = self.task_vm_time_cost_idx[task_index, vm_index]
            task_completion_times_idx[task_index] = start_time + execution_time
            vm_completion_times_idx[vm_index] = task_completion_times_idx[task_index]

        makespan = task_completion_times_idx.max()
        return -makespan
