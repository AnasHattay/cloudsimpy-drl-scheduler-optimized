import copy
import random
from typing import Any, Callable

import gymnasium as gym

from scheduler.config.settings import MAX_TRAINING_DS_SEED
from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.dataset_generator.core.models import Dataset
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation

from scheduler.rl_model.core.env.state import EnvState, TaskState, VmState, StaticState
from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi, is_suitable
from scheduler.rl_model.core.utils.task_mapper import TaskMapper


global_reset_counter = 0


class CloudSchedulingGymEnvironment(gym.Env):
    dataset_generator: Callable[[int | None], Dataset]
    state: EnvState | None = None

    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dataset: Dataset | None = None, dataset_args: DatasetArgs | None = None):
        super().__init__()
        if dataset is not None:
            assert dataset_args is None, "When dataset is passed, dataset_arg must be None"
            self.dataset_generator = lambda _: dataset
        if dataset_args is not None:
            assert dataset is None, "When dataset_arg is passed, dataset must be None"
            self.dataset_generator = lambda seed: self.gen_dataset(seed, dataset_args)

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[EnvObservation, dict[str, Any]]:
        global global_reset_counter

        super().reset(seed=seed, options=options)
        global_reset_counter += 1

        # Generate tasks and vms according to seed
        dataset = self.dataset_generator(seed)

        # Map the tasks and VMs from the dataset to the required format
        host_map = {host.id: host for host in dataset.hosts}
        vms = [VmDto.from_vm(vm, host_map[vm.host_id]) for vm in dataset.vms]
        tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
        task_mapper = TaskMapper(tasks)
        mapped_tasks = task_mapper.map_tasks()

        # Sanity check - we should be able to use index and id interchangeably
        for i, task in enumerate(mapped_tasks):
            assert task.id == i, f"Sanity Check Failed: Task ID mismatch, {task.id} != {i}"
        for i, vm in enumerate(vms):
            assert vm.id == i, f"Sanity Check Failed: VM ID mismatch, {vm.id} != {i}"

        # Initial states of tasks and VMs
        task_states = [TaskState() for _ in mapped_tasks]
        vm_states = [VmState() for _ in vms]

        # Dummy task is scheduled initially
        task_states[0].assigned_vm_id = 0
        for task_id in mapped_tasks[0].child_ids:
            task_states[task_id].is_ready = True
        # Create compatibility edges (task, vm)
        compatibilities = [
            (task_id, vm_id)
            for task_id in range(len(mapped_tasks))
            for vm_id in range(len(vms))
            if is_suitable(vms[vm_id], mapped_tasks[task_id])
        ]
        # Create dependencies from parent->child relations
        dependencies = set(
            (task_id, child_id)
            for task_id in range(len(mapped_tasks))  #
            for child_id in mapped_tasks[task_id].child_ids
        )

        # Map to the state
        self.state = EnvState(
            static_state=StaticState(
                task_mapper=task_mapper,
                tasks=mapped_tasks,
                vms=vms,
                compatibilities=compatibilities,
            ),
            task_states=task_states,
            vm_states=vm_states,
            task_dependencies=dependencies,
        )

        return EnvObservation(self.state), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: EnvAction) -> tuple[EnvObservation, float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "State must be initialized"

        # Check validity of the action
        penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.state.task_states)
        if not (0 <= action.task_id < len(self.state.task_states)):
            return EnvObservation(self.state), penalty, True, False, {"error": f"{action}: Invalid task (out of range)"}
        if self.state.task_states[action.task_id].assigned_vm_id is not None:
            return EnvObservation(self.state), penalty, True, False, {"error": f"{action}: Already scheduled task"}
        if not self.state.task_states[action.task_id].is_ready:
            return EnvObservation(self.state), penalty, True, False, {"error": f"{action}: Not ready task"}
        if not is_suitable(self.state.static_state.vms[action.vm_id], self.state.static_state.tasks[action.task_id]):
            return EnvObservation(self.state), penalty, True, False, {"error": f"{action}: Task/VM are not compatible"}

        child_task_ids = [c_id for (p_id, c_id) in self.state.task_dependencies if p_id == action.task_id]
        parent_task_ids = [p_id for (p_id, c_id) in self.state.task_dependencies if c_id == action.task_id]
        processing_time = (
            self.state.static_state.tasks[action.task_id].length
            / self.state.static_state.vms[action.vm_id].cpu_speed_mips
        )

        new_task_states = copy.deepcopy(self.state.task_states)
        new_vm_states = copy.deepcopy(self.state.vm_states)

        # Update scheduled states
        new_task_states[action.task_id].assigned_vm_id = action.vm_id
        new_vm_states[action.vm_id].assigned_task_id = action.task_id
        # Update ready states using new state
        new_task_states[action.task_id].is_ready = False
        for child_id in child_task_ids:
            new_task_states[child_id].is_ready = True
            child_parent_task_ids = [p_id for (p_id, c_id) in self.state.task_dependencies if c_id == child_id]
            for child_parent_task_id in child_parent_task_ids:
                if new_task_states[child_parent_task_id].assigned_vm_id is None:
                    new_task_states[child_id].is_ready = False
                    break

        # Update completion times
        start_time = self.state.vm_states[action.vm_id].completion_time
        for parent_id in parent_task_ids:
            start_time = max(start_time, self.state.task_states[parent_id].completion_time)
        new_task_states[action.task_id].start_time = start_time
        new_task_states[action.task_id].completion_time = start_time + processing_time
        new_vm_states[action.vm_id].completion_time = start_time + processing_time

        # Update energy consumption
        new_task_states[action.task_id].energy_consumption = (
            active_energy_consumption_per_mi(self.state.static_state.vms[action.vm_id])
            * self.state.static_state.tasks[action.task_id].length
        )

        # New dependencies (a new edge between the old task in the VM and this task)
        new_task_dependencies = copy.deepcopy(self.state.task_dependencies)
        vm_prev_task_id = self.state.vm_states[action.vm_id].assigned_task_id or 0
        new_task_dependencies.add((vm_prev_task_id, action.task_id))

        # Check if dummy end active, then all tasks have been scheduled
        # If so, assign it to 0 vm (it doesn\'t matter since length is 0)
        if new_task_states[-1].is_ready:
            new_task_states[-1].is_ready = False
            new_task_states[-1].assigned_vm_id = 0
            new_task_states[-1].start_time = max(vm.completion_time for vm in new_vm_states)
            new_task_states[-1].completion_time = new_task_states[-1].start_time
            new_vm_states[0].assigned_task_id = len(new_task_states) - 1

        # Change the state
        self.state = EnvState(
            static_state=self.state.static_state,
            task_states=new_task_states,
            vm_states=new_vm_states,
            task_dependencies=new_task_dependencies,
        )

        # If the final task is not submitted yet, we can give the immediate rewards (if any)
        if self.state.task_states[-1].assigned_vm_id is None:
            return EnvObservation(self.state), 0, False, False, {}

        # This is final step...
        # Create an assignment array that is sorted by time
        sorted_assignments: list[tuple[float, VmAssignmentDto]] = []
        for task_id, task in enumerate(self.state.task_states):
            assert task.assigned_vm_id is not None, "Final task assigned but there are tasks without VM assigned"
            if task_id == 0 or task_id == len(self.state.task_states) - 1:
                continue
            u_workflow_id, u_vm_id = self.state.static_state.task_mapper.unmap_id(task_id)
            u_assignment = VmAssignmentDto(task.assigned_vm_id, u_workflow_id, u_vm_id)
            sorted_assignments.append((task.completion_time, u_assignment))
        sorted_assignments.sort(key=lambda x: x[0])

        # Store the assignment in info and submit it
        info = {"assignments": [assignment[1] for assignment in sorted_assignments]}
        
        # Calculate makespan and energy consumption for the reward
        makespan = max(vm.completion_time for vm in self.state.vm_states)
        idle_energy = EnvObservation(self.state).idle_energy()
        total_energy = EnvObservation(self.state).total_energy_consumption()

        # The reward function is a combination of negative makespan and negative idle energy.
        # The idle energy is scaled by a factor (e.g., 0.03) to balance its contribution.
        reward = -makespan - (0.03 * idle_energy)

        # Add energy metrics to info for logging in train.py
        info["energy/idle"] = idle_energy
        info["energy/total"] = total_energy

        return EnvObservation(self.state), reward, False, True, info

    @staticmethod
    def gen_dataset(seed: int | None, dataset_args: DatasetArgs):
        return generate_dataset(
            seed=seed if seed is not None else random.randint(1, MAX_TRAINING_DS_SEED),
            host_count=dataset_args.host_count,
            vm_count=dataset_args.vm_count,
            workflow_count=dataset_args.workflow_count,
            gnp_max_n=dataset_args.gnp_max_n,
            gnp_min_n=dataset_args.gnp_min_n,
            max_memory_gb=dataset_args.max_memory_gb,
            min_cpu_speed_mips=dataset_args.min_cpu_speed,
            max_cpu_speed_mips=dataset_args.max_cpu_speed,
            dag_method=dataset_args.dag_method,
            task_length_dist=dataset_args.task_length_dist,
            min_task_length=dataset_args.min_task_length,
            max_task_length=dataset_args.max_task_length,
            task_arrival=dataset_args.task_arrival,
            arrival_rate=dataset_args.arrival_rate,
        )


