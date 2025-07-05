import copy
from dataclasses import dataclass

import numpy as np

from scheduler.rl_model.core.env.state import EnvState


@dataclass
class EnvObservation:
    task_observations: list["TaskObservation"]
    vm_observations: list["VmObservation"]
    task_dependencies: list[tuple[int, int]]
    compatibilities: list[tuple[int, int]]

    _makespan: float | None = None
    _energy_consumption: float | None = None
    _task_completion_time: np.ndarray | None = None

    def __init__(self, state: EnvState):
        self.task_observations = [
            TaskObservation(
                is_ready=state.task_states[task_id].is_ready,
                assigned_vm_id=state.task_states[task_id].assigned_vm_id,
                start_time=state.task_states[task_id].start_time,
                completion_time=state.task_states[task_id].completion_time,
                energy_consumption=state.task_states[task_id].energy_consumption,
                length=state.static_state.tasks[task_id].length,
            )
            for task_id in range(len(state.task_states))
        ]
        self.vm_observations = [
            VmObservation(
                id=vm_id,
                assigned_task_id=state.vm_states[vm_id].assigned_task_id,
                completion_time=state.vm_states[vm_id].completion_time,
                cpu_speed_mips=state.static_state.vms[vm_id].cpu_speed_mips,
                host_power_idle_watt=state.static_state.vms[vm_id].host_power_idle_watt,
                host_power_peak_watt=state.static_state.vms[vm_id].host_power_peak_watt,
                host_cpu_speed_mips=state.static_state.vms[vm_id].host_cpu_speed_mips,
            )
            for vm_id in range(len(state.vm_states))
        ]
        self.task_dependencies = copy.deepcopy(list(state.task_dependencies))
        self.compatibilities = copy.deepcopy(list(state.static_state.compatibilities))

    def makespan(self):
        if self._makespan is not None:
            return self._makespan

        # Calculates the makespan of an observation or and estimate of it if the env is still running
        # Uses max task completion time (task will complete either after parent or after VM completion time)
        task_completion_time = np.ones(len(self.task_observations)) * 1e8
        for task_id in range(len(self.task_observations)):
            # Check if already scheduled task
            if self.task_observations[task_id].assigned_vm_id is not None:
                task_completion_time[task_id] = self.task_observations[task_id].completion_time
                continue

            parent_ids = [pid for pid, cid in self.task_dependencies if cid == task_id]
            compatible_vm_ids = [vid for tid, vid in self.compatibilities if tid == task_id]

            parent_comp_time = max(task_completion_time[parent_ids], default=0)
            for vm_id in compatible_vm_ids:
                vm_comp_time = self.vm_observations[vm_id].completion_time
                vm_speed = self.vm_observations[vm_id].cpu_speed_mips
                task_exec_time = self.task_observations[task_id].length / vm_speed
                new_comp_time = max(parent_comp_time, vm_comp_time) + task_exec_time
                task_completion_time[task_id] = min(new_comp_time, task_completion_time[task_id].item())

        self._makespan = task_completion_time[-1].item()
        self._task_completion_time = task_completion_time
        return self._makespan

    def energy_consumption(self):
        if self._energy_consumption is not None:
            return self._energy_consumption

        from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi

        # Calculates the energy consumption of an observation or and estimate of it if the env is still running
        # Uses minimum possible energy for each unscheduled task
        task_energy_consumption = np.ones(len(self.task_observations)) * 1e8
        for task_id in range(len(self.task_observations)):
            # Check if already scheduled task
            if self.task_observations[task_id].assigned_vm_id is not None:
                task_energy_consumption[task_id] = self.task_observations[task_id].energy_consumption
                continue

            compatible_vm_ids = [vid for tid, vid in self.compatibilities if tid == task_id]
            for vm_id in compatible_vm_ids:
                energy_consumption_rate = active_energy_consumption_per_mi(self.vm_observations[vm_id])
                new_energy_consumption = self.task_observations[task_id].length * energy_consumption_rate
                task_energy_consumption[task_id] = min(new_energy_consumption, task_energy_consumption[task_id].item())

        self._energy_consumption = float(task_energy_consumption.sum())
        return self._energy_consumption

    def task_completion_time(self):
        if self._task_completion_time is None:
            self.makespan()
        return self._task_completion_time

    def idle_energy(self) -> float:
        """Calculates idle energy considering VM reuse."""
        idle_energy = 0.0
        makespan =self.makespan()

        for vm in self.vm_observations:
            if vm.assigned_task_id is None:
                # Case 1: VM never used → idle for entire makespan
                idle_energy += vm.host_power_idle_watt * makespan
            else:
                # Case 2: VM used at least once
                # Get all tasks assigned to this VM (sorted by completion time)
                vm_tasks = sorted(
                    [task for task in self.task_observations
                     if task.assigned_vm_id == vm.id],
                    key=lambda x: x.start_time
                )

                # Idle before first task
                idle_time = vm_tasks[0].start_time

                # Idle between consecutive tasks
                for i in range(1, len(vm_tasks)):
                    idle_time += max(0.0, vm_tasks[i].start_time - vm_tasks[i - 1].completion_time)

                # Idle after last task
                idle_time += max(0.0, makespan - vm_tasks[-1].completion_time)

                idle_energy += vm.host_power_idle_watt * idle_time

        return idle_energy



    def total_energy_consumption(self) -> float:
        """Computes the integral of power over time for all VMs."""
        total_energy = 0.0
        makespan = self.makespan()

        for vm in self.vm_observations:
            # Get all tasks assigned to this VM (sorted by start time)
            vm_tasks = sorted(
                [task for task in self.task_observations
                 if task.assigned_vm_id == vm.id],
                key=lambda x: x.start_time
            )

            current_time = 0.0
            vm_energy = 0.0

            for task in vm_tasks:
                # Idle period before task
                idle_duration = max(0.0, task.start_time - current_time)
                vm_energy += vm.host_power_idle_watt * idle_duration

                # Active period during task
                task_duration = task.completion_time - task.start_time
                vm_energy += vm.host_power_peak_watt * task_duration

                current_time = task.completion_time

            # Idle period after last task
            idle_duration = max(0.0, makespan - current_time)
            vm_energy += vm.host_power_idle_watt * idle_duration

            total_energy += vm_energy

        return total_energy
@dataclass
class TaskObservation:
    is_ready: bool
    assigned_vm_id: int | None
    start_time: float
    completion_time: float
    energy_consumption: float
    length: float


@dataclass
class VmObservation:
    id: int

    assigned_task_id: int | None
    completion_time: float
    cpu_speed_mips: float
    host_power_idle_watt: float
    host_power_peak_watt: float
    host_cpu_speed_mips: float
