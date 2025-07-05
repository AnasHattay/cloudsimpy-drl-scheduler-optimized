"""
Power-Aware Heuristic Algorithms for Workflow Scheduling

This module implements various heuristic scheduling algorithms that serve as
baselines for comparison with DRL agents. Includes both classic algorithms
and power-aware variants.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

from scheduler.dataset_generator.core import Dataset, Workflow, Task, Vm, Host


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision"""
    task_id: int
    vm_id: int
    start_time: float
    estimated_finish_time: float
    estimated_energy: float


@dataclass
class VMState:
    """Tracks the state of a VM during scheduling"""
    vm_id: int
    available_time: float
    total_energy: float
    task_queue: List[Tuple[float, int]]  # (finish_time, task_id)
    
    def __post_init__(self):
        if not hasattr(self, 'task_queue'):
            self.task_queue = []


class SchedulingAlgorithm(ABC):
    """Abstract base class for scheduling algorithms"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        """Schedule tasks and return scheduling decisions"""
        pass
    
    def calculate_execution_time(self, task: Task, vm: Vm) -> float:
        """Calculate execution time for a task on a VM"""
        return task.length / vm.cpu_speed_mips
    
    def calculate_energy_consumption(self, task: Task, vm: Vm, host: Host, execution_time: float) -> float:
        """Calculate energy consumption for executing a task"""
        # Energy = (Peak Power - Idle Power) * Execution Time + Idle Power * Execution Time
        # Simplified model: assume linear relationship between CPU utilization and power
        cpu_utilization = min(1.0, task.length / (vm.cpu_speed_mips * execution_time))
        power_consumption = host.power_idle_watt + (host.power_peak_watt - host.power_idle_watt) * cpu_utilization
        return power_consumption * execution_time / 3600.0  # Convert to Wh
    
    def get_task_dependencies(self, workflows: List[Workflow]) -> Dict[int, Set[int]]:
        """Build dependency graph for all tasks"""
        dependencies = {}  # task_id -> set of prerequisite task_ids
        
        for workflow in workflows:
            for task in workflow.tasks:
                dependencies[task.id] = set()
                
                # Find tasks that have this task as a child
                for other_task in workflow.tasks:
                    if task.id in other_task.child_ids:
                        dependencies[task.id].add(other_task.id)
        
        return dependencies
    
    def get_ready_tasks(self, workflows: List[Workflow], scheduled_tasks: Set[int], dependencies: Dict[int, Set[int]]) -> List[Task]:
        """Get tasks that are ready to be scheduled"""
        ready_tasks = []
        
        for workflow in workflows:
            for task in workflow.tasks:
                if task.id not in scheduled_tasks:
                    # Check if all dependencies are satisfied
                    if dependencies[task.id].issubset(scheduled_tasks):
                        ready_tasks.append(task)
        
        return ready_tasks


class RandomScheduler(SchedulingAlgorithm):
    """Random scheduling algorithm for baseline comparison"""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)
    
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        decisions = []
        vm_states = {vm.id: VMState(vm.id, 0.0, 0.0, []) for vm in dataset.vms}
        scheduled_tasks = set()
        dependencies = self.get_task_dependencies(dataset.workflows)
        
        # Create VM to host mapping
        vm_to_host = {vm.id: next(h for h in dataset.hosts if h.id == vm.host_id) for vm in dataset.vms}
        vm_objects = {vm.id: vm for vm in dataset.vms}
        
        while len(scheduled_tasks) < sum(len(w.tasks) for w in dataset.workflows):
            ready_tasks = self.get_ready_tasks(dataset.workflows, scheduled_tasks, dependencies)
            
            if not ready_tasks:
                break
            
            # Randomly select a task and VM
            task = self.rng.choice(ready_tasks)
            compatible_vms = [vm for vm in dataset.vms if vm.memory_mb >= task.req_memory_mb]
            
            if not compatible_vms:
                scheduled_tasks.add(task.id)  # Skip incompatible task
                continue
            
            vm = self.rng.choice(compatible_vms)
            vm_state = vm_states[vm.id]
            host = vm_to_host[vm.id]
            
            # Calculate timing and energy
            execution_time = self.calculate_execution_time(task, vm)
            start_time = vm_state.available_time
            finish_time = start_time + execution_time
            energy = self.calculate_energy_consumption(task, vm, host, execution_time)
            
            # Update VM state
            vm_state.available_time = finish_time
            vm_state.total_energy += energy
            
            # Record decision
            decisions.append(SchedulingDecision(
                task_id=task.id,
                vm_id=vm.id,
                start_time=start_time,
                estimated_finish_time=finish_time,
                estimated_energy=energy
            ))
            
            scheduled_tasks.add(task.id)
        
        return decisions


class MinMinScheduler(SchedulingAlgorithm):
    """Min-Min scheduling algorithm"""
    
    def __init__(self):
        super().__init__("Min-Min")
    
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        decisions = []
        vm_states = {vm.id: VMState(vm.id, 0.0, 0.0, []) for vm in dataset.vms}
        scheduled_tasks = set()
        dependencies = self.get_task_dependencies(dataset.workflows)
        
        # Create mappings
        vm_to_host = {vm.id: next(h for h in dataset.hosts if h.id == vm.host_id) for vm in dataset.vms}
        vm_objects = {vm.id: vm for vm in dataset.vms}
        
        while len(scheduled_tasks) < sum(len(w.tasks) for w in dataset.workflows):
            ready_tasks = self.get_ready_tasks(dataset.workflows, scheduled_tasks, dependencies)
            
            if not ready_tasks:
                break
            
            best_task = None
            best_vm = None
            best_finish_time = float('inf')
            
            # For each ready task, find the VM that gives minimum completion time
            for task in ready_tasks:
                compatible_vms = [vm for vm in dataset.vms if vm.memory_mb >= task.req_memory_mb]
                
                for vm in compatible_vms:
                    vm_state = vm_states[vm.id]
                    execution_time = self.calculate_execution_time(task, vm)
                    finish_time = vm_state.available_time + execution_time
                    
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_task = task
                        best_vm = vm
            
            if best_task is None:
                break
            
            # Schedule the best task-VM pair
            vm_state = vm_states[best_vm.id]
            host = vm_to_host[best_vm.id]
            execution_time = self.calculate_execution_time(best_task, best_vm)
            start_time = vm_state.available_time
            finish_time = start_time + execution_time
            energy = self.calculate_energy_consumption(best_task, best_vm, host, execution_time)
            
            # Update VM state
            vm_state.available_time = finish_time
            vm_state.total_energy += energy
            
            # Record decision
            decisions.append(SchedulingDecision(
                task_id=best_task.id,
                vm_id=best_vm.id,
                start_time=start_time,
                estimated_finish_time=finish_time,
                estimated_energy=energy
            ))
            
            scheduled_tasks.add(best_task.id)
        
        return decisions


class MaxMinScheduler(SchedulingAlgorithm):
    """Max-Min scheduling algorithm"""
    
    def __init__(self):
        super().__init__("Max-Min")
    
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        decisions = []
        vm_states = {vm.id: VMState(vm.id, 0.0, 0.0, []) for vm in dataset.vms}
        scheduled_tasks = set()
        dependencies = self.get_task_dependencies(dataset.workflows)
        
        # Create mappings
        vm_to_host = {vm.id: next(h for h in dataset.hosts if h.id == vm.host_id) for vm in dataset.vms}
        vm_objects = {vm.id: vm for vm in dataset.vms}
        
        while len(scheduled_tasks) < sum(len(w.tasks) for w in dataset.workflows):
            ready_tasks = self.get_ready_tasks(dataset.workflows, scheduled_tasks, dependencies)
            
            if not ready_tasks:
                break
            
            # Find minimum completion time for each task
            task_min_times = {}
            task_best_vm = {}
            
            for task in ready_tasks:
                compatible_vms = [vm for vm in dataset.vms if vm.memory_mb >= task.req_memory_mb]
                min_time = float('inf')
                best_vm = None
                
                for vm in compatible_vms:
                    vm_state = vm_states[vm.id]
                    execution_time = self.calculate_execution_time(task, vm)
                    finish_time = vm_state.available_time + execution_time
                    
                    if finish_time < min_time:
                        min_time = finish_time
                        best_vm = vm
                
                if best_vm is not None:
                    task_min_times[task.id] = min_time
                    task_best_vm[task.id] = best_vm
            
            if not task_min_times:
                break
            
            # Select task with maximum minimum completion time
            selected_task_id = max(task_min_times.keys(), key=lambda tid: task_min_times[tid])
            selected_task = next(t for t in ready_tasks if t.id == selected_task_id)
            selected_vm = task_best_vm[selected_task_id]
            
            # Schedule the selected task
            vm_state = vm_states[selected_vm.id]
            host = vm_to_host[selected_vm.id]
            execution_time = self.calculate_execution_time(selected_task, selected_vm)
            start_time = vm_state.available_time
            finish_time = start_time + execution_time
            energy = self.calculate_energy_consumption(selected_task, selected_vm, host, execution_time)
            
            # Update VM state
            vm_state.available_time = finish_time
            vm_state.total_energy += energy
            
            # Record decision
            decisions.append(SchedulingDecision(
                task_id=selected_task.id,
                vm_id=selected_vm.id,
                start_time=start_time,
                estimated_finish_time=finish_time,
                estimated_energy=energy
            ))
            
            scheduled_tasks.add(selected_task.id)
        
        return decisions


class HEFTScheduler(SchedulingAlgorithm):
    """Heterogeneous Earliest Finish Time (HEFT) algorithm"""
    
    def __init__(self):
        super().__init__("HEFT")
    
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        decisions = []
        vm_states = {vm.id: VMState(vm.id, 0.0, 0.0, []) for vm in dataset.vms}
        dependencies = self.get_task_dependencies(dataset.workflows)
        
        # Create mappings
        vm_to_host = {vm.id: next(h for h in dataset.hosts if h.id == vm.host_id) for vm in dataset.vms}
        
        # Calculate task priorities (upward rank)
        all_tasks = []
        for workflow in dataset.workflows:
            all_tasks.extend(workflow.tasks)
        
        task_priorities = self._calculate_upward_rank(all_tasks, dataset.vms, dependencies)
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(all_tasks, key=lambda t: task_priorities[t.id], reverse=True)
        
        # Schedule tasks in priority order
        for task in sorted_tasks:
            compatible_vms = [vm for vm in dataset.vms if vm.memory_mb >= task.req_memory_mb]
            
            if not compatible_vms:
                continue
            
            best_vm = None
            best_finish_time = float('inf')
            best_start_time = 0
            
            for vm in compatible_vms:
                vm_state = vm_states[vm.id]
                execution_time = self.calculate_execution_time(task, vm)
                
                # Find earliest start time considering dependencies
                earliest_start = vm_state.available_time
                
                # Check dependency constraints
                for dep_task_id in dependencies[task.id]:
                    for decision in decisions:
                        if decision.task_id == dep_task_id:
                            earliest_start = max(earliest_start, decision.estimated_finish_time)
                            break
                
                finish_time = earliest_start + execution_time
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_start_time = earliest_start
                    best_vm = vm
            
            if best_vm is None:
                continue
            
            # Schedule the task
            vm_state = vm_states[best_vm.id]
            host = vm_to_host[best_vm.id]
            execution_time = self.calculate_execution_time(task, best_vm)
            energy = self.calculate_energy_consumption(task, best_vm, host, execution_time)
            
            # Update VM state
            vm_state.available_time = best_finish_time
            vm_state.total_energy += energy
            
            # Record decision
            decisions.append(SchedulingDecision(
                task_id=task.id,
                vm_id=best_vm.id,
                start_time=best_start_time,
                estimated_finish_time=best_finish_time,
                estimated_energy=energy
            ))
        
        return decisions
    
    def _calculate_upward_rank(self, tasks: List[Task], vms: List[Vm], dependencies: Dict[int, Set[int]]) -> Dict[int, float]:
        """Calculate upward rank for task prioritization"""
        ranks = {}
        
        # Calculate average execution time for each task
        avg_exec_times = {}
        for task in tasks:
            compatible_vms = [vm for vm in vms if vm.memory_mb >= task.req_memory_mb]
            if compatible_vms:
                exec_times = [self.calculate_execution_time(task, vm) for vm in compatible_vms]
                avg_exec_times[task.id] = sum(exec_times) / len(exec_times)
            else:
                avg_exec_times[task.id] = float('inf')
        
        # Calculate ranks using topological sort (bottom-up)
        def calculate_rank(task_id: int) -> float:
            if task_id in ranks:
                return ranks[task_id]
            
            task = next(t for t in tasks if t.id == task_id)
            
            # Base case: no children
            if not task.child_ids:
                ranks[task_id] = avg_exec_times[task_id]
                return ranks[task_id]
            
            # Recursive case: max of children ranks
            max_child_rank = 0
            for child_id in task.child_ids:
                max_child_rank = max(max_child_rank, calculate_rank(child_id))
            
            ranks[task_id] = avg_exec_times[task_id] + max_child_rank
            return ranks[task_id]
        
        # Calculate ranks for all tasks
        for task in tasks:
            calculate_rank(task.id)
        
        return ranks


class PowerAwareMinMinScheduler(SchedulingAlgorithm):
    """Power-aware variant of Min-Min algorithm"""
    
    def __init__(self, energy_weight: float = 0.3):
        super().__init__("Power-Aware Min-Min")
        self.energy_weight = energy_weight  # Weight for energy vs makespan
    
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        decisions = []
        vm_states = {vm.id: VMState(vm.id, 0.0, 0.0, []) for vm in dataset.vms}
        scheduled_tasks = set()
        dependencies = self.get_task_dependencies(dataset.workflows)
        
        # Create mappings
        vm_to_host = {vm.id: next(h for h in dataset.hosts if h.id == vm.host_id) for vm in dataset.vms}
        
        while len(scheduled_tasks) < sum(len(w.tasks) for w in dataset.workflows):
            ready_tasks = self.get_ready_tasks(dataset.workflows, scheduled_tasks, dependencies)
            
            if not ready_tasks:
                break
            
            best_task = None
            best_vm = None
            best_score = float('inf')
            
            # For each ready task, find the VM that gives best time-energy tradeoff
            for task in ready_tasks:
                compatible_vms = [vm for vm in dataset.vms if vm.memory_mb >= task.req_memory_mb]
                
                for vm in compatible_vms:
                    vm_state = vm_states[vm.id]
                    host = vm_to_host[vm.id]
                    execution_time = self.calculate_execution_time(task, vm)
                    finish_time = vm_state.available_time + execution_time
                    energy = self.calculate_energy_consumption(task, vm, host, execution_time)
                    
                    # Combined score: weighted sum of normalized time and energy
                    # Normalize by dividing by maximum possible values
                    max_time = max(vm_state.available_time + self.calculate_execution_time(task, vm) 
                                 for vm in compatible_vms)
                    max_energy = max(self.calculate_energy_consumption(task, vm, vm_to_host[vm.id], 
                                   self.calculate_execution_time(task, vm)) for vm in compatible_vms)
                    
                    if max_time > 0 and max_energy > 0:
                        normalized_time = finish_time / max_time
                        normalized_energy = energy / max_energy
                        score = (1 - self.energy_weight) * normalized_time + self.energy_weight * normalized_energy
                        
                        if score < best_score:
                            best_score = score
                            best_task = task
                            best_vm = vm
            
            if best_task is None:
                break
            
            # Schedule the best task-VM pair
            vm_state = vm_states[best_vm.id]
            host = vm_to_host[best_vm.id]
            execution_time = self.calculate_execution_time(best_task, best_vm)
            start_time = vm_state.available_time
            finish_time = start_time + execution_time
            energy = self.calculate_energy_consumption(best_task, best_vm, host, execution_time)
            
            # Update VM state
            vm_state.available_time = finish_time
            vm_state.total_energy += energy
            
            # Record decision
            decisions.append(SchedulingDecision(
                task_id=best_task.id,
                vm_id=best_vm.id,
                start_time=start_time,
                estimated_finish_time=finish_time,
                estimated_energy=energy
            ))
            
            scheduled_tasks.add(best_task.id)
        
        return decisions


class EnergyEfficientHEFTScheduler(SchedulingAlgorithm):
    """Energy-efficient variant of HEFT algorithm"""
    
    def __init__(self, energy_weight: float = 0.4):
        super().__init__("Energy-Efficient HEFT")
        self.energy_weight = energy_weight
    
    def schedule(self, dataset: Dataset) -> List[SchedulingDecision]:
        decisions = []
        vm_states = {vm.id: VMState(vm.id, 0.0, 0.0, []) for vm in dataset.vms}
        dependencies = self.get_task_dependencies(dataset.workflows)
        
        # Create mappings
        vm_to_host = {vm.id: next(h for h in dataset.hosts if h.id == vm.host_id) for vm in dataset.vms}
        
        # Calculate task priorities considering both time and energy
        all_tasks = []
        for workflow in dataset.workflows:
            all_tasks.extend(workflow.tasks)
        
        task_priorities = self._calculate_energy_aware_rank(all_tasks, dataset.vms, vm_to_host, dependencies)
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(all_tasks, key=lambda t: task_priorities[t.id], reverse=True)
        
        # Schedule tasks in priority order
        for task in sorted_tasks:
            compatible_vms = [vm for vm in dataset.vms if vm.memory_mb >= task.req_memory_mb]
            
            if not compatible_vms:
                continue
            
            best_vm = None
            best_score = float('inf')
            best_start_time = 0
            best_finish_time = 0
            
            for vm in compatible_vms:
                vm_state = vm_states[vm.id]
                host = vm_to_host[vm.id]
                execution_time = self.calculate_execution_time(task, vm)
                
                # Find earliest start time considering dependencies
                earliest_start = vm_state.available_time
                
                for dep_task_id in dependencies[task.id]:
                    for decision in decisions:
                        if decision.task_id == dep_task_id:
                            earliest_start = max(earliest_start, decision.estimated_finish_time)
                            break
                
                finish_time = earliest_start + execution_time
                energy = self.calculate_energy_consumption(task, vm, host, execution_time)
                
                # Combined score considering both time and energy
                score = (1 - self.energy_weight) * finish_time + self.energy_weight * energy * 1000  # Scale energy
                
                if score < best_score:
                    best_score = score
                    best_start_time = earliest_start
                    best_finish_time = finish_time
                    best_vm = vm
            
            if best_vm is None:
                continue
            
            # Schedule the task
            vm_state = vm_states[best_vm.id]
            host = vm_to_host[best_vm.id]
            execution_time = self.calculate_execution_time(task, best_vm)
            energy = self.calculate_energy_consumption(task, best_vm, host, execution_time)
            
            # Update VM state
            vm_state.available_time = best_finish_time
            vm_state.total_energy += energy
            
            # Record decision
            decisions.append(SchedulingDecision(
                task_id=task.id,
                vm_id=best_vm.id,
                start_time=best_start_time,
                estimated_finish_time=best_finish_time,
                estimated_energy=energy
            ))
        
        return decisions
    
    def _calculate_energy_aware_rank(self, tasks: List[Task], vms: List[Vm], 
                                   vm_to_host: Dict[int, Host], dependencies: Dict[int, Set[int]]) -> Dict[int, float]:
        """Calculate energy-aware upward rank for task prioritization"""
        ranks = {}
        
        # Calculate average execution time and energy for each task
        avg_metrics = {}
        for task in tasks:
            compatible_vms = [vm for vm in vms if vm.memory_mb >= task.req_memory_mb]
            if compatible_vms:
                exec_times = []
                energies = []
                for vm in compatible_vms:
                    exec_time = self.calculate_execution_time(task, vm)
                    energy = self.calculate_energy_consumption(task, vm, vm_to_host[vm.id], exec_time)
                    exec_times.append(exec_time)
                    energies.append(energy)
                
                avg_time = sum(exec_times) / len(exec_times)
                avg_energy = sum(energies) / len(energies)
                # Combined metric
                avg_metrics[task.id] = (1 - self.energy_weight) * avg_time + self.energy_weight * avg_energy * 1000
            else:
                avg_metrics[task.id] = float('inf')
        
        # Calculate ranks using topological sort (bottom-up)
        def calculate_rank(task_id: int) -> float:
            if task_id in ranks:
                return ranks[task_id]
            
            task = next(t for t in tasks if t.id == task_id)
            
            # Base case: no children
            if not task.child_ids:
                ranks[task_id] = avg_metrics[task_id]
                return ranks[task_id]
            
            # Recursive case: max of children ranks
            max_child_rank = 0
            for child_id in task.child_ids:
                max_child_rank = max(max_child_rank, calculate_rank(child_id))
            
            ranks[task_id] = avg_metrics[task_id] + max_child_rank
            return ranks[task_id]
        
        # Calculate ranks for all tasks
        for task in tasks:
            calculate_rank(task.id)
        
        return ranks


def get_all_heuristic_algorithms() -> List[SchedulingAlgorithm]:
    """Get all available heuristic algorithms"""
    return [
        RandomScheduler(),
        MinMinScheduler(),
        MaxMinScheduler(),
        HEFTScheduler(),
        PowerAwareMinMinScheduler(energy_weight=0.3),
        PowerAwareMinMinScheduler(energy_weight=0.5),
        EnergyEfficientHEFTScheduler(energy_weight=0.3),
        EnergyEfficientHEFTScheduler(energy_weight=0.5),
    ]


if __name__ == "__main__":
    # Example usage
    from evaluation.scenario_generator import create_evaluation_scenarios
    
    # Create a sample scenario
    generator = create_evaluation_scenarios()
    scenario = generator.get_all_scenarios()[0]
    dataset = generator.generate_scenario_dataset(scenario)
    
    print(f"Testing heuristic algorithms on scenario: {scenario.name}")
    print(f"Dataset: {len(dataset.workflows)} workflows, {sum(len(w.tasks) for w in dataset.workflows)} tasks")
    
    # Test all algorithms
    algorithms = get_all_heuristic_algorithms()
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm.name}...")
        decisions = algorithm.schedule(dataset)
        
        if decisions:
            makespan = max(d.estimated_finish_time for d in decisions)
            total_energy = sum(d.estimated_energy for d in decisions)
            print(f"  Makespan: {makespan:.2f}")
            print(f"  Total Energy: {total_energy:.4f} Wh")
            print(f"  Scheduled Tasks: {len(decisions)}")
        else:
            print("  No valid schedule found")

