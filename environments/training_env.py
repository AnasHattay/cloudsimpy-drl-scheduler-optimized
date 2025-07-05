"""
Lightweight Training Environment for CloudSimPy DRL Scheduler

This module implements a lightweight environment optimized for training speed
using mathematical models instead of full discrete event simulation.
"""

import numpy as np
import copy
from typing import Any, Dict, List, Tuple, Optional
from gymnasium import spaces
from dataclasses import dataclass

from .abstract_env import AbstractSchedulingEnvironment, EnvironmentConfig
from scheduler.dataset_generator.gen_dataset import generate_dataset


@dataclass
class TaskDto:
    """Lightweight task representation"""
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    req_cores: int
    child_ids: List[int]
    parent_ids: List[int]


@dataclass
class VmDto:
    """Lightweight VM representation"""
    id: int
    host_id: int
    cpu_speed_mips: int
    memory_mb: int
    cores: int
    power_idle_watt: float
    power_peak_watt: float


@dataclass
class TaskState:
    """State information for a task"""
    assigned_vm_id: Optional[int] = None
    is_ready: bool = False
    is_completed: bool = False
    start_time: float = 0.0
    completion_time: float = 0.0
    energy_consumption: float = 0.0


@dataclass
class VmState:
    """State information for a VM"""
    assigned_tasks: List[int] = None
    current_completion_time: float = 0.0
    total_energy: float = 0.0
    memory_used: int = 0
    
    def __post_init__(self):
        if self.assigned_tasks is None:
            self.assigned_tasks = []


class LightweightState:
    """Lightweight state representation for fast training"""
    
    def __init__(self, tasks: List[TaskDto], vms: List[VmDto]):
        self.tasks = tasks
        self.vms = vms
        self.task_states = [TaskState() for _ in tasks]
        self.vm_states = [VmState() for _ in vms]
        self.current_time = 0.0
        self.completed_tasks = set()
        
        # Build dependency graph
        self.dependencies = {}  # task_id -> set of parent task_ids
        self.dependents = {}    # task_id -> set of child task_ids
        
        for task in tasks:
            self.dependencies[task.id] = set(task.parent_ids)
            self.dependents[task.id] = set(task.child_ids)
        
        # Initialize ready tasks
        self._update_ready_tasks()
    
    def copy(self):
        """Create a deep copy of the state"""
        new_state = LightweightState(self.tasks.copy(), self.vms.copy())
        new_state.task_states = [copy.deepcopy(ts) for ts in self.task_states]
        new_state.vm_states = [copy.deepcopy(vs) for vs in self.vm_states]
        new_state.current_time = self.current_time
        new_state.completed_tasks = self.completed_tasks.copy()
        new_state.dependencies = {k: v.copy() for k, v in self.dependencies.items()}
        new_state.dependents = {k: v.copy() for k, v in self.dependents.items()}
        return new_state
    
    def is_task_ready(self, task_id: int) -> bool:
        """Check if task is ready for assignment"""
        return (task_id < len(self.task_states) and 
                self.task_states[task_id].is_ready and
                not self.task_states[task_id].is_completed and
                self.task_states[task_id].assigned_vm_id is None)
    
    def is_task_assigned(self, task_id: int) -> bool:
        """Check if task is already assigned"""
        return (task_id < len(self.task_states) and 
                self.task_states[task_id].assigned_vm_id is not None)
    
    def is_task_completed(self, task_id: int) -> bool:
        """Check if task is completed"""
        return task_id in self.completed_tasks
    
    def is_compatible(self, task_id: int, vm_id: int) -> bool:
        """Check if task can be assigned to VM"""
        if task_id >= len(self.tasks) or vm_id >= len(self.vms):
            return False
        
        task = self.tasks[task_id]
        vm = self.vms[vm_id]
        vm_state = self.vm_states[vm_id]
        
        # Check memory constraint
        return vm_state.memory_used + task.req_memory_mb <= vm.memory_mb
    
    def assign_task(self, task_id: int, vm_id: int, completion_time: float, energy: float):
        """Assign task to VM"""
        task = self.tasks[task_id]
        vm = self.vms[vm_id]
        
        # Update task state
        self.task_states[task_id].assigned_vm_id = vm_id
        self.task_states[task_id].start_time = self.vm_states[vm_id].current_completion_time
        self.task_states[task_id].completion_time = completion_time
        self.task_states[task_id].energy_consumption = energy
        self.task_states[task_id].is_completed = True
        
        # Update VM state
        self.vm_states[vm_id].assigned_tasks.append(task_id)
        self.vm_states[vm_id].current_completion_time = completion_time
        self.vm_states[vm_id].total_energy += energy
        self.vm_states[vm_id].memory_used += task.req_memory_mb
        
        # Mark task as completed
        self.completed_tasks.add(task_id)
        
        # Update current time
        self.current_time = max(self.current_time, completion_time)
    
    def _update_ready_tasks(self):
        """Update which tasks are ready for assignment"""
        for task_id, task in enumerate(self.tasks):
            if task_id in self.completed_tasks:
                continue
                
            # Check if all dependencies are completed
            dependencies_met = all(
                dep_id in self.completed_tasks 
                for dep_id in self.dependencies.get(task_id, set())
            )
            
            self.task_states[task_id].is_ready = dependencies_met
    
    def get_makespan(self) -> float:
        """Calculate current makespan"""
        if not self.completed_tasks:
            return 0.0
        return max(
            self.task_states[task_id].completion_time 
            for task_id in self.completed_tasks
        )
    
    def get_total_energy(self) -> float:
        """Calculate total energy consumption"""
        return sum(vm_state.total_energy for vm_state in self.vm_states)
    
    def get_vm_load(self, vm_id: int) -> float:
        """Get current load on VM"""
        return len(self.vm_states[vm_id].assigned_tasks)
    
    def get_memory_pressure(self, vm_id: int) -> float:
        """Get memory pressure on VM"""
        vm = self.vms[vm_id]
        vm_state = self.vm_states[vm_id]
        return vm_state.memory_used / vm.memory_mb
    
    def is_terminal(self) -> bool:
        """Check if all tasks are completed"""
        return len(self.completed_tasks) == len(self.tasks)
    
    def get_ready_tasks(self) -> List[int]:
        """Get list of ready task IDs"""
        return [
            task_id for task_id in range(len(self.tasks))
            if self.is_task_ready(task_id)
        ]
    
    def to_observation_array(self) -> np.ndarray:
        """Convert state to observation array"""
        features = []
        
        # Task features
        for task_id, task in enumerate(self.tasks):
            task_state = self.task_states[task_id]
            task_features = [
                float(task.length),
                float(task.req_memory_mb),
                float(task.req_cores),
                float(task_state.is_ready),
                float(task_state.is_completed),
                float(task_state.assigned_vm_id is not None),
                task_state.completion_time,
                task_state.energy_consumption
            ]
            features.extend(task_features)
        
        # VM features
        for vm_id, vm in enumerate(self.vms):
            vm_state = self.vm_states[vm_id]
            vm_features = [
                float(vm.cores),
                float(vm.cpu_speed_mips),
                float(vm.memory_mb),
                float(vm.power_idle_watt),
                float(vm.power_peak_watt),
                vm_state.current_completion_time,
                vm_state.total_energy,
                float(len(vm_state.assigned_tasks)),
                float(vm_state.memory_used)
            ]
            features.extend(vm_features)
        
        # Global features
        global_features = [
            self.current_time,
            self.get_makespan(),
            self.get_total_energy(),
            float(len(self.completed_tasks)),
            float(len(self.get_ready_tasks()))
        ]
        features.extend(global_features)
        
        return np.array(features, dtype=np.float32)
    
    @classmethod
    def from_dataset(cls, dataset):
        """Create state from dataset"""
        # Convert tasks
        tasks = []
        task_id_map = {}
        
        for workflow in dataset.workflows:
            for task in workflow.tasks:
                # Build parent relationships
                parent_ids = []
                for other_workflow in dataset.workflows:
                    for other_task in other_workflow.tasks:
                        if task.id in other_task.child_ids:
                            parent_ids.append(other_task.id)
                
                task_dto = TaskDto(
                    id=task.id,
                    workflow_id=task.workflow_id,
                    length=task.length,
                    req_memory_mb=task.req_memory_mb,
                    req_cores=getattr(task, 'req_cores', 1),
                    child_ids=task.child_ids.copy(),
                    parent_ids=parent_ids
                )
                tasks.append(task_dto)
                task_id_map[task.id] = len(tasks) - 1
        
        # Convert VMs
        host_map = {host.id: host for host in dataset.hosts}
        vms = []
        
        for vm in dataset.vms:
            host = host_map[vm.host_id]
            vm_dto = VmDto(
                id=vm.id,
                host_id=vm.host_id,
                cpu_speed_mips=vm.cpu_speed_mips,
                memory_mb=vm.memory_mb,
                cores=host.cores,
                power_idle_watt=host.power_idle_watt,
                power_peak_watt=host.power_peak_watt
            )
            vms.append(vm_dto)
        
        return cls(tasks, vms)


class TaskCompletionModel:
    """Mathematical model for predicting task completion times"""
    
    def predict(self, task: TaskDto, vm: VmDto, state: LightweightState) -> float:
        """Predict task completion time on given VM"""
        # Base execution time
        base_time = task.length / vm.cpu_speed_mips
        
        # Apply utilization factor
        utilization_factor = self._calculate_utilization_factor(vm, state)
        
        # Apply contention factor
        contention_factor = self._calculate_contention_factor(task, vm, state)
        
        # Calculate start time (when VM becomes available)
        start_time = state.vm_states[vm.id].current_completion_time
        
        # Total completion time
        execution_time = base_time * utilization_factor * contention_factor
        completion_time = start_time + execution_time
        
        return completion_time
    
    def _calculate_utilization_factor(self, vm: VmDto, state: LightweightState) -> float:
        """Calculate VM utilization factor"""
        current_load = len(state.vm_states[vm.id].assigned_tasks)
        # Higher load = slower execution (simplified model)
        return 1.0 + (current_load / vm.cores) * 0.3
    
    def _calculate_contention_factor(self, task: TaskDto, vm: VmDto, state: LightweightState) -> float:
        """Calculate resource contention factor"""
        memory_pressure = state.get_memory_pressure(vm.id)
        # Memory pressure affects performance
        return 1.0 + memory_pressure * 0.2


class EnergyConsumptionModel:
    """Mathematical model for predicting energy consumption"""
    
    def predict(self, task: TaskDto, vm: VmDto, execution_time: float) -> float:
        """Predict energy consumption for task execution"""
        # Dynamic power consumption model
        idle_power = vm.power_idle_watt
        peak_power = vm.power_peak_watt
        
        # Utilization-based power consumption
        utilization = min(1.0, task.req_cores / vm.cores)
        power_consumption = idle_power + (peak_power - idle_power) * utilization
        
        # Energy = Power * Time (convert to Wh)
        return power_consumption * execution_time / 3600.0


class LightweightTrainingEnvironment(AbstractSchedulingEnvironment):
    """Lightweight environment optimized for training speed"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.state: Optional[LightweightState] = None
        self.task_completion_model = TaskCompletionModel()
        self.energy_model = EnergyConsumptionModel()
        self.dataset_generator = None
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _setup_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        """Setup observation and action spaces based on dataset"""
        # Get dataset size estimates
        dataset_args = self.config.dataset_args
        max_tasks = dataset_args.get('num_workflows', 10) * dataset_args.get('num_tasks_per_workflow', 20)
        max_vms = dataset_args.get('num_vms', 16)
        
        # Action space: discrete choice of task-vm pairs
        self.action_space = spaces.Discrete(max_tasks * max_vms)
        
        # Observation space: task features + vm features + global features
        task_features = max_tasks * 8  # 8 features per task
        vm_features = max_vms * 9      # 9 features per VM
        global_features = 5            # 5 global features
        obs_dim = task_features + vm_features + global_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        return self.observation_space, self.action_space
    
    def _reset_implementation(self, seed: Optional[int], options: Optional[Dict[str, Any]]):
        """Reset to initial state with new dataset"""
        # Generate dataset
        dataset = self._generate_dataset(seed)
        
        # Create lightweight state
        self.state = LightweightState.from_dataset(dataset)
        
        # Update spaces based on actual dataset size
        self._update_spaces_from_state()
        
        observation = self._get_observation()
        info = self._get_info(self.state)
        
        return observation, info
    
    def _step_implementation(self, action):
        """Execute action in lightweight simulation"""
        # Decode action to task-vm assignment
        task_id, vm_id = self._decode_action(action)
        
        # Store old state for reward calculation
        old_makespan = self.state.get_makespan()
        old_energy = self.state.get_total_energy()
        
        # Apply assignment using mathematical models
        self._apply_assignment(task_id, vm_id)
        
        # Calculate reward
        reward = self._calculate_reward_from_metrics(old_makespan, old_energy)
        
        # Check termination
        terminated = self._is_terminal(self.state)
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info(self.state)
        
        return observation, reward, terminated, truncated, info
    
    def _apply_assignment(self, task_id: int, vm_id: int):
        """Apply task-vm assignment using mathematical models"""
        if not self.state.is_task_ready(task_id) or self.state.is_task_assigned(task_id):
            return  # Invalid assignment
        
        task = self.state.tasks[task_id]
        vm = self.state.vms[vm_id]
        
        # Calculate completion time and energy using models
        completion_time = self.task_completion_model.predict(task, vm, self.state)
        energy_consumption = self.energy_model.predict(task, vm, completion_time - self.state.vm_states[vm_id].current_completion_time)
        
        # Update state
        self.state.assign_task(task_id, vm_id, completion_time, energy_consumption)
        self.state._update_ready_tasks()
    
    def _calculate_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Calculate reward based on state transition (not used in this implementation)"""
        return 0.0
    
    def _calculate_reward_from_metrics(self, old_makespan: float, old_energy: float) -> float:
        """Calculate reward based on makespan and energy improvements"""
        new_makespan = self.state.get_makespan()
        new_energy = self.state.get_total_energy()
        
        # Calculate improvements (negative means worse)
        makespan_improvement = old_makespan - new_makespan if old_makespan > 0 else 0
        energy_improvement = old_energy - new_energy if old_energy > 0 else 0
        
        # Weighted combination
        weights = self.config.reward_weights
        reward = (weights["makespan"] * makespan_improvement + 
                 weights["energy"] * energy_improvement / 1000.0)
        
        # Add completion bonus
        if self.state.is_terminal():
            reward += 100.0
        
        # Add constraint violation penalties
        if self._has_constraint_violations():
            reward -= 50.0
        
        return reward
    
    def _has_constraint_violations(self) -> bool:
        """Check for constraint violations"""
        # Check memory constraints
        for vm_id, vm in enumerate(self.state.vms):
            if self.state.vm_states[vm_id].memory_used > vm.memory_mb:
                return True
        return False
    
    def _is_terminal(self, state: Any) -> bool:
        """Check if current state is terminal"""
        return self.state.is_terminal()
    
    def _get_info(self, state: Any) -> Dict[str, Any]:
        """Get additional information about current state"""
        return {
            "makespan": self.state.get_makespan(),
            "total_energy": self.state.get_total_energy(),
            "completed_tasks": len(self.state.completed_tasks),
            "total_tasks": len(self.state.tasks),
            "ready_tasks": len(self.state.get_ready_tasks()),
            "episode": self.get_episode_count(),
            "step": self.get_step_count()
        }
    
    def _validate_action(self, action: Any) -> bool:
        """Validate action legality"""
        if not isinstance(action, int) or action < 0 or action >= self.action_space.n:
            return False
        
        task_id, vm_id = self._decode_action(action)
        
        # Check bounds
        if task_id >= len(self.state.tasks) or vm_id >= len(self.state.vms):
            return False
        
        return (self.state.is_task_ready(task_id) and 
                not self.state.is_task_assigned(task_id) and
                self.state.is_compatible(task_id, vm_id))
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode discrete action to task-vm pair"""
        num_vms = len(self.state.vms)
        task_id = action // num_vms
        vm_id = action % num_vms
        return task_id, vm_id
    
    def _get_observation(self) -> np.ndarray:
        """Convert state to observation array"""
        return self.state.to_observation_array()
    
    def _generate_dataset(self, seed: Optional[int]):
        """Generate dataset for training"""
        if seed is None:
            seed = self.config.seed or 42
        
        # Map our dataset args to the expected format
        dataset_args = self.config.dataset_args
        
        # Provide defaults for missing parameters
        defaults = {
            "host_count": 2,
            "vm_count": 4,
            "max_memory_gb": 10,
            "min_cpu_speed_mips": 500,
            "max_cpu_speed_mips": 5000,
            "workflow_count": 3,
            "dag_method": "gnp",
            "gnp_min_n": 1,
            "gnp_max_n": 10,
            "task_length_dist": "normal",
            "min_task_length": 500,
            "max_task_length": 100000,
            "task_arrival": "static",
            "arrival_rate": 3.0
        }
        
        # Map from our naming convention to the expected naming
        param_mapping = {
            "num_hosts": "host_count",
            "num_vms": "vm_count", 
            "num_workflows": "workflow_count",
            "num_tasks_per_workflow": "gnp_max_n"
        }
        
        # Build final parameters
        final_params = defaults.copy()
        
        # Apply user-provided parameters with mapping
        for key, value in dataset_args.items():
            mapped_key = param_mapping.get(key, key)
            final_params[mapped_key] = value
        
        # Ensure gnp_min_n is reasonable
        if final_params["gnp_max_n"] > 1:
            final_params["gnp_min_n"] = max(1, final_params["gnp_max_n"] // 2)
        
        return generate_dataset(
            seed=seed,
            host_count=final_params["host_count"],
            vm_count=final_params["vm_count"],
            max_memory_gb=final_params["max_memory_gb"],
            min_cpu_speed_mips=final_params["min_cpu_speed_mips"],
            max_cpu_speed_mips=final_params["max_cpu_speed_mips"],
            workflow_count=final_params["workflow_count"],
            dag_method=final_params["dag_method"],
            gnp_min_n=final_params["gnp_min_n"],
            gnp_max_n=final_params["gnp_max_n"],
            task_length_dist=final_params["task_length_dist"],
            min_task_length=final_params["min_task_length"],
            max_task_length=final_params["max_task_length"],
            task_arrival=final_params["task_arrival"],
            arrival_rate=final_params["arrival_rate"]
        )
    
    def _update_spaces_from_state(self):
        """Update action and observation spaces based on actual state"""
        if self.state is None:
            return
        
        num_tasks = len(self.state.tasks)
        num_vms = len(self.state.vms)
        
        # Update action space
        self.action_space = spaces.Discrete(num_tasks * num_vms)
        
        # Update observation space
        obs_array = self.state.to_observation_array()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_array.shape, dtype=np.float32
        )

