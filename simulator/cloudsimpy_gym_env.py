"""
Enhanced CloudSimPy Gym Environment with Complete Integration

This is the final, optimized version that properly integrates:
1. CloudSimPy simulation engine
2. Comprehensive energy monitoring with idle power
3. Simplified and clean interface
4. Real-time simulation feedback
"""

import simpy
from typing import Any, Dict, List, Tuple, Optional, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import logging

# Import CloudSimPy components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.simulation import Simulation
from core.cluster import Cluster
from core.machine import Machine, MachineConfig
from core.job import Job, Task, TaskInstance
from core.config import JobConfig, TaskConfig
from core.broker import Broker
from core.scheduler import Scheduler
from core.alogrithm import Algorithm
from core.energy_monitor import EnergyMonitor, EnergyProfile, EnergyAwareMachine, MachineState


@dataclass
class TaskInfo:
    """Simplified task information"""
    id: int
    workflow_id: int
    length_mi: int  # Million Instructions
    memory_mb: int
    dependencies: List[int]  # Parent task IDs
    
    # State
    is_ready: bool = False
    is_scheduled: bool = False
    is_completed: bool = False
    assigned_machine: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None


@dataclass
class MachineInfo:
    """Simplified machine information"""
    id: int
    cores: int
    cpu_mips: int
    memory_mb: int
    idle_power_w: float
    peak_power_w: float
    
    # State
    is_busy: bool = False
    current_task: Optional[int] = None
    total_energy_wh: float = 0.0


class DRLSchedulingAlgorithm(Algorithm):
    """Simplified DRL algorithm for CloudSimPy"""
    
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.pending_decisions = []
    
    def __call__(self, cluster, clock):
        """Execute pending scheduling decisions"""
        while self.pending_decisions:
            task_id, machine_id = self.pending_decisions.pop(0)
            self.gym_env._execute_scheduling_decision(task_id, machine_id, clock)
    
    def add_decision(self, task_id: int, machine_id: int):
        """Add a scheduling decision from DRL agent"""
        self.pending_decisions.append((task_id, machine_id))


class EnhancedCloudSimPyGymEnvironment(gym.Env):
    """
    Enhanced CloudSimPy Gym Environment
    
    Key Features:
    - Proper CloudSimPy simulation integration
    - Comprehensive energy monitoring with idle power
    - Simplified, clean interface
    - Real-time simulation feedback
    - Accurate makespan and energy calculations
    """
    
    def __init__(self, dataset=None, dataset_generator: Callable = None,
                 max_episode_steps: int = 500, simulation_time_limit: float = 10000.0):
        super().__init__()
        
        self.dataset = dataset
        self.dataset_generator = dataset_generator
        self.max_episode_steps = max_episode_steps
        self.simulation_time_limit = simulation_time_limit
        
        # Environment state
        self.tasks: List[TaskInfo] = []
        self.machines: List[MachineInfo] = []
        self.task_dependencies: Dict[int, List[int]] = {}
        
        # CloudSimPy components
        self.simpy_env: Optional[simpy.Environment] = None
        self.energy_monitor: Optional[EnergyMonitor] = None
        self.energy_machines: List[EnergyAwareMachine] = []
        self.simulation: Optional[Simulation] = None
        self.algorithm: Optional[DRLSchedulingAlgorithm] = None
        
        # Episode tracking
        self.current_step = 0
        self.simulation_time = 0.0
        self.episode_complete = False
        
        # Metrics
        self.makespan = 0.0
        self.total_energy_wh = 0.0
        self.tasks_completed = 0
        
        # Spaces (will be updated in reset)
        self.action_space = spaces.Discrete(1000)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(100,), dtype=np.float32
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and initialize simulation"""
        super().reset(seed=seed, options=options)
        
        # Get dataset
        if self.dataset_generator:
            dataset = self.dataset_generator(seed)
        else:
            dataset = self.dataset
        
        if dataset is None:
            raise ValueError("No dataset provided")
        
        # Setup from dataset
        self._setup_from_dataset(dataset)
        
        # Initialize CloudSimPy simulation
        self._initialize_simulation()
        
        # Reset episode state
        self.current_step = 0
        self.simulation_time = 0.0
        self.episode_complete = False
        self.makespan = 0.0
        self.total_energy_wh = 0.0
        self.tasks_completed = 0
        
        # Update spaces
        self._update_spaces()
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step"""
        self.current_step += 1
        
        # Decode action
        task_id, machine_id = self._decode_action(action)
        
        # Validate action
        if not self._is_valid_action(task_id, machine_id):
            return self._get_observation(), -100.0, False, True, {"error": "Invalid action"}
        
        # Add decision to algorithm
        self.algorithm.add_decision(task_id, machine_id)
        
        # Run simulation step
        self._run_simulation_step()
        
        # Update metrics
        self._update_metrics()
        
        # Check if done
        done = self._check_episode_complete()
        
        # Calculate reward
        reward = self._calculate_reward(done)
        
        info = {}
        if done:
            info = self._get_episode_info()
        
        return self._get_observation(), reward, False, done, info
    
    def _setup_from_dataset(self, dataset):
        """Setup environment from dataset"""
        # Create machines from VMs and hosts
        host_map = {h.id: h for h in dataset.hosts}
        self.machines = []
        
        for vm in dataset.vms:
            host = host_map[vm.host_id]
            machine = MachineInfo(
                id=len(self.machines),
                cores=host.cores,
                cpu_mips=vm.cpu_speed_mips,
                memory_mb=vm.memory_mb,
                idle_power_w=host.power_idle_watt,
                peak_power_w=host.power_peak_watt
            )
            self.machines.append(machine)
        
        # Create tasks from workflows
        self.tasks = []
        original_to_new_id = {}
        
        for workflow in dataset.workflows:
            for task in workflow.tasks:
                new_id = len(self.tasks)
                original_to_new_id[task.id] = new_id
                
                task_info = TaskInfo(
                    id=new_id,
                    workflow_id=task.workflow_id,
                    length_mi=task.length,
                    memory_mb=task.req_memory_mb,
                    dependencies=[]  # Will be filled below
                )
                self.tasks.append(task_info)
        
        # Setup dependencies
        self.task_dependencies = {}
        
        # Initialize all task dependencies
        for task in self.tasks:
            self.task_dependencies[task.id] = []
        
        # Fill dependencies
        for workflow in dataset.workflows:
            for task in workflow.tasks:
                new_id = original_to_new_id[task.id]
                
                # Add parent dependencies
                if hasattr(task, 'child_ids') and task.child_ids:
                    for child_id in task.child_ids:
                        if child_id in original_to_new_id:
                            child_new_id = original_to_new_id[child_id]
                            self.task_dependencies[child_new_id].append(new_id)
                            self.tasks[child_new_id].dependencies.append(new_id)
        
        # Mark initially ready tasks
        for task in self.tasks:
            if not self.task_dependencies.get(task.id, []):
                task.is_ready = True
    
    def _initialize_simulation(self):
        """Initialize CloudSimPy simulation with energy monitoring"""
        # Create SimPy environment
        self.simpy_env = simpy.Environment()
        
        # Create energy monitor
        self.energy_monitor = EnergyMonitor(self.simpy_env, enable_logging=False)
        
        # Create energy-aware machines
        self.energy_machines = []
        for machine in self.machines:
            energy_profile = EnergyProfile(
                idle_power_watt=machine.idle_power_w,
                peak_power_watt=machine.peak_power_w
            )
            energy_machine = EnergyAwareMachine(
                machine.id, energy_profile, self.energy_monitor
            )
            self.energy_machines.append(energy_machine)
        
        # Create CloudSimPy cluster (simplified)
        self.cluster = Cluster()
        machine_configs = []
        for machine in self.machines:
            config = MachineConfig(
                cpu_capacity=machine.cores,
                memory_capacity=machine.memory_mb,
                disk_capacity=50000
            )
            machine_configs.append(config)
        self.cluster.add_machines(machine_configs)
        
        # Create algorithm
        self.algorithm = DRLSchedulingAlgorithm(self)
        
        # Create scheduler
        self.scheduler = Scheduler(self.simpy_env, self.algorithm)
        
        # Create broker (simplified - we'll handle job submission manually)
        self.broker = Broker(self.simpy_env, [])
        
        # Create simulation
        self.simulation = Simulation(
            env=self.simpy_env,
            cluster=self.cluster,
            task_broker=self.broker,
            scheduler=self.scheduler,
            event_file=None
        )
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action to (task_id, machine_id)"""
        if len(self.machines) == 0:
            return 0, 0
        
        task_id = action // len(self.machines)
        machine_id = action % len(self.machines)
        
        # Clamp to valid ranges
        task_id = min(task_id, len(self.tasks) - 1)
        machine_id = min(machine_id, len(self.machines) - 1)
        
        return task_id, machine_id
    
    def _is_valid_action(self, task_id: int, machine_id: int) -> bool:
        """Check if action is valid"""
        # Check bounds
        if not (0 <= task_id < len(self.tasks)):
            return False
        if not (0 <= machine_id < len(self.machines)):
            return False
        
        task = self.tasks[task_id]
        machine = self.machines[machine_id]
        
        # Check task state
        if not task.is_ready or task.is_scheduled:
            return False
        
        # Check machine availability
        if machine.is_busy:
            return False
        
        # Check resource compatibility
        if machine.memory_mb < task.memory_mb:
            return False
        
        return True
    
    def _run_simulation_step(self):
        """Run simulation for a small time step"""
        if self.simpy_env:
            try:
                # Run for 1 time unit or until next event
                target_time = self.simpy_env.now + 1.0
                self.simpy_env.run(until=target_time)
                self.simulation_time = self.simpy_env.now
            except simpy.core.EmptySchedule:
                # No more events
                pass
    
    def _execute_scheduling_decision(self, task_id: int, machine_id: int, current_time: float):
        """Execute a scheduling decision"""
        if not self._is_valid_action(task_id, machine_id):
            return
        
        task = self.tasks[task_id]
        machine = self.machines[machine_id]
        energy_machine = self.energy_machines[machine_id]
        
        # Update task state
        task.is_scheduled = True
        task.assigned_machine = machine_id
        task.start_time = current_time
        
        # Calculate execution time
        execution_time = task.length_mi / machine.cpu_mips  # seconds
        task.finish_time = current_time + execution_time
        
        # Update machine state
        machine.is_busy = True
        machine.current_task = task_id
        
        # Start energy monitoring
        energy_machine.start_task(task_id, execution_time, 1.0)
        
        # Schedule task completion
        self.simpy_env.process(self._task_execution_process(task_id, machine_id, execution_time))
    
    def _task_execution_process(self, task_id: int, machine_id: int, duration: float):
        """SimPy process for task execution"""
        yield self.simpy_env.timeout(duration)
        
        # Mark task as completed
        task = self.tasks[task_id]
        machine = self.machines[machine_id]
        energy_machine = self.energy_machines[machine_id]
        
        task.is_completed = True
        machine.is_busy = False
        machine.current_task = None
        
        # Complete energy monitoring
        energy_machine.complete_task(task_id)
        
        # Update ready tasks
        self._update_ready_tasks()
        
        self.tasks_completed += 1
    
    def _update_ready_tasks(self):
        """Update which tasks are ready"""
        for task in self.tasks:
            if not task.is_ready and not task.is_scheduled:
                # Check if all dependencies are completed
                dependencies = self.task_dependencies.get(task.id, [])
                if all(self.tasks[dep_id].is_completed for dep_id in dependencies):
                    task.is_ready = True
    
    def _update_metrics(self):
        """Update episode metrics"""
        if self.energy_monitor:
            self.total_energy_wh = self.energy_monitor.get_total_energy_consumption()
        
        # Update machine energy
        for i, machine in enumerate(self.machines):
            if i < len(self.energy_machines):
                machine.total_energy_wh = self.energy_machines[i].get_energy_consumption()
    
    def _check_episode_complete(self) -> bool:
        """Check if episode is complete"""
        # Max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Simulation time limit
        if self.simulation_time >= self.simulation_time_limit:
            return True
        
        # All tasks completed
        if all(task.is_completed for task in self.tasks):
            self.makespan = self.simulation_time
            return True
        
        return False
    
    def _calculate_reward(self, done: bool) -> float:
        """Calculate reward"""
        if done:
            if all(task.is_completed for task in self.tasks):
                # Success - reward based on efficiency
                makespan_penalty = self.makespan / 1000.0
                energy_penalty = self.total_energy_wh / 10.0
                
                # Weighted combination
                total_penalty = 0.6 * makespan_penalty + 0.4 * energy_penalty
                reward = 1000.0 / (1.0 + total_penalty)
            else:
                # Failure
                reward = -1000.0
        else:
            # Progress reward
            progress = self.tasks_completed / len(self.tasks)
            reward = progress * 10.0 - 0.1  # Small step penalty
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        features = []
        
        # Task features (normalized)
        for task in self.tasks:  # Limit to first 20 tasks
            features.extend([
                task.length_mi / 10000.0,  # Normalized length
                task.memory_mb / 4096.0,   # Normalized memory
                float(task.is_ready),
                float(task.is_scheduled),
                float(task.is_completed),
                len(task.dependencies) / 10.0  # Normalized dependencies
            ])
        

        
        # Machine features (normalized)
        for machine in self.machines:  # Limit to first 10 machines
            features.extend([
                machine.cores / 16.0,      # Normalized cores
                machine.cpu_mips / 5000.0, # Normalized CPU
                machine.memory_mb / 8192.0, # Normalized memory
                float(machine.is_busy),
                machine.total_energy_wh / 100.0,  # Normalized energy
                machine.idle_power_w / 200.0,     # Normalized idle power
                machine.peak_power_w / 500.0      # Normalized peak power
            ])
        
        # Pad if fewer than 10 machines
        while len(features) < 20 * 6 + 10 * 7:
            features.extend([0.0] * 7)
        
        # Global features
        features.extend([
            self.current_step / self.max_episode_steps,
            self.simulation_time / self.simulation_time_limit,
            self.tasks_completed / len(self.tasks),
            self.total_energy_wh / 1000.0
        ])
        
        return np.array(features[:200], dtype=np.float32)  # Fixed size
    
    def _update_spaces(self):
        """Update action and observation spaces"""
        # Action space
        num_actions = len(self.tasks) * len(self.machines)
        self.action_space = spaces.Discrete(max(1, num_actions))
        
        # Observation space (fixed size)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(200,), dtype=np.float32
        )
    
    def _get_episode_info(self) -> Dict[str, Any]:
        """Get episode completion info"""
        energy_metrics = self.energy_monitor.get_energy_efficiency_metrics() if self.energy_monitor else {}
        
        return {
            "makespan": self.makespan,
            "total_energy_wh": self.total_energy_wh,
            "tasks_completed": self.tasks_completed,
            "total_tasks": len(self.tasks),
            "success_rate": self.tasks_completed / len(self.tasks),
            "energy_efficiency": self.tasks_completed / max(self.total_energy_wh, 0.001),
            "episode_length": self.current_step,
            "simulation_time": self.simulation_time,
            "energy_metrics": energy_metrics
        }
    
    def get_detailed_energy_report(self) -> str:
        """Get detailed energy consumption report"""
        if self.energy_monitor:
            return self.energy_monitor.generate_energy_report()
        return "Energy monitoring not available"

