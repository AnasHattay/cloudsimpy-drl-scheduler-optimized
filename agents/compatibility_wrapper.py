"""
Compatibility Wrapper for Existing DRL Agents

This module provides a compatibility layer that allows existing DRL agents
to work with both the new separated environments and the original CloudSimPy-based 
environment without modification. It handles the translation between different 
action and observation formats.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

# Import both old and new environment interfaces
try:
    from simulator.cloudsimpy_gym_env import EnhancedCloudSimPyGymEnvironment as CloudSimPyGymEnvironment

except ImportError:
    CloudSimPyGymEnvironment = None
    EnvAction = None
    EnvObservation = None

# Import new separated environments
try:
    from environments import AbstractSchedulingEnvironment, EnvironmentFactory, create_environment
except ImportError:
    AbstractSchedulingEnvironment = None
    EnvironmentFactory = None
    create_environment = None


class CompatibilityWrapper(gym.Wrapper):
    """
    Wrapper that provides compatibility with existing DRL agents.
    
    This wrapper translates between the new CloudSimPy environment interface
    and the interface expected by existing DRL agents.
    """
    
    def __init__(self, env: CloudSimPyGymEnvironment):
        super().__init__(env)
        self.env = env
        self._setup_compatibility_spaces()
    
    def _setup_compatibility_spaces(self):
        """Setup action and observation spaces for compatibility"""
        # The original environment expects discrete actions
        # We'll map these to (task_id, vm_id) pairs
        self.action_space = spaces.Discrete(1000)  # Will be updated after reset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1000,), dtype=np.float32
        )  # Will be updated after reset
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return compatible observation"""
        obs, info = self.env.reset(**kwargs)
        
        # Update action space based on actual environment size
        if self.env.state:
            num_tasks = len(self.env.state.static_state.tasks)
            num_vms = len(self.env.state.static_state.vms)
            self.action_space = spaces.Discrete(num_tasks * num_vms)
            
            # Update observation space
            obs_array = obs.to_array()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_array.shape, dtype=np.float32
            )
        
        return self._convert_observation(obs), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with action conversion"""
        # Convert discrete action to (task_id, vm_id) pair
        env_action = self._convert_action(action)
        
        if env_action is None:
            # Invalid action
            obs = EnvObservation(self.env.state)
            return self._convert_observation(obs), -1000.0, True, False, {"error": "Invalid action"}
        
        # Execute action in underlying environment
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        
        return self._convert_observation(obs), reward, terminated, truncated, info
    
    def _convert_action(self, action: int) -> Optional[EnvAction]:
        """Convert discrete action to environment action"""
        if self.env.state is None:
            return None
        
        num_tasks = len(self.env.state.static_state.tasks)
        num_vms = len(self.env.state.static_state.vms)
        
        if action >= num_tasks * num_vms:
            return None
        
        task_id = action // num_vms
        vm_id = action % num_vms
        
        return EnvAction(task_id=task_id, vm_id=vm_id)
    
    def _convert_observation(self, obs: EnvObservation) -> np.ndarray:
        """Convert environment observation to numpy array"""
        return obs.to_array()


class LegacyEnvironmentAdapter:
    """
    Adapter that provides the exact interface expected by legacy DRL agents.
    
    This adapter mimics the original CloudSchedulingGymEnvironment interface
    while using CloudSimPy internally.
    """
    
    def __init__(self, dataset=None, dataset_args=None):
        # Create the underlying CloudSimPy environment
        if dataset is not None:
            self.env = CloudSimPyGymEnvironment(dataset=dataset)
        elif dataset_args is not None:
            from scheduler.dataset_generator.gen_dataset import generate_dataset
            dataset_generator = lambda seed: generate_dataset(seed, dataset_args)
            self.env = CloudSimPyGymEnvironment(dataset_generator=dataset_generator)
        else:
            raise ValueError("Either dataset or dataset_args must be provided")
        
        # Wrap with compatibility layer
        self.wrapped_env = CompatibilityWrapper(self.env)
        
        # Expose the same interface as the original environment
        self.action_space = self.wrapped_env.action_space
        self.observation_space = self.wrapped_env.observation_space
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset environment - compatible with original interface"""
        return self.wrapped_env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Step environment - compatible with original interface"""
        return self.wrapped_env.step(action)
    
    def close(self):
        """Close environment"""
        if hasattr(self.wrapped_env, 'close'):
            self.wrapped_env.close()
    
    def render(self, mode='human'):
        """Render environment (placeholder)"""
        pass
    
    @property
    def state(self):
        """Access to internal state for compatibility"""
        return self.env.state


class TaskMapper:
    """
    Task mapper for compatibility with existing code.
    
    This class provides the same interface as the original TaskMapper
    while working with the new CloudSimPy-based data structures.
    """
    
    def __init__(self, tasks: List):
        self.tasks = tasks
        self._id_mapping = {}
        self._reverse_mapping = {}
        
        # Create mapping between original and mapped IDs
        for i, task in enumerate(tasks):
            self._id_mapping[task.id] = i
            self._reverse_mapping[i] = (task.workflow_id, task.id)
    
    def map_tasks(self):
        """Map tasks to sequential IDs"""
        mapped_tasks = []
        for i, task in enumerate(self.tasks):
            # Create a mapped task with sequential ID
            mapped_task = type('MappedTask', (), {
                'id': i,
                'workflow_id': task.workflow_id,
                'length': task.length,
                'req_cores': task.req_cores,
                'child_ids': [self._id_mapping.get(child_id, child_id) for child_id in task.child_ids]
            })()
            mapped_tasks.append(mapped_task)
        
        return mapped_tasks
    
    def unmap_id(self, mapped_id: int) -> Tuple[int, int]:
        """Unmap a sequential ID back to (workflow_id, original_id)"""
        return self._reverse_mapping.get(mapped_id, (0, mapped_id))


def create_legacy_environment(dataset=None, dataset_args=None):
    """
    Factory function to create a legacy-compatible environment.
    
    This function can be used as a drop-in replacement for the original
    CloudSchedulingGymEnvironment constructor.
    """
    return LegacyEnvironmentAdapter(dataset=dataset, dataset_args=dataset_args)


# Utility functions for energy calculation compatibility
def active_energy_consumption_per_mi(vm) -> float:
    """Calculate active energy consumption per million instructions"""
    if hasattr(vm, 'power_peak_watt') and hasattr(vm, 'power_idle_watt'):
        return (vm.power_peak_watt - vm.power_idle_watt) / vm.cpu_speed_mips
    else:
        # Default calculation if power information is not available
        return 0.001  # Default energy per MI


def is_suitable(vm, task) -> bool:
    """Check if a VM is suitable for a task"""
    return vm.memory_mb >= task.req_memory_mb


# Compatibility imports for existing code
class VmAssignmentDto:
    """VM assignment data transfer object for compatibility"""
    
    def __init__(self, vm_id: int, workflow_id: int, task_id: int):
        self.vm_id = vm_id
        self.workflow_id = workflow_id
        self.task_id = task_id


# Export the main compatibility interface
__all__ = [
    'CompatibilityWrapper',
    'LegacyEnvironmentAdapter', 
    'TaskMapper',
    'create_legacy_environment',
    'active_energy_consumption_per_mi',
    'is_suitable',
    'VmAssignmentDto'
]



def create_legacy_environment(dataset=None, dataset_args=None, env_type="testing"):
    """
    Enhanced factory function to create a legacy-compatible environment.
    
    This function can create either separated environments or the original
    CloudSimPy environment based on availability and configuration.
    
    Args:
        dataset: Pre-generated dataset (for original compatibility)
        dataset_args: Dataset generation arguments (for new separated environments)
        env_type: Environment type ("training" or "testing")
        
    Returns:
        Environment instance compatible with existing agents
    """
    # Try to use new separated environments first
    if create_environment is not None and dataset_args is not None:
        try:
            return create_environment(env_type, dataset_args)
        except Exception as e:
            print(f"Warning: Could not create separated environment: {e}")
            print("Falling back to original CloudSimPy environment...")
    
    # Fall back to original implementation
    if dataset is not None:
        return LegacyEnvironmentAdapter(dataset=dataset)
    elif dataset_args is not None:
        return LegacyEnvironmentAdapter(dataset_args=dataset_args)
    else:
        raise ValueError("Either dataset or dataset_args must be provided")


def create_training_environment_legacy(dataset_args, **kwargs):
    """
    Create a training environment with legacy compatibility.
    
    Args:
        dataset_args: Dataset generation arguments
        **kwargs: Additional configuration parameters
        
    Returns:
        Training environment instance
    """
    return create_legacy_environment(dataset_args=dataset_args, env_type="training", **kwargs)


def create_testing_environment_legacy(dataset_args, **kwargs):
    """
    Create a testing environment with legacy compatibility.
    
    Args:
        dataset_args: Dataset generation arguments
        **kwargs: Additional configuration parameters
        
    Returns:
        Testing environment instance
    """
    return create_legacy_environment(dataset_args=dataset_args, env_type="testing", **kwargs)


# Export the main compatibility interface with enhanced functionality
__all__ = [
    'CompatibilityWrapper',
    'LegacyEnvironmentAdapter', 
    'TaskMapper',
    'create_legacy_environment',
    'create_training_environment_legacy',
    'create_testing_environment_legacy',
    'active_energy_consumption_per_mi',
    'is_suitable',
    'VmAssignmentDto'
]

