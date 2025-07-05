"""
Enhanced Testing Environment for CloudSimPy DRL Scheduler

This module implements an enhanced testing environment that builds upon
the existing CloudSimPy integration with additional instrumentation,
metrics collection, and analysis capabilities.
"""

import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, deque
from gymnasium import spaces
import logging

from .abstract_env import AbstractSchedulingEnvironment, EnvironmentConfig
from simulator.cloudsimpy_gym_env import EnhancedCloudSimPyGymEnvironment as CloudSimPyGymEnvironment
from agents.compatibility_wrapper import CompatibilityWrapper

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects detailed metrics during testing"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.actions_taken = []
        self.rewards_received = []
        self.state_transitions = []
        self.resource_usage = defaultdict(list)
        self.energy_consumption = defaultdict(list)
        self.task_assignments = []
        self.vm_utilization = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.start_time = time.time()
        self.step_times = []
    
    def record_action(self, action, state, timestamp=None):
        """Record action and current state"""
        if timestamp is None:
            timestamp = time.time()
        
        self.actions_taken.append({
            "action": action,
            "timestamp": timestamp,
            "state_summary": self._summarize_state(state) if state else {}
        })
    
    def record_step_result(self, obs, reward, terminated, info):
        """Record step results"""
        self.rewards_received.append(reward)
        
        # Record resource utilization if available
        if "resource_utilization" in info:
            for vm_id, utilization in info["resource_utilization"].items():
                self.resource_usage[vm_id].append(utilization)
        
        # Record energy consumption if available
        if "energy_consumption" in info:
            for vm_id, energy in info["energy_consumption"].items():
                self.energy_consumption[vm_id].append(energy)
        
        # Record VM utilization
        if "vm_utilization" in info:
            for vm_id, util in info["vm_utilization"].items():
                self.vm_utilization[vm_id].append(util)
        
        # Record memory usage
        if "memory_usage" in info:
            for vm_id, memory in info["memory_usage"].items():
                self.memory_usage[vm_id].append(memory)
    
    def record_step_time(self, step_time):
        """Record step execution time"""
        self.step_times.append(step_time)
    
    def record_task_assignment(self, task_id, vm_id, timestamp=None):
        """Record task assignment"""
        if timestamp is None:
            timestamp = time.time()
        
        self.task_assignments.append({
            "task_id": task_id,
            "vm_id": vm_id,
            "timestamp": timestamp
        })
    
    def _summarize_state(self, state) -> Dict[str, Any]:
        """Create a summary of the current state"""
        if hasattr(state, 'static_state'):
            return {
                "num_tasks": len(state.static_state.tasks),
                "num_vms": len(state.static_state.vms),
                "completed_tasks": sum(1 for ts in state.task_states if ts.completion_time > 0),
                "assigned_tasks": sum(1 for ts in state.task_states if ts.assigned_vm_id is not None)
            }
        return {}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            "total_actions": len(self.actions_taken),
            "total_reward": sum(self.rewards_received),
            "avg_reward": np.mean(self.rewards_received) if self.rewards_received else 0,
            "avg_step_time": np.mean(self.step_times) if self.step_times else 0,
            "total_assignments": len(self.task_assignments),
            "elapsed_time": time.time() - self.start_time
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        summary = self.get_current_metrics()
        
        # Add detailed statistics
        if self.rewards_received:
            summary.update({
                "reward_std": np.std(self.rewards_received),
                "reward_min": np.min(self.rewards_received),
                "reward_max": np.max(self.rewards_received)
            })
        
        if self.step_times:
            summary.update({
                "step_time_std": np.std(self.step_times),
                "step_time_min": np.min(self.step_times),
                "step_time_max": np.max(self.step_times)
            })
        
        # Resource utilization statistics
        if self.resource_usage:
            avg_utilization = {}
            for vm_id, utilizations in self.resource_usage.items():
                avg_utilization[vm_id] = np.mean(utilizations)
            summary["avg_resource_utilization"] = avg_utilization
        
        # Energy consumption statistics
        if self.energy_consumption:
            total_energy = {}
            for vm_id, energies in self.energy_consumption.items():
                total_energy[vm_id] = sum(energies)
            summary["total_energy_by_vm"] = total_energy
        
        return summary


class PerformanceProfiler:
    """Profiles performance characteristics of the environment"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset profiling data"""
        self.reset_times = []
        self.step_times = []
        self.action_processing_times = []
        self.reward_calculation_times = []
        self.observation_generation_times = []
        self.memory_usage = []
        self.start_time = time.time()
    
    def record_reset_time(self, reset_time):
        """Record environment reset time"""
        self.reset_times.append(reset_time)
    
    def record_step_time(self, step_time):
        """Record step execution time"""
        self.step_times.append(step_time)
    
    def record_action_processing_time(self, processing_time):
        """Record action processing time"""
        self.action_processing_times.append(processing_time)
    
    def record_reward_calculation_time(self, calculation_time):
        """Record reward calculation time"""
        self.reward_calculation_times.append(calculation_time)
    
    def record_observation_generation_time(self, generation_time):
        """Record observation generation time"""
        self.observation_generation_times.append(generation_time)
    
    def get_current_profile(self) -> Dict[str, Any]:
        """Get current performance profile"""
        return {
            "avg_reset_time": np.mean(self.reset_times) if self.reset_times else 0,
            "avg_step_time": np.mean(self.step_times) if self.step_times else 0,
            "avg_action_processing_time": np.mean(self.action_processing_times) if self.action_processing_times else 0,
            "total_runtime": time.time() - self.start_time,
            "total_steps": len(self.step_times),
            "total_resets": len(self.reset_times)
        }
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get detailed performance analysis"""
        analysis = self.get_current_profile()
        
        # Add statistical analysis
        if self.step_times:
            analysis.update({
                "step_time_std": np.std(self.step_times),
                "step_time_percentiles": {
                    "p50": np.percentile(self.step_times, 50),
                    "p90": np.percentile(self.step_times, 90),
                    "p95": np.percentile(self.step_times, 95),
                    "p99": np.percentile(self.step_times, 99)
                }
            })
        
        if self.reset_times:
            analysis.update({
                "reset_time_std": np.std(self.reset_times),
                "reset_time_percentiles": {
                    "p50": np.percentile(self.reset_times, 50),
                    "p90": np.percentile(self.reset_times, 90),
                    "p95": np.percentile(self.reset_times, 95)
                }
            })
        
        return analysis


class BaselineComparator:
    """Compares agent performance against baseline algorithms"""
    
    def __init__(self):
        self.baseline_results = {}
        self.agent_results = []
    
    def add_baseline_result(self, algorithm_name: str, result: Dict[str, Any]):
        """Add baseline algorithm result"""
        self.baseline_results[algorithm_name] = result
    
    def add_agent_result(self, result: Dict[str, Any]):
        """Add agent result"""
        self.agent_results.append(result)
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison between agent and baselines"""
        if not self.agent_results:
            return {"error": "No agent results available"}
        
        # Calculate agent statistics
        agent_makespans = [r.get("makespan", 0) for r in self.agent_results]
        agent_energies = [r.get("total_energy", 0) for r in self.agent_results]
        agent_rewards = [r.get("reward", 0) for r in self.agent_results]
        
        agent_stats = {
            "makespan": {
                "mean": np.mean(agent_makespans),
                "std": np.std(agent_makespans),
                "min": np.min(agent_makespans),
                "max": np.max(agent_makespans)
            },
            "energy": {
                "mean": np.mean(agent_energies),
                "std": np.std(agent_energies),
                "min": np.min(agent_energies),
                "max": np.max(agent_energies)
            },
            "reward": {
                "mean": np.mean(agent_rewards),
                "std": np.std(agent_rewards),
                "min": np.min(agent_rewards),
                "max": np.max(agent_rewards)
            }
        }
        
        # Compare with baselines
        comparisons = {}
        for baseline_name, baseline_result in self.baseline_results.items():
            comparison = {}
            
            if "makespan" in baseline_result:
                improvement = (baseline_result["makespan"] - agent_stats["makespan"]["mean"]) / baseline_result["makespan"]
                comparison["makespan_improvement"] = improvement
            
            if "total_energy" in baseline_result:
                improvement = (baseline_result["total_energy"] - agent_stats["energy"]["mean"]) / baseline_result["total_energy"]
                comparison["energy_improvement"] = improvement
            
            comparisons[baseline_name] = comparison
        
        return {
            "agent_stats": agent_stats,
            "baseline_comparisons": comparisons,
            "num_episodes": len(self.agent_results)
        }


class EnhancedTestingEnvironment(AbstractSchedulingEnvironment):
    """Enhanced CloudSimPy environment for comprehensive testing"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        
        # Create underlying CloudSimPy environment
        self.cloudsimpy_env = None
        self.wrapped_env = None
        
        # Initialize monitoring components
        self.metrics_collector = MetricsCollector()
        self.performance_profiler = PerformanceProfiler()
        self.baseline_comparator = BaselineComparator()
        
        # Configuration
        self.detailed_logging = config.debug_mode
        self.collect_detailed_metrics = True
        
        # Initialize logger
        if self.detailed_logging:
            logging.basicConfig(level=logging.DEBUG)
    
    def _setup_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        """Setup observation and action spaces"""
        # Spaces will be set by the underlying CloudSimPy environment
        return self.observation_space, self.action_space
    
    def _reset_implementation(self, seed: Optional[int], options: Optional[Dict[str, Any]]):
        """Reset with enhanced monitoring"""
        start_time = time.time()
        
        # Create CloudSimPy environment if not exists
        if self.cloudsimpy_env is None:
            self._initialize_cloudsimpy_environment()
        
        # Reset underlying CloudSimPy environment
        obs, info = self.wrapped_env.reset(seed=seed, options=options)
        
        # Update our spaces to match the underlying environment
        self.observation_space = self.wrapped_env.observation_space
        self.action_space = self.wrapped_env.action_space
        
        # Initialize metrics collection
        self.metrics_collector.reset()
        self.performance_profiler.reset()
        
        # Record reset time
        reset_time = time.time() - start_time
        self.performance_profiler.record_reset_time(reset_time)
        
        # Enhanced info
        enhanced_info = self._enhance_info(info)
        
        if self.detailed_logging:
            logger.debug(f"Environment reset completed in {reset_time:.4f}s")
        
        return obs, enhanced_info
    
    def _step_implementation(self, action):
        """Step with detailed instrumentation"""
        step_start_time = time.time()
        
        # Record action for analysis
        self.metrics_collector.record_action(action, self.wrapped_env.env.state)
        
        # Time action processing
        action_start_time = time.time()
        
        # Execute step in CloudSimPy
        obs, reward, terminated, truncated, info = self.wrapped_env.step(action)
        
        action_processing_time = time.time() - action_start_time
        self.performance_profiler.record_action_processing_time(action_processing_time)
        
        # Collect detailed metrics
        step_time = time.time() - step_start_time
        self.performance_profiler.record_step_time(step_time)
        self.metrics_collector.record_step_time(step_time)
        self.metrics_collector.record_step_result(obs, reward, terminated, info)
        
        # Enhance info with additional metrics
        enhanced_info = self._enhance_info(info)
        
        if self.detailed_logging:
            logger.debug(f"Step completed in {step_time:.4f}s, reward: {reward:.3f}")
        
        return obs, reward, terminated, truncated, enhanced_info
    
    def _initialize_cloudsimpy_environment(self):
        """Initialize the underlying CloudSimPy environment"""
        try:
            # Import dataset generation
            from scheduler.dataset_generator.gen_dataset import generate_dataset
            
            # Create dataset generator with proper parameter mapping
            def dataset_generator(seed):
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
            
            # Create CloudSimPy environment
            self.cloudsimpy_env = CloudSimPyGymEnvironment(dataset_generator=dataset_generator)
            
            # Wrap with compatibility layer
            self.wrapped_env = CompatibilityWrapper(self.cloudsimpy_env)
            
            if self.detailed_logging:
                logger.info("CloudSimPy environment initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize CloudSimPy environment: {e}")
            raise
    
    def _enhance_info(self, base_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add detailed metrics to info dictionary"""
        enhanced_info = base_info.copy()
        
        # Add performance metrics
        enhanced_info.update({
            "detailed_metrics": self.metrics_collector.get_current_metrics(),
            "performance_profile": self.performance_profiler.get_current_profile(),
        })
        
        # Add resource utilization if available
        if hasattr(self.wrapped_env.env, 'state') and self.wrapped_env.env.state:
            enhanced_info.update({
                "resource_utilization": self._calculate_resource_utilization(),
                "energy_breakdown": self._calculate_energy_breakdown(),
                "vm_utilization": self._calculate_vm_utilization(),
                "memory_usage": self._calculate_memory_usage(),
                "task_completion_stats": self._calculate_task_completion_stats()
            })
        
        return enhanced_info
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        if not hasattr(self.wrapped_env.env, 'state') or not self.wrapped_env.env.state:
            return {}
        
        state = self.wrapped_env.env.state
        utilization = {}
        
        for i, vm in enumerate(state.static_state.vms):
            vm_state = state.vm_states[i] if i < len(state.vm_states) else None
            if vm_state:
                # Calculate utilization based on assigned tasks
                assigned_tasks = sum(1 for ts in state.task_states if ts.assigned_vm_id == i)
                utilization[vm.id] = assigned_tasks / vm.cores if vm.cores > 0 else 0
        
        return utilization
    
    def _calculate_energy_breakdown(self) -> Dict[str, float]:
        """Calculate energy consumption breakdown"""
        if not hasattr(self.wrapped_env.env, 'state') or not self.wrapped_env.env.state:
            return {}
        
        state = self.wrapped_env.env.state
        energy_breakdown = {}
        
        for i, vm in enumerate(state.static_state.vms):
            vm_state = state.vm_states[i] if i < len(state.vm_states) else None
            if vm_state:
                energy_breakdown[vm.id] = vm_state.total_energy
        
        return energy_breakdown
    
    def _calculate_vm_utilization(self) -> Dict[str, float]:
        """Calculate VM utilization metrics"""
        if not hasattr(self.wrapped_env.env, 'state') or not self.wrapped_env.env.state:
            return {}
        
        state = self.wrapped_env.env.state
        vm_utilization = {}
        
        for i, vm in enumerate(state.static_state.vms):
            vm_state = state.vm_states[i] if i < len(state.vm_states) else None
            if vm_state:
                # Calculate time-based utilization
                total_time = max(vm_state.completion_time, 1.0)
                busy_time = sum(
                    ts.completion_time - ts.start_time 
                    for ts in state.task_states 
                    if ts.assigned_vm_id == i and ts.completion_time > 0
                )
                vm_utilization[vm.id] = busy_time / total_time
        
        return vm_utilization
    
    def _calculate_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage statistics"""
        if not hasattr(self.wrapped_env.env, 'state') or not self.wrapped_env.env.state:
            return {}
        
        state = self.wrapped_env.env.state
        memory_usage = {}
        
        for i, vm in enumerate(state.static_state.vms):
            # Calculate memory usage based on assigned tasks
            used_memory = sum(
                task.req_memory_mb 
                for j, task in enumerate(state.static_state.tasks)
                if j < len(state.task_states) and state.task_states[j].assigned_vm_id == i
            )
            memory_usage[vm.id] = used_memory / vm.memory_mb if vm.memory_mb > 0 else 0
        
        return memory_usage
    
    def _calculate_task_completion_stats(self) -> Dict[str, Any]:
        """Calculate task completion statistics"""
        if not hasattr(self.wrapped_env.env, 'state') or not self.wrapped_env.env.state:
            return {}
        
        state = self.wrapped_env.env.state
        completed_tasks = sum(1 for ts in state.task_states if ts.completion_time > 0)
        total_tasks = len(state.task_states)
        
        completion_times = [ts.completion_time for ts in state.task_states if ts.completion_time > 0]
        
        stats = {
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
        }
        
        if completion_times:
            stats.update({
                "avg_completion_time": np.mean(completion_times),
                "max_completion_time": np.max(completion_times),
                "min_completion_time": np.min(completion_times)
            })
        
        return stats
    
    def _calculate_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Calculate reward (delegated to underlying environment)"""
        return 0.0  # Not used directly
    
    def _is_terminal(self, state: Any) -> bool:
        """Check if terminal (delegated to underlying environment)"""
        return False  # Not used directly
    
    def _get_info(self, state: Any) -> Dict[str, Any]:
        """Get info (delegated to underlying environment)"""
        return {}  # Not used directly
    
    def _validate_action(self, action: Any) -> bool:
        """Validate action (delegated to underlying environment)"""
        if self.wrapped_env:
            # Check if wrapped environment has validation method
            if hasattr(self.wrapped_env, '_validate_action'):
                return self.wrapped_env._validate_action(action)
            elif hasattr(self.wrapped_env, 'action_space'):
                # Basic validation using action space
                return self.wrapped_env.action_space.contains(action)
        return True
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "metrics_summary": self.metrics_collector.get_summary(),
            "performance_analysis": self.performance_profiler.get_analysis(),
            "baseline_comparison": self.baseline_comparator.get_comparison(),
            "environment_config": {
                "env_type": self.config.env_type.value,
                "dataset_args": self.config.dataset_args,
                "reward_weights": self.config.reward_weights,
                "performance_mode": self.config.performance_mode,
                "debug_mode": self.config.debug_mode
            }
        }
    
    def add_baseline_result(self, algorithm_name: str, result: Dict[str, Any]):
        """Add baseline algorithm result for comparison"""
        self.baseline_comparator.add_baseline_result(algorithm_name, result)
    
    def add_agent_result(self, result: Dict[str, Any]):
        """Add agent result for comparison"""
        self.baseline_comparator.add_agent_result(result)
    
    def close(self):
        """Close environment and clean up resources"""
        if self.wrapped_env:
            self.wrapped_env.close()
        
        if self.detailed_logging:
            logger.info("Enhanced testing environment closed")

