"""
Comprehensive Scenario Generator for DRL vs Heuristic Evaluation

This module generates various types of DAG scenarios and resource conditions
to thoroughly evaluate scheduling algorithms across different challenges.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum
import json

from dataset_generator.core.gen_dataset import generate_dataset
from dataset_generator.core.models import Dataset, Task, Vm, Host


class DAGType(Enum):
    """Different types of DAG structures for evaluation"""
    LINEAR = "linear"           # Sequential tasks (high critical path)
    PARALLEL = "parallel"       # Independent tasks (low critical path)
    DIAMOND = "diamond"         # Fork-join pattern
    TREE = "tree"              # Hierarchical structure
    COMPLEX = "complex"        # Mixed dependencies
    WIDE_SHALLOW = "wide_shallow"  # Many parallel tasks, few levels
    NARROW_DEEP = "narrow_deep"    # Few parallel tasks, many levels


class ResourceCondition(Enum):
    """Different resource availability scenarios"""
    ABUNDANT = "abundant"       # Resources >> workload
    BALANCED = "balanced"       # Resources ≈ workload
    CONSTRAINED = "constrained" # Resources < workload
    BOTTLENECK = "bottleneck"   # Severe resource limitation


@dataclass
class ScenarioConfig:
    """Configuration for a specific evaluation scenario"""
    name: str
    description: str
    dag_type: DAGType
    resource_condition: ResourceCondition
    
    # DAG parameters
    task_count: int
    workflow_count: int
    dependency_ratio: float  # Ratio of dependencies to tasks
    critical_path_length: int
    parallelism_factor: float  # Average parallel tasks per level
    
    # Resource parameters
    host_count: int
    vm_count: int
    cpu_capacity_range: Tuple[int, int]  # (min, max) MIPS
    memory_capacity_range: Tuple[int, int]  # (min, max) MB
    
    # Task parameters
    task_length_range: Tuple[int, int]  # (min, max) MI
    memory_requirement_range: Tuple[int, int]  # (min, max) MB
    task_heterogeneity: float  # Variance in task requirements
    
    # Power parameters
    power_idle_range: Tuple[int, int]  # (min, max) watts
    power_peak_range: Tuple[int, int]  # (min, max) watts
    
    # Evaluation parameters
    seed: int = 42
    replications: int = 10


class ScenarioGenerator:
    """Generates evaluation scenarios for comprehensive testing"""
    
    def __init__(self):
        self.scenarios = []
        self._initialize_standard_scenarios()
    
    def _initialize_standard_scenarios(self):
        """Initialize a comprehensive set of standard evaluation scenarios"""
        
        # 1. Linear DAG scenarios (high critical path dependency)
        self.scenarios.extend([
            ScenarioConfig(
                name="linear_abundant",
                description="Linear DAG with abundant resources - tests critical path optimization",
                dag_type=DAGType.LINEAR,
                resource_condition=ResourceCondition.ABUNDANT,
                task_count=10,
                workflow_count=3,
                dependency_ratio=0.9,
                critical_path_length=10,
                parallelism_factor=1.0,
                host_count=4,
                vm_count=8,
                cpu_capacity_range=(5000, 10000),
                memory_capacity_range=(4096, 8192),
                task_length_range=(1000, 5000),
                memory_requirement_range=(512, 1024),
                task_heterogeneity=0.3,
                power_idle_range=(100, 150),
                power_peak_range=(300, 500)
            ),
            ScenarioConfig(
                name="linear_constrained",
                description="Linear DAG with constrained resources - tests resource allocation under pressure",
                dag_type=DAGType.LINEAR,
                resource_condition=ResourceCondition.CONSTRAINED,
                task_count=12,
                workflow_count=4,
                dependency_ratio=0.9,
                critical_path_length=12,
                parallelism_factor=1.0,
                host_count=2,
                vm_count=3,
                cpu_capacity_range=(2000, 4000),
                memory_capacity_range=(1024, 2048),
                task_length_range=(2000, 8000),
                memory_requirement_range=(512, 1536),
                task_heterogeneity=0.4,
                power_idle_range=(80, 120),
                power_peak_range=(250, 400)
            )
        ])
        
        # 2. Parallel DAG scenarios (low critical path, high parallelism)
        self.scenarios.extend([
            ScenarioConfig(
                name="parallel_abundant",
                description="Parallel DAG with abundant resources - tests parallel execution optimization",
                dag_type=DAGType.PARALLEL,
                resource_condition=ResourceCondition.ABUNDANT,
                task_count=15,
                workflow_count=2,
                dependency_ratio=0.1,
                critical_path_length=3,
                parallelism_factor=5.0,
                host_count=6,
                vm_count=12,
                cpu_capacity_range=(4000, 8000),
                memory_capacity_range=(2048, 4096),
                task_length_range=(1500, 4000),
                memory_requirement_range=(256, 512),
                task_heterogeneity=0.2,
                power_idle_range=(120, 180),
                power_peak_range=(350, 600)
            ),
            ScenarioConfig(
                name="parallel_bottleneck",
                description="Parallel DAG with severe resource bottleneck - tests load balancing",
                dag_type=DAGType.PARALLEL,
                resource_condition=ResourceCondition.BOTTLENECK,
                task_count=20,
                workflow_count=3,
                dependency_ratio=0.1,
                critical_path_length=2,
                parallelism_factor=10.0,
                host_count=1,
                vm_count=2,
                cpu_capacity_range=(1500, 2500),
                memory_capacity_range=(512, 1024),
                task_length_range=(3000, 10000),
                memory_requirement_range=(256, 768),
                task_heterogeneity=0.5,
                power_idle_range=(60, 100),
                power_peak_range=(200, 350)
            )
        ])
        
        # 3. Diamond DAG scenarios (fork-join patterns)
        self.scenarios.extend([
            ScenarioConfig(
                name="diamond_balanced",
                description="Diamond DAG with balanced resources - tests fork-join optimization",
                dag_type=DAGType.DIAMOND,
                resource_condition=ResourceCondition.BALANCED,
                task_count=8,
                workflow_count=4,
                dependency_ratio=0.6,
                critical_path_length=4,
                parallelism_factor=2.0,
                host_count=3,
                vm_count=6,
                cpu_capacity_range=(3000, 6000),
                memory_capacity_range=(1536, 3072),
                task_length_range=(2000, 6000),
                memory_requirement_range=(384, 1024),
                task_heterogeneity=0.3,
                power_idle_range=(90, 140),
                power_peak_range=(280, 450)
            )
        ])
        
        # 4. Complex heterogeneous scenarios
        self.scenarios.extend([
            ScenarioConfig(
                name="complex_heterogeneous",
                description="Complex DAG with heterogeneous tasks and mixed resource conditions",
                dag_type=DAGType.COMPLEX,
                resource_condition=ResourceCondition.BALANCED,
                task_count=16,
                workflow_count=3,
                dependency_ratio=0.5,
                critical_path_length=6,
                parallelism_factor=2.5,
                host_count=4,
                vm_count=7,
                cpu_capacity_range=(2000, 8000),
                memory_capacity_range=(1024, 4096),
                task_length_range=(500, 15000),
                memory_requirement_range=(128, 2048),
                task_heterogeneity=0.8,
                power_idle_range=(70, 200),
                power_peak_range=(200, 700)
            )
        ])
        
        # 5. Wide-shallow vs Narrow-deep scenarios
        self.scenarios.extend([
            ScenarioConfig(
                name="wide_shallow",
                description="Wide-shallow DAG - many parallel tasks, few dependency levels",
                dag_type=DAGType.WIDE_SHALLOW,
                resource_condition=ResourceCondition.BALANCED,
                task_count=24,
                workflow_count=2,
                dependency_ratio=0.2,
                critical_path_length=3,
                parallelism_factor=8.0,
                host_count=5,
                vm_count=10,
                cpu_capacity_range=(3000, 6000),
                memory_capacity_range=(1024, 2048),
                task_length_range=(1000, 4000),
                memory_requirement_range=(256, 512),
                task_heterogeneity=0.3,
                power_idle_range=(100, 150),
                power_peak_range=(300, 500)
            ),
            ScenarioConfig(
                name="narrow_deep",
                description="Narrow-deep DAG - few parallel tasks, many dependency levels",
                dag_type=DAGType.NARROW_DEEP,
                resource_condition=ResourceCondition.BALANCED,
                task_count=16,
                workflow_count=2,
                dependency_ratio=0.8,
                critical_path_length=12,
                parallelism_factor=1.3,
                host_count=3,
                vm_count=5,
                cpu_capacity_range=(4000, 7000),
                memory_capacity_range=(2048, 4096),
                task_length_range=(1500, 5000),
                memory_requirement_range=(512, 1024),
                task_heterogeneity=0.4,
                power_idle_range=(120, 180),
                power_peak_range=(350, 550)
            )
        ])
        
        # 6. Power-focused scenarios
        self.scenarios.extend([
            ScenarioConfig(
                name="power_efficient_focus",
                description="Scenario designed to test power efficiency with long idle periods",
                dag_type=DAGType.COMPLEX,
                resource_condition=ResourceCondition.ABUNDANT,
                task_count=12,
                workflow_count=5,
                dependency_ratio=0.4,
                critical_path_length=5,
                parallelism_factor=2.0,
                host_count=6,
                vm_count=12,
                cpu_capacity_range=(6000, 10000),
                memory_capacity_range=(4096, 8192),
                task_length_range=(500, 2000),  # Short tasks = more idle time
                memory_requirement_range=(256, 512),
                task_heterogeneity=0.2,
                power_idle_range=(150, 250),  # High idle power
                power_peak_range=(400, 800)   # High peak power
            )
        ])
    
    def generate_scenario_dataset(self, scenario: ScenarioConfig) -> Dataset:
        """Generate a dataset for a specific scenario"""
        
        # Adjust generation parameters based on DAG type
        if scenario.dag_type == DAGType.LINEAR:
            dag_method = "gnp"
            gnp_min_n = scenario.task_count
            gnp_max_n = scenario.task_count
        elif scenario.dag_type == DAGType.PARALLEL:
            dag_method = "gnp"
            gnp_min_n = scenario.task_count
            gnp_max_n = scenario.task_count
        else:
            dag_method = "gnp"
            gnp_min_n = max(3, scenario.task_count - 3)
            gnp_max_n = scenario.task_count + 3
        
        # Generate base dataset
        dataset = generate_dataset(
            seed=scenario.seed,
            host_count=scenario.host_count,
            vm_count=scenario.vm_count,
            max_memory_gb=scenario.memory_capacity_range[1] // 1024,
            min_cpu_speed_mips=scenario.cpu_capacity_range[0],
            max_cpu_speed_mips=scenario.cpu_capacity_range[1],
            workflow_count=scenario.workflow_count,
            dag_method=dag_method,
            gnp_min_n=gnp_min_n,
            gnp_max_n=gnp_max_n,
            task_length_dist="normal",
            min_task_length=scenario.task_length_range[0],
            max_task_length=scenario.task_length_range[1],
            task_arrival="static",
            arrival_rate=1.0
        )
        
        # Customize dataset based on scenario requirements
        dataset = self._customize_dataset_for_scenario(dataset, scenario)
        
        return dataset
    
    def _customize_dataset_for_scenario(self, dataset: Dataset, scenario: ScenarioConfig) -> Dataset:
        """Customize dataset to match specific scenario requirements"""
        
        # Customize task dependencies based on DAG type
        for workflow in dataset.workflows:
            workflow.tasks = self._adjust_task_dependencies(
                workflow.tasks, scenario.dag_type, scenario.dependency_ratio
            )
            
            # Adjust task requirements based on scenario
            for task in workflow.tasks:
                # Adjust memory requirements
                min_mem, max_mem = scenario.memory_requirement_range
                task.req_memory_mb = np.random.randint(min_mem, max_mem + 1)
                
                # Adjust task length with heterogeneity
                base_length = task.length
                if scenario.task_heterogeneity > 0:
                    variance = base_length * scenario.task_heterogeneity
                    task.length = max(100, int(np.random.normal(base_length, variance)))
        
        # Customize VMs and hosts based on resource conditions
        dataset.hosts = self._adjust_host_resources(dataset.hosts, scenario)
        dataset.vms = self._adjust_vm_resources(dataset.vms, scenario)
        
        return dataset
    
    def _adjust_task_dependencies(self, tasks: List[Task], dag_type: DAGType, dependency_ratio: float) -> List[Task]:
        """Adjust task dependencies to match the specified DAG type"""
        
        if dag_type == DAGType.LINEAR:
            # Create linear chain
            for i, task in enumerate(tasks):
                if i == 0:
                    task.child_ids = [tasks[1].id] if len(tasks) > 1 else []
                elif i == len(tasks) - 1:
                    task.child_ids = []
                else:
                    task.child_ids = [tasks[i + 1].id]
        
        elif dag_type == DAGType.PARALLEL:
            # Make most tasks independent
            for task in tasks:
                task.child_ids = []
        
        elif dag_type == DAGType.DIAMOND:
            # Create fork-join pattern
            if len(tasks) >= 4:
                # First task forks to middle tasks
                tasks[0].child_ids = [t.id for t in tasks[1:-1]]
                # Middle tasks join to last task
                for task in tasks[1:-1]:
                    task.child_ids = [tasks[-1].id]
                # Last task has no children
                tasks[-1].child_ids = []
        
        elif dag_type == DAGType.WIDE_SHALLOW:
            # Few levels, many parallel tasks per level
            level_size = len(tasks) // 3
            for i, task in enumerate(tasks):
                if i < level_size:  # First level
                    task.child_ids = [t.id for t in tasks[level_size:2*level_size]]
                elif i < 2 * level_size:  # Second level
                    task.child_ids = [t.id for t in tasks[2*level_size:]]
                else:  # Third level
                    task.child_ids = []
        
        elif dag_type == DAGType.NARROW_DEEP:
            # Many levels, few parallel tasks per level
            for i, task in enumerate(tasks):
                if i < len(tasks) - 2:
                    # Each task depends on next 1-2 tasks
                    task.child_ids = [tasks[i + 1].id]
                    if i < len(tasks) - 3 and np.random.random() < 0.3:
                        task.child_ids.append(tasks[i + 2].id)
                else:
                    task.child_ids = []
        
        return tasks
    
    def _adjust_host_resources(self, hosts: List[Host], scenario: ScenarioConfig) -> List[Host]:
        """Adjust host resources based on scenario requirements"""
        
        for host in hosts:
            # Adjust CPU capacity
            min_cpu, max_cpu = scenario.cpu_capacity_range
            host.cpu_speed_mips = np.random.randint(min_cpu, max_cpu + 1)
            
            # Adjust power consumption
            min_idle, max_idle = scenario.power_idle_range
            min_peak, max_peak = scenario.power_peak_range
            host.power_idle_watt = np.random.randint(min_idle, max_idle + 1)
            host.power_peak_watt = np.random.randint(min_peak, max_peak + 1)
            
            # Adjust memory capacity
            min_mem, max_mem = scenario.memory_capacity_range
            host.memory_mb = np.random.randint(min_mem, max_mem + 1)
        
        return hosts
    
    def _adjust_vm_resources(self, vms: List[Vm], scenario: ScenarioConfig) -> List[Vm]:
        """Adjust VM resources based on scenario requirements"""
        
        for vm in vms:
            # Adjust memory capacity
            min_mem, max_mem = scenario.memory_capacity_range
            vm.memory_mb = min(vm.memory_mb, np.random.randint(min_mem//2, max_mem + 1))
            
            # Adjust CPU speed
            min_cpu, max_cpu = scenario.cpu_capacity_range
            vm.cpu_speed_mips = np.random.randint(min_cpu//2, max_cpu + 1)
        
        return vms
    
    def get_all_scenarios(self) -> List[ScenarioConfig]:
        """Get all available evaluation scenarios"""
        return self.scenarios.copy()
    
    def get_scenarios_by_type(self, dag_type: DAGType) -> List[ScenarioConfig]:
        """Get scenarios filtered by DAG type"""
        return [s for s in self.scenarios if s.dag_type == dag_type]
    
    def get_scenarios_by_resource_condition(self, condition: ResourceCondition) -> List[ScenarioConfig]:
        """Get scenarios filtered by resource condition"""
        return [s for s in self.scenarios if s.resource_condition == condition]
    
    def add_custom_scenario(self, scenario: ScenarioConfig):
        """Add a custom scenario to the evaluation suite"""
        self.scenarios.append(scenario)
    
    def save_scenarios_config(self, filepath: str):
        """Save scenario configurations to JSON file"""
        scenarios_data = []
        for scenario in self.scenarios:
            scenario_dict = {
                'name': scenario.name,
                'description': scenario.description,
                'dag_type': scenario.dag_type.value,
                'resource_condition': scenario.resource_condition.value,
                'task_count': scenario.task_count,
                'workflow_count': scenario.workflow_count,
                'dependency_ratio': scenario.dependency_ratio,
                'critical_path_length': scenario.critical_path_length,
                'parallelism_factor': scenario.parallelism_factor,
                'host_count': scenario.host_count,
                'vm_count': scenario.vm_count,
                'cpu_capacity_range': scenario.cpu_capacity_range,
                'memory_capacity_range': scenario.memory_capacity_range,
                'task_length_range': scenario.task_length_range,
                'memory_requirement_range': scenario.memory_requirement_range,
                'task_heterogeneity': scenario.task_heterogeneity,
                'power_idle_range': scenario.power_idle_range,
                'power_peak_range': scenario.power_peak_range,
                'seed': scenario.seed,
                'replications': scenario.replications
            }
            scenarios_data.append(scenario_dict)
        
        with open(filepath, 'w') as f:
            json.dump(scenarios_data, f, indent=2)


def create_evaluation_scenarios() -> ScenarioGenerator:
    """Create and return a comprehensive set of evaluation scenarios"""
    return ScenarioGenerator()


if __name__ == "__main__":
    # Example usage
    generator = create_evaluation_scenarios()
    
    print("Available Evaluation Scenarios:")
    print("=" * 50)
    
    for scenario in generator.get_all_scenarios():
        print(f"\n{scenario.name}:")
        print(f"  Description: {scenario.description}")
        print(f"  DAG Type: {scenario.dag_type.value}")
        print(f"  Resource Condition: {scenario.resource_condition.value}")
        print(f"  Tasks: {scenario.task_count}, Workflows: {scenario.workflow_count}")
        print(f"  Hosts: {scenario.host_count}, VMs: {scenario.vm_count}")
    
    # Generate a sample dataset
    sample_scenario = generator.get_all_scenarios()[0]
    print(f"\nGenerating sample dataset for '{sample_scenario.name}'...")
    dataset = generator.generate_scenario_dataset(sample_scenario)
    print(f"Generated: {len(dataset.workflows)} workflows, {len(dataset.vms)} VMs, {len(dataset.hosts)} hosts")

