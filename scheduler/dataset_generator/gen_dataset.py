from dataclasses import dataclass

import tyro
import json

from scheduler.dataset_generator.core.gen_dataset import generate_dataset


@dataclass
class DatasetArgs:
    seed: int = 42
    """random seed"""
    host_count: int = 2
    """number of hosts"""
    vm_count: int = 1
    """number of VMs"""
    max_memory_gb: int = 10
    """maximum amount of RAM for a VM (in GB)"""
    min_cpu_speed: int = 500
    """minimum CPU speed in MIPS"""
    max_cpu_speed: int = 5000
    """maximum CPU speed in MIPS"""
    workflow_count: int = 3
    """number of workflows"""
    dag_method: str = "gnp"
    """DAG generation method (pegasus, gnp)"""
    gnp_min_n: int = 1
    """minimum number of tasks per workflow (for G(n,p) method)"""
    gnp_max_n: int = 10
    """maximum number of tasks per workflow (for G(n,p) method)"""
    task_length_dist: str = "normal"
    """task length distribution (normal, uniform, left_skewed, right_skewed)"""
    min_task_length: int = 500
    """minimum task length"""
    max_task_length: int = 100_000
    """maximum task length"""
    task_arrival: str = "dynamic"
    """task arrival mode (static, dynamic)"""
    arrival_rate: float = 3
    """arrival rate of workflows/second (for dynamic arrival)"""


def main(args: DatasetArgs):
    dataset = generate_dataset(
        seed=args.seed,
        host_count=args.host_count,
        vm_count=args.vm_count,
        max_memory_gb=args.max_memory_gb,
        min_cpu_speed_mips=args.min_cpu_speed,
        max_cpu_speed_mips=args.max_cpu_speed,
        workflow_count=args.workflow_count,
        dag_method=args.dag_method,
        gnp_min_n=args.gnp_min_n,
        gnp_max_n=args.gnp_max_n,
        task_length_dist=args.task_length_dist,
        min_task_length=args.min_task_length,
        max_task_length=args.max_task_length,
        task_arrival=args.task_arrival,
        arrival_rate=args.arrival_rate,
    )

    json_data = json.dumps(dataset.to_json())
    print(json_data)


if __name__ == "__main__":
    main(tyro.cli(DatasetArgs))
