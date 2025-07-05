import numpy as np

from dataset_generator.core.gen_task import generate_delay, generate_task_length, generate_dag
from dataset_generator.core.models import Task, Workflow


def generate_workflows(
    workflow_count: int,
    dag_method: str,
    gnp_min_n: int,
    gnp_max_n: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    max_req_memory_mb: int,
    task_arrival: str,
    arrival_rate: float,
    rng: np.random.RandomState,
) -> list[Workflow]:
    """
    Generate a list of workflows.
    """

    def delay_gen() -> int:
        return int(generate_delay(task_arrival, arrival_rate=arrival_rate, rng=rng))

    def task_length_gen() -> int:
        return int(generate_task_length(task_length_dist, min_task_length, max_task_length, rng))

    def req_memory_gen() -> int:
        return (1 + rng.randint(0, max_req_memory_mb // 1024)) * 1024

    def dag_gen() -> dict[int, set[int]]:
        return generate_dag(dag_method, gnp_min_n=gnp_min_n, gnp_max_n=gnp_max_n, rng=rng)

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(workflow_count):
        dag = dag_gen()
        tasks: list[Task] = [
            Task(
                id=task_id,
                workflow_id=workflow_id,
                length=task_length_gen(),
                req_memory_mb=req_memory_gen(),
                child_ids=list(child_ids),
            )
            for task_id, child_ids in dag.items()
        ]
        arrival_time += delay_gen()
        workflows.append(Workflow(id=workflow_id, tasks=tasks, arrival_time=arrival_time))

    return workflows
