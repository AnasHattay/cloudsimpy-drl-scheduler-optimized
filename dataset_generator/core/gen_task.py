import math
import os

import numpy as np
from scipy import stats

# Define WORKFLOW_FILES relative to this file
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")
WORKFLOW_FILES = [
    os.path.join(DATA_PATH, "pegasus_workflows", "sipht.dag"),
]

# Generating Delay
# ----------------------------------------------------------------------------------------------------------------------


def generate_poisson_delay(lam: float, rng: np.random.RandomState) -> float:
    """
    Generate a random delay between workflows in a Poisson process.
    The delay is exponentially distributed with parameter lambda.
    """

    return stats.expon.rvs(scale=1 / lam, random_state=rng)


def generate_delay(method: str, rng: np.random.RandomState, **kwargs) -> float:
    """
    Generate a random delay between workflows based on the specified method.
    Available methods: dynamic, static
    """

    if method == "dynamic":
        return generate_poisson_delay(kwargs["arrival_rate"], rng)
    elif method == "static":
        return 0
    else:
        raise ValueError(f"Invalid method: {method}")


# Generating Task Length
# ----------------------------------------------------------------------------------------------------------------------


def generate_task_length(dist: str, low: float, high: float, rng: np.random.RandomState) -> float:
    """
    Generate a random task length based on the specified method. <br/>
    Available methods: uniform, normal, left_skewed, right_skewed <br/>
    """

    if dist == "uniform":
        return stats.uniform.rvs(loc=low, scale=high - low)

    # 99.7% of the data is within 3 standard deviations (by empirical rule)
    # so we set the standard deviation to be 1/6 of the range
    mean = (low + high) / 2
    std = (high - low) / 6
    if dist == "normal":
        method = lambda: stats.norm.rvs(loc=mean, scale=std, random_state=rng)
    elif dist == "left_skewed":
        method = lambda: stats.skewnorm.rvs(-5, loc=mean, scale=std, random_state=rng)
    elif dist == "right_skewed":
        method = lambda: stats.skewnorm.rvs(5, loc=mean, scale=std, random_state=rng)
    else:
        raise ValueError(f"Invalid distribution: {dist}")

    value = low - 1
    while value < low or value > high:
        value = method()
    return value


# Generating Task DAG
# ----------------------------------------------------------------------------------------------------------------------


def generate_dag_gnp(n: int, p: float | None, rng: np.random.RandomState) -> dict[int, set[int]]:
    """
    Generate a random Directed Acyclic Graph (DAG) using the G(n, p) model.
    The resulting graph is represented as an adjacency list. <br/>
    The resulting graph has n nodes, with node 0 being the starting node. <br/>
    If p is None, it is set to log(n + eps) / n where n is the number of generated nodes.
    """

    if n == 1:
        return {0: set()}

    if p is None:
        p = math.log(n + 0.1) / n

    nodes: dict[int, set[int]] = {i: set() for i in range(n)}
    start_nodes: set[int] = set(range(1, n))

    for i in range(1, n):
        for j in range(i + 1, n):
            if rng.random() < p:
                nodes[i].add(j)
                start_nodes.discard(j)

    for i in start_nodes:
        nodes[0].add(i)

    return nodes


def generate_dag_pegasus(dag_file: str) -> tuple[dict[int, set[int]], dict[str, int]]:
    """
    Generate a Directed Acyclic Graph (DAG) using the Pegasus workflow generator.
    The resulting graph is represented as an adjacency list. <br/>
    This will return 2 values: the nodes and the node numbers.
    Node numbers are the assigned numbers of the nodes in the DAG file.
    """

    with open(dag_file) as f:
        lines = f.readlines()

    nodes: dict[int, set[int]] = {}
    node_numbers: dict[str, int] = {}
    for line in lines:
        if line.startswith("JOB"):
            node_name = line.split()[1]
            node_number = len(node_numbers)
            node_numbers[node_name] = node_number
            nodes[node_number] = set()

    for line in lines:
        if line.startswith("PARENT"):
            parent_name = line.split()[1]
            child_name = line.split()[3]
            parent = node_numbers[parent_name]
            child = node_numbers[child_name]
            nodes[parent].add(child)

    return nodes, node_numbers


def generate_dag_pegasus_random(rng: np.random.RandomState) -> tuple[dict[int, set[int]], dict[str, int]]:
    """
    Generate a random Directed Acyclic Graph (DAG) using the Pegasus workflow generator.
    The resulting graph is represented as an adjacency list. <br/>
    This will return 2 values: the nodes and the node numbers.
    Node numbers are the assigned numbers of the nodes in the DAG file.
    """
    dag_file = WORKFLOW_FILES[rng.randint(0, len(WORKFLOW_FILES))]
    return generate_dag_pegasus(str(dag_file), rng)


def generate_dag(method: str, rng: np.random.RandomState, **kwargs) -> dict[int, set[int]]:
    """
    Generate a Directed Acyclic Graph (DAG) using the specified method.
    """

    if method == "gnp":
        gnp_min_n = kwargs["gnp_min_n"]
        gnp_max_n = kwargs["gnp_max_n"]
        gnp_p = kwargs.get("gnp_p", None)
        task_count = rng.randint(gnp_min_n, gnp_max_n + 1)
        return generate_dag_gnp(task_count, gnp_p, rng)
    elif method == "pegasus":
        graph, _ = generate_dag_pegasus_random(rng)
        return graph
    else:
        raise ValueError(f"Invalid method: {method}")
