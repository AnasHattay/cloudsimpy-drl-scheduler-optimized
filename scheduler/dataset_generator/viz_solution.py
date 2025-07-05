import json
from dataclasses import dataclass

import tyro

import networkx as nx
import matplotlib.pyplot as plt

from scheduler.dataset_generator.core.models import Solution
from scheduler.dataset_generator.visualizers.plotters import (
    plot_execution_graph,
    plot_gantt_chart,
    plot_workflow_graphs,
)
from scheduler.dataset_generator.visualizers.printers import print_solution
from scheduler.dataset_generator.visualizers.utils import save_agraph


@dataclass
class Args:
    prefix: str = "logs/data/viz_solution"
    """file prefix to use (with directory)"""


def main(args: Args):
    dataset_str = input()
    dataset_dict = json.loads(dataset_str)
    solution = Solution.from_json(dataset_dict)

    # Workflow graph
    _, ax = plt.subplots()
    g_w: nx.DiGraph = nx.DiGraph()
    a_w = plot_workflow_graphs(g_w, solution.dataset.workflows)
    save_agraph(a_w, f"{args.prefix}_workflows.png")

    # Solution
    print_solution(solution.dataset.workflows, solution.vm_assignments)

    # Execution graph
    _, ax = plt.subplots()
    g_e: nx.DiGraph = nx.DiGraph()
    a_e = plot_execution_graph(g_e, solution.dataset.workflows, solution.dataset.vms, solution.vm_assignments)
    save_agraph(a_e, f"{args.prefix}_execution.png", dir_lr=True)

    # Gantt chart
    _, ax = plt.subplots()
    plot_gantt_chart(ax, solution.dataset.workflows, solution.dataset.vms, solution.vm_assignments, label=False)
    plt.savefig(f"{args.prefix}_gantt.png")


if __name__ == "__main__":
    main(tyro.cli(Args))
