import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygraphviz as pgv

from scheduler.dataset_generator.core.models import Workflow, VmAssignment, Vm


# Helper functions
# ----------------------------------------------------------------------------------------------------------------------


def get_color(color_id: int) -> str:
    color_map = ["#FADFA1", "#7EACB5", "#E6B9A6", "#939185", "#FFF078", "#939185"]
    return color_map[color_id % len(color_map)]


def get_node_id(workflow_id: int, task_id: int) -> str:
    return f"{workflow_id}-{task_id}"


# Graphing functions for Workflow and Execution graphs
# ----------------------------------------------------------------------------------------------------------------------


def plot_workflow_graphs(g: nx.DiGraph, workflows: list[Workflow]) -> pgv.AGraph:
    """
    Plot the workflows on the provided DiGraph.
    """

    for workflow in workflows:
        for task in workflow.tasks:
            node_id = get_node_id(workflow.id, task.id)
            node_label = f"{task.length} MI\n{task.req_memory_mb // 1024} GB"
            node_color = get_color(workflow.id)
            g.add_node(node_id, label=node_label, fillcolor=node_color, style="filled", fontname="Arial")
            for child_id in task.child_ids:
                g.add_edge(node_id, get_node_id(workflow.id, child_id), color="black")

    return nx.nx_agraph.to_agraph(g)


def plot_execution_graph(g: nx.DiGraph, workflows: list[Workflow], vms: list[Vm], result: list[VmAssignment]):
    """
    Plot the execution graph of the provided workflows on the provided DiGraph.
    """

    # Assignments
    vm_nodes: dict[int, list[str]] = {vm.id: [] for vm in vms}
    start_times: dict[str, float] = {}
    for assignment in result:
        node_id = get_node_id(assignment.workflow_id, assignment.task_id)
        vm_nodes[assignment.vm_id].append(node_id)
        start_times[node_id] = assignment.start_time

    # Add edges between tasks executed on the same VM
    for vm_id in vm_nodes:
        vm_nodes[vm_id].sort(key=lambda nid: start_times[nid])
        for i in range(1, len(vm_nodes[vm_id])):
            g.add_edge(vm_nodes[vm_id][i - 1], vm_nodes[vm_id][i], color="lightgray")

    # Add nodes and edges for tasks
    task_map = {(task.workflow_id, task.id): task for workflow in workflows for task in workflow.tasks}
    processing: set[tuple[int, int]] = {(assignment.workflow_id, assignment.task_id) for assignment in result}
    no_vm_nodes: list[str] = []
    while processing:
        workflow_id, task_id = processing.pop()
        task = task_map[(workflow_id, task_id)]
        node_id = get_node_id(workflow_id, task_id)
        node_label = f"{task.length} MI\n{task.req_memory_mb//1024} GB"
        node_color = get_color(workflow_id)
        g.add_node(node_id, label=node_label, fillcolor=node_color, style="filled", fontname="Arial", shape="box")
        for child_id in task.child_ids:
            child_node_id = get_node_id(workflow_id, child_id)
            g.add_edge(node_id, child_node_id, color=node_color)
            if (workflow_id, child_id) not in processing:
                no_vm_nodes.append(child_node_id)
                processing.add((workflow_id, child_id))

    a = nx.nx_agraph.to_agraph(g)

    # Add subgraphs for each VM
    for vm in vms:
        if vm_nodes[vm.id]:
            label = f"VM {vm.id}\n{int(vm.cpu_speed_mips)} MIPS\n{vm.memory_mb // 1024} GB"
            a.add_subgraph(vm_nodes[vm.id], name=f"cluster_{vm.id}", style="dashed", fontname="Arial", label=label)
    for node_id in no_vm_nodes:
        a.add_subgraph(node_id, name="cluster_no_vm", style="dashed", fontname="Arial", label="Unscheduled")

    return a


# Graphing functions for Gantt chart
# ----------------------------------------------------------------------------------------------------------------------


def plot_gantt_chart(ax: plt.Axes, workflows: list[Workflow], vms: list[Vm], result: list[VmAssignment], label=True):
    result_map: dict[tuple[int, int], VmAssignment] = {
        (assignment.workflow_id, assignment.task_id): assignment for assignment in result
    }

    for workflow in workflows:
        for task in workflow.tasks:
            if (workflow.id, task.id) not in result_map:
                continue

            assigned_task = result_map[(workflow.id, task.id)]
            if assigned_task.end_time < 0:
                continue

            ax.broken_barh(
                [(assigned_task.start_time, assigned_task.end_time - assigned_task.start_time)],
                (int(assigned_task.vm_id) - 0.3, 0.6),
                color=get_color(workflow.id),
                edgecolor="black",
                linewidth=0.5,
            )
            if label:
                ax.text(
                    x=assigned_task.start_time + (assigned_task.end_time - assigned_task.start_time) / 2,
                    y=int(assigned_task.vm_id),
                    s=f"W{workflow.id} T{task.id} ({assigned_task.end_time - assigned_task.start_time:.0f}s)\n{task.length}MI {task.req_memory_mb // 1024}GB",
                    ha="center",
                    va="center",
                )

    ax.set_yticks(range(len(vms)))
    ax.set_yticklabels([f"VM {vm.id}\n{int(vm.cpu_speed_mips)}MIPS {vm.memory_mb // 1024}GB" for vm in vms])
    ax.set_xlabel("Time")


# Graphing functions for Pegasus DAG
# ----------------------------------------------------------------------------------------------------------------------


def plot_pegasus_dag(graph: dict[int, set[int]], node_numbers: dict[str, int]) -> pgv.AGraph:
    node_names = {v: k for k, v in node_numbers.items()}

    g: nx.DiGraph = nx.DiGraph()
    for node in graph.keys():
        g.add_node(node)
        for child in graph[node]:
            g.add_edge(node, child)

    groups: dict[str, set[int]] = {}
    for node in graph.keys():
        node_name = node_names[node]
        group = node_name.split("_ID")[0]
        if group not in groups:
            groups[group] = set()
        groups[group].add(node)

    a: pgv.AGraph = nx.nx_agraph.to_agraph(g)
    for group in groups:
        a.add_subgraph(groups[group], name=f"cluster_{group}", label=group, style="dashed")
        color = f"/set312/{random.randint(1, 12)}"
        for node in groups[group]:
            n = a.get_node(node)
            n.attr["fillcolor"] = color
            n.attr["style"] = "filled"
            n.attr["label"] = ""

    return a


# ----------------------------------------------------------------------------------------------------------------------


def plot_2d_matrix(ax: plt.Axes, title: str, matrix: np.ndarray):
    ax.matshow(matrix)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks(np.arange(matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if int(matrix[i, j]) == matrix[i, j]:
                label = f"{int(matrix[i, j])}"
            else:
                label = f"{matrix[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="w", fontsize=8)

    ax.set_title(title)
