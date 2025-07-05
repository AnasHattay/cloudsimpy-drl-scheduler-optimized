import sys

from scheduler.dataset_generator.core.models import Workflow, VmAssignment


def print_solution(workflows: list[Workflow], result: list[VmAssignment]):
    all_tasks = {(workflow.id, task.id) for workflow in workflows for task in workflow.tasks}
    scheduled_tasks = {(assignment.workflow_id, assignment.task_id) for assignment in result}
    finished_tasks = {(assignment.workflow_id, assignment.task_id) for assignment in result if assignment.end_time > 0}

    makespan_start = {workflow.id: float("inf") for workflow in workflows}
    makespan_end = {workflow.id: float("-inf") for workflow in workflows}

    for assignment in result:
        workflow_id = assignment.workflow_id
        makespan_start[workflow_id] = min(makespan_start[workflow_id], assignment.start_time)
        makespan_end[workflow_id] = max(makespan_end[workflow_id], assignment.end_time)
    total_makespan = sum(
        makespan_end[workflow_id] - makespan_start[workflow_id]
        for workflow_id in makespan_start
        if makespan_start[workflow_id] != float("inf") and makespan_end[workflow_id] != float("-inf")
    )

    print(file=sys.stderr)
    print(f"Total tasks     : {len(all_tasks)}", file=sys.stderr)
    print(f"Scheduled tasks : {len(scheduled_tasks)}", file=sys.stderr)
    print(f"Finished tasks  : {len(finished_tasks)}", file=sys.stderr)
    print(f"Total makespan  : {total_makespan}", file=sys.stderr)
