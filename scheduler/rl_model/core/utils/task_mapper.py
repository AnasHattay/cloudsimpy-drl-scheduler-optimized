from scheduler.rl_model.core.types import TaskDto


class TaskMapper:
    """
    Maps task IDs to a single sequence of task IDs.

    0 is reserved for the dummy start task.
    1 to N are the tasks.
    N+1 is reserved for the dummy end task.
    All tasks are combined into a single workflow with id 0.
    """

    def __init__(self, tasks: list[TaskDto]):
        self._tasks = tasks

        # Create a cumulative count of tasks per workflow
        max_workflow_id = max(_task.workflow_id for _task in tasks)
        self._task_counts_cum = [0] * (max_workflow_id + 2)
        for task in tasks:
            self._task_counts_cum[task.workflow_id + 1] += 1

        # Convert counts to cumulative sums
        for i in range(1, len(self._task_counts_cum)):
            self._task_counts_cum[i] += self._task_counts_cum[i - 1]

    def map_id(self, workflow_id: int, task_id: int) -> int:
        """Map a task ID to a single sequence of task IDs."""
        mapped_task_id = self._task_counts_cum[workflow_id] + task_id
        mapped_task_id += 1  # Dummy start task has ID 0 (offset 1)
        return mapped_task_id

    def unmap_id(self, mapped_task_id: int) -> tuple[int, int]:
        """Unmap a task ID from a single sequence of task IDs."""
        mapped_task_id -= 1  # Remove offset by dummy start task
        for workflow_id in range(1, len(self._task_counts_cum)):
            if self._task_counts_cum[workflow_id - 1] <= mapped_task_id < self._task_counts_cum[workflow_id]:
                return workflow_id - 1, mapped_task_id - self._task_counts_cum[workflow_id - 1]

        raise ValueError("Out of range")

    # noinspection PyMethodMayBeStatic
    def dummy_start_task_id(self) -> int:
        """Get the ID of the dummy start task."""
        return 0

    def dummy_end_task_id(self) -> int:
        """Get the ID of the dummy end task."""
        return self._task_counts_cum[-1] + 1

    def map_tasks(self) -> list[TaskDto]:
        """Map all tasks to a single sequence of task IDs."""
        dummy_start_task = TaskDto(
            id=self.dummy_start_task_id(),
            workflow_id=0,
            length=0,
            req_memory_mb=0,
            child_ids=[self.map_id(_task.workflow_id, 0) for _task in self._tasks if _task.id == 0],
        )
        dummy_end_task = TaskDto(
            id=self.dummy_end_task_id(),
            workflow_id=0,
            length=0,
            req_memory_mb=0,
            child_ids=[],
        )

        mapped_tasks: list[TaskDto] = [dummy_start_task]
        for task in self._tasks:
            mapped_child_ids = [self.map_id(task.workflow_id, child_id) for child_id in task.child_ids]
            if not mapped_child_ids:
                # If there are no children, link to the dummy end task
                mapped_child_ids = [dummy_end_task.id]

            mapped_tasks.append(
                TaskDto(
                    id=self.map_id(task.workflow_id, task.id),
                    workflow_id=0,
                    length=task.length,
                    req_memory_mb=task.req_memory_mb,
                    child_ids=mapped_child_ids,
                )
            )
        mapped_tasks.append(dummy_end_task)

        return mapped_tasks
