from abc import ABC

from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto


class BaseScheduler(ABC):
    """Base class for scheduling algorithms."""

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        """Schedule the tasks on the VMs."""
        raise NotImplementedError

    def is_optimal(self) -> bool:
        """Check if the scheduling algorithm is optimal."""
        return False
