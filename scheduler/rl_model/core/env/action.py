from dataclasses import dataclass


@dataclass
class EnvAction:
    task_id: int
    vm_id: int
