from scheduler.rl_model.core.env.observation import VmObservation
from scheduler.rl_model.core.types import VmDto, TaskDto


def is_suitable(vm: VmDto, task: TaskDto):
    """Check if the VM is suitable for the task."""
    return vm.memory_mb >= task.req_memory_mb


def active_energy_consumption_per_mi(vm: VmDto | VmObservation):
    """How much additional energy is consumed by a task running in this VM (per length)"""
    # return (vm.host_power_peak_watt - vm.host_power_idle_watt) / vm.host_cpu_speed_mips
    return vm.host_power_peak_watt / vm.host_cpu_speed_mips
