import json
import os

import numpy as np

from dataset_generator.core.models import Host, Vm

# Define HOST_SPECS_PATH relative to this file
HOST_SPECS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "host_specs.json")


# Generating Hosts
# ----------------------------------------------------------------------------------------------------------------------


def generate_hosts(n: int, rng: np.random.RandomState) -> list[Host]:
    """
    Generate a list of hosts with the specified number of hosts.
    Uses the host specifications from data/host_specs.json.
    """

    with open(HOST_SPECS_PATH) as f:
        available_hosts: list = json.load(f)

    hosts: list[Host] = []
    for i in range(n):
        spec = available_hosts[rng.randint(0, len(available_hosts))]
        hosts.append(
            Host(
                id=i,
                cores=int(spec["cores"]),
                cpu_speed_mips=int(spec["cpu_speed_gips"] * 1e3),
                memory_mb=int(spec["memory_gb"] * 1024),
                disk_mb=int(spec["disk_tb"] * 1e6),
                bandwidth_mbps=int(spec["bandwidth_gbps"] * 1024),
                power_idle_watt=int(spec["power_idle_watt"]),
                power_peak_watt=int(spec["power_peak_watt"]),
            )
        )
    return hosts


# Generating and Allocating VMs
# ----------------------------------------------------------------------------------------------------------------------


def generate_vms(
    n: int, max_memory_gb: int, min_cpu_speed_mips: int, max_cpu_speed_mips: int, rng: np.random.RandomState
) -> list[Vm]:
    """
    Generate a list of VMs with the specified number of VMs.
    """

    vms = []
    for i in range(n):
        ram_mb = rng.randint(1, max_memory_gb + 1) * 1024
        cpu_speed = rng.randint(min_cpu_speed_mips, max_cpu_speed_mips + 1)
        host_id = -1  # Unallocated
        vms.append(Vm(i, host_id, cpu_speed, memory_mb=ram_mb, disk_mb=1024, bandwidth_mbps=50, vmm="Xen"))
    return vms


def allocate_vms(vms: list[Vm], hosts: list[Host], rng: np.random.RandomState):
    """
    Allocate VMs to hosts randomly.
    """

    for vm in vms:
        host = hosts[rng.randint(0, len(hosts))]
        vm.host_id = host.id
