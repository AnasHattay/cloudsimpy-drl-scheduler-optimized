from scheduler.dataset_generator.core.models import Dataset, VmAssignment
from scheduler.dataset_generator.solvers.cp_sat_solver import solve_cp_sat


def solve(method: str, dataset: Dataset) -> list[VmAssignment]:
    if method == "sat":
        is_optimal, assignments = solve_cp_sat(workflows=dataset.workflows, vms=dataset.vms)
        if not is_optimal:
            print("Warning: CP-SAT solver did not find an optimal solution.")
        return assignments

    raise ValueError(f"Unknown method: {method}")
