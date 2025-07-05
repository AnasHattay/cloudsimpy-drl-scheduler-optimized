import json
from dataclasses import dataclass

import tyro

from scheduler.dataset_generator.core.models import Dataset, Solution
from scheduler.dataset_generator.solvers.solver import solve
from scheduler.dataset_generator.visualizers.printers import print_solution


@dataclass
class Args:
    method: str = "sat"
    """method to solve the dataset (sat)"""


def main(args: Args):
    dataset_str = input()
    dataset_dict = json.loads(dataset_str)
    dataset = Dataset.from_json(dataset_dict)

    # Solution
    vm_assignments = solve(args.method, dataset)
    print_solution(dataset.workflows, vm_assignments)

    solution = Solution(dataset=dataset, vm_assignments=vm_assignments)
    json_data = json.dumps(solution.to_json())
    print(json_data)


if __name__ == "__main__":
    main(tyro.cli(Args))
