from scheduler.config.settings import DEFAULT_MODEL_DIR
from scheduler.viz_results.algorithms.base import BaseScheduler
from scheduler.viz_results.algorithms.best_fit import BestFitScheduler
from scheduler.viz_results.algorithms.cp_sat import CpSatScheduler
from scheduler.viz_results.algorithms.ferpts import FerptsScheduler
from scheduler.viz_results.algorithms.ga import GAScheduler
from scheduler.viz_results.algorithms.gin_agent import GinAgentScheduler
from scheduler.viz_results.algorithms.heft_ins import InsertionHeftScheduler
from scheduler.viz_results.algorithms.heft import HeftScheduler
from scheduler.viz_results.algorithms.max_min import MaxMinScheduler
from scheduler.viz_results.algorithms.min_min import MinMinScheduler
from scheduler.viz_results.algorithms.power_saving import PowerSavingScheduler
from scheduler.viz_results.algorithms.random import RandomScheduler
from scheduler.viz_results.algorithms.round_robin import RoundRobinScheduler


def get_scheduler(algorithm: str) -> BaseScheduler:
    strategy, *args = algorithm.split(":")
    if strategy == "random":
        return RandomScheduler()
    if strategy == "round_robin":
        return RoundRobinScheduler()
    if strategy == "ferpts":
        return FerptsScheduler()
    elif strategy == "best_fit":
        return BestFitScheduler()
    elif strategy == "min_min":
        return MinMinScheduler()
    elif strategy == "max_min":
        return MaxMinScheduler()
    elif strategy == "cp_sat":
        return CpSatScheduler()
    elif strategy == "insertion_heft":
        return InsertionHeftScheduler()
    elif strategy == "heft":
        return HeftScheduler()
    elif strategy == "power_saving":
        return PowerSavingScheduler()
    elif strategy == "ga":
        return GAScheduler()
    elif strategy == "gin":
        return GinAgentScheduler(model_path=str(DEFAULT_MODEL_DIR / args[0] / args[1]))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
