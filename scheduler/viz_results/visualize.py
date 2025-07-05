import dataclasses
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tyro
from typing import List

from scheduler.config.settings import ALGORITHMS


@dataclasses.dataclass
class Args:
    import_csv: str
    """File to import the CSV"""


def main(args: Args):
    # Load data
    df = pd.read_csv(args.import_csv)
    avg_df = df.groupby(["SettingId", "Algorithm"], as_index=False).agg({"Makespan": "mean", "EnergyJ": "mean"})

    # Reorder the data based on the provided order
    algorithm_order = [x[0] for x in ALGORITHMS]
    avg_df["Algorithm"] = pd.Categorical(avg_df["Algorithm"], categories=algorithm_order, ordered=True)
    avg_df = avg_df.sort_values("Algorithm")

    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(6, 5))

    # Bar plot for Makespan on primary Y-axis
    color1 = "#dffdb9"
    bars = ax1.bar(avg_df["Algorithm"], avg_df["Makespan"], color=color1, label="Makespan", edgecolor="black")
    ax1.set_ylabel("Makespan")
    ax1.tick_params(axis="y")
    ax1.set_xlabel("Algorithm")

    # Line plot for EnergyJ on secondary Y-axis
    ax2 = ax1.twinx()
    color2 = "#ff5757"
    (line,) = ax2.plot(
        avg_df["Algorithm"], avg_df["EnergyJ"], color=color2, marker="o", linestyle="-", label="Energy Consumption"
    )
    ax2.set_ylabel("Energy Consumption (J)")
    ax2.tick_params(axis="y")

    # Angle the X-axis labels
    ax1.set_xticklabels(avg_df["Algorithm"], rotation=45, ha="right")

    # Combine legends
    bars_legend = ax1.legend(loc="upper left")
    line_legend = ax2.legend(loc="upper right")
    ax1.add_artist(bars_legend)
    ax2.add_artist(line_legend)

    # Title and layout adjustments
    fig.suptitle("Comparison of Makespan and Energy Consumption Across Algorithms")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
