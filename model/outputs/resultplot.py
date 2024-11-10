import argparse
import glob
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (7.2, 4.45),
        "text.usetex": True,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 15,
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 15,
    },
)
colors = sns.color_palette("colorblind", 4)
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_column(files, column_name, aggregation, sep=",", ma=1):
    ep_numbers = []
    values = []

    for file in files:
        df = pd.read_csv(file, sep=sep)
        if aggregation == "avg":
            value = df[column_name].mean()
        elif aggregation == "max":
            value = df[column_name].max()
        elif aggregation == "sum":
            value = df[column_name].sum()
        else:
            raise ValueError("Invalid aggregation type. Use 'avg', 'max', or 'sum'.")

        ep_number = int(file.split("_ep")[1].split(".csv")[0])

        ep_numbers.append(ep_number)
        values.append(value)

    # Sort by episode number
    sorted_indices = np.argsort(ep_numbers)
    ep_numbers = np.array(ep_numbers)[sorted_indices]
    values = np.array(values)[sorted_indices]

    if ma > 1:
        values = moving_average(values, ma)

    plt.plot(
        ep_numbers,
        values,
        label=f"{aggregation.capitalize()} {column_name.replace('_', ' ').title()}",
        color=next(colors),
        linestyle=next(dashes_styles),
    )
    plt.xlabel("Episode")
    plt.ylabel(f"{aggregation.capitalize()} {column_name.replace('_', ' ').title()}")
    plt.title(
        f"{aggregation.capitalize()} {column_name.replace('_', ' ').title()} per Episode"
    )
    plt.ylim(bottom=0)
    plt.legend()


def plot_step_vs_column(file, column_name, sep=","):
    df = pd.read_csv(file, sep=sep)
    steps = np.arange(len(df))
    values = df[column_name].values

    plt.plot(
        steps,
        values,
        label=f"{column_name.replace('_', ' ').title()}",
        color=next(colors),
        linestyle=next(dashes_styles),
    )
    plt.xlabel("Step")
    plt.ylabel(f"{column_name.replace('_', ' ').title()}")
    plt.title(f"{column_name.replace('_', ' ').title()} per Step")
    plt.ylim(bottom=0)
    plt.legend()


def plot_all_columns(files, sep=",", ma=1):
    columns = [
        "system_mean_waiting_time",
        "system_total_waiting_time",
        "t_average_density",
    ]
    aggregations = ["avg", "max", "sum"]

    for column in columns:
        for agg in aggregations:
            plt.figure()
            plot_column(files, column, agg, sep=sep, ma=ma)
            output_file = os.path.join(args.output, f"s{args.start}_{agg}_{column}.pdf")
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()

    # Plot sum of total waiting time
    plt.figure()
    plot_column(files, "system_total_waiting_time", "sum", sep=sep, ma=ma)
    output_file = os.path.join(
        args.output, f"s{args.start}_sum_system_total_waiting_time.pdf"
    )
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Plot Column Value per Episode""",
    )
    prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
    prs.add_argument(
        "-col",
        type=str,
        required=True,
        help="Column name to plot or 'ALL' for all columns.\n",
    )
    prs.add_argument(
        "-agg",
        type=str,
        choices=["avg", "max", "sum"],
        help="Aggregation type: 'avg', 'max', or 'sum'.\n",
    )
    prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument("-output", type=str, default=None, help="Output folder path.\n")
    prs.add_argument("-start", type=int, default=0, help="Start file number.\n")
    prs.add_argument("-end", type=int, help="End file number.\n")

    args = prs.parse_args()

    if args.output is not None:
        # Create the output folder if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

    # Filter files based on start and end numbers
    if args.end is None:
        files = [
            f for f in args.f if int(f.split("_ep")[1].split(".csv")[0]) >= args.start
        ]
    else:
        files = [
            f
            for f in args.f
            if args.start <= int(f.split("_ep")[1].split(".csv")[0]) <= args.end
        ]

    if args.start == args.end:
        if args.col == "ALL":
            columns = [
                "system_mean_waiting_time",
                "system_total_waiting_time",
                "t_average_density",
            ]
            for column in columns:
                plt.figure()
                plot_step_vs_column(files[0], column, sep=args.sep)
                output_file = os.path.join(
                    args.output, f"ep{args.start}_step_vs_{column}.pdf"
                )
                plt.savefig(output_file, bbox_inches="tight")
                plt.close()
        else:
            plt.figure()
            plot_step_vs_column(files[0], args.col, sep=args.sep)
            output_file = os.path.join(
                args.output, f"ep{args.start}_step_vs_{args.col}.pdf"
            )
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
    else:
        if args.col == "ALL":
            plot_all_columns(files, sep=args.sep, ma=args.ma)
        else:
            if args.agg is None:
                raise ValueError(
                    "Aggregation type must be specified when not using 'ALL' for columns."
                )
            plt.figure()
            # File reading and plotting
            plot_column(files, args.col, args.agg, sep=args.sep, ma=args.ma)

            if args.output is not None:
                # Generate the output file name
                output_file = os.path.join(
                    args.output, f"s{args.start}_{args.agg}_{args.col}.pdf"
                )
                plt.savefig(output_file, bbox_inches="tight")

            plt.show()

# Example usage:
# start, end optional. If not specified, all files are used.
# Plot the average system mean waiting time for episodes 0 to 10
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col system_mean_waiting_time -agg avg -output outputs/plots/ -start 0 -end 10

# Plot the maximum system total waiting time for episodes 0 to 10 and save the output to the specified folder.
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col system_total_waiting_time -agg max -output outputs/plots/ -start 0 -end 10

# Plot the average t average density for episodes 0 to 10 and save the output to the specified folder.
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col t_average_density -agg avg -output outputs/plots/ -start 0 -end 10

# Plot all specified columns (system_mean_waiting_time, system_total_waiting_time, t_average_density) with all aggregations (avg, max, sum) for episodes 0 to 10 and save the outputs to the specified folder.
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/ -start 0 -end 10


# Plot step vs. system mean waiting time for episode 0
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col system_mean_waiting_time -output outputs/plots/ -start 0 -end 0

# Plot step vs. all specified columns (system_mean_waiting_time, system_total_waiting_time, t_average_density) for episode 0
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/ -start 0 -end 0


# main commands:
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/ -start 500
# python outputs/resultplot.py -f outputs/2way-single-intersection/PPO_conn0_ep*.csv -col ALL -output outputs/plots/ -start 2900 -end 2900
