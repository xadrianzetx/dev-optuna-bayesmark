import argparse
import json
import os
import subprocess
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
from matplotlib import cm, colors
from matplotlib.axes import Axes

_RUN_NAME = "bo_optuna_run"


def run(args: argparse.Namespace) -> None:

    sampler_list = args.sampler_list.split()
    sampler_kwargs_list = args.sampler_kwargs_list.split()
    pruner_list = args.pruner_list.split()
    pruner_kwargs_list = args.pruner_kwargs_list.split()

    config = {}
    for sampler, sampler_kwargs in zip(sampler_list, sampler_kwargs_list):
        for pruner, pruner_kwargs in zip(pruner_list, pruner_kwargs_list):
            optimizer_name = f"{sampler}-{pruner}"
            optimizer_kwargs = {
                "sampler": sampler,
                "sampler_kwargs": json.loads(sampler_kwargs),
                "pruner": pruner,
                "pruner_kwargs": json.loads(pruner_kwargs),
            }
            config[optimizer_name] = ["optuna_optimizer.py", optimizer_kwargs]

    with open("config.json", "w") as file:
        json.dump(config, file, indent=4)

    samplers = " ".join(config.keys())
    metric = "nll" if args.dataset in ["breast", "iris", "wine", "digits"] else "mse"
    cmd = (
        f"bayesmark-launch -n {args.budget} -r {args.repeat} -dir runs -b {_RUN_NAME} "
        f"-o {samplers} "
        f"-c {args.model} -d {args.dataset} -m {metric} --opt-root . -v"
    )
    subprocess.run(cmd, shell=True)


def partial_report(args: argparse.Namespace) -> None:

    eval_path = os.path.join("runs", _RUN_NAME, "eval")
    time_path = os.path.join("runs", _RUN_NAME, "time")
    studies = os.listdir(eval_path)
    summaries: List[pd.DataFrame] = []

    for study in studies:
        table_buffer: List[pd.DataFrame] = []
        column_buffer: List[str] = []
        for path in [eval_path, time_path]:
            with open(os.path.join(path, study), "r") as file:
                data = json.load(file)
                df = xarray.Dataset.from_dict(data["data"]).to_dataframe().droplevel("suggestion")

            for argument, meatadata in data["meta"]["args"].items():
                colname = argument[2:] if argument.startswith("--") else argument
                if colname not in column_buffer:
                    df[colname] = meatadata
                    column_buffer.append(colname)

            table_buffer.append(df)

        summary = pd.merge(*table_buffer, left_index=True, right_index=True)
        summaries.append(summary.reset_index())

    filename = f"{args.dataset}-{args.model}-partial-report.json"
    sampler_args = (
        pd.read_json("config.json")
        .T[1]
        .reset_index()
        .rename(columns={"index": "opt", 1: "sampler_args"})
    )
    (
        pd.concat(summaries)
        .reset_index(drop=True)
        .merge(sampler_args, on="opt")
        .to_json(os.path.join("partial", filename))
    )


def visuals(args: argparse.Namespace) -> None:

    filename = f"{args.dataset}-{args.model}-partial-report.json"
    df = pd.read_json(os.path.join("partial", filename))
    stats = (
        df.groupby(["opt", "iter"])
        .generalization.agg(["mean", "std"])
        # FIXME Better naming for those cols.
        .rename(columns={"mean": "best_mean", "std": "best_std"})
        .reset_index()
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    warmup = json.loads(args.plot_warmup)
    metric = df.metric[0]
    color_lookup = build_color_dict(sorted(df["opt"].unique()))

    for optimizer, summary in stats.groupby("opt"):
        color = color_lookup[optimizer]
        make_plot(summary, ax, optimizer, metric, warmup, color)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(f"Bayesmark-{args.dataset.capitalize()}-{args.model}")
    fig.savefig(f"plots/optuna-{args.dataset}-{args.model}-sumamry.png")


def make_plot(
    summary: pd.DataFrame,
    ax: Axes,
    optimizer: str,
    metric: str,
    plot_warmup: bool,
    color: np.ndarray,
) -> None:

    idx = 0 if plot_warmup else 10
    argpos = summary.best_mean.expanding().apply(np.argmin).astype(int)
    best_found = summary.best_mean.values[argpos.values]
    sdev = summary.best_std.values[argpos.values]

    if len(best_found) <= idx:
        return

    ax.fill_between(
        np.arange(len(best_found))[idx:],
        (best_found - sdev)[idx:],
        (best_found + sdev)[idx:],
        color=color,
        alpha=0.25,
        step="mid",
    )

    ax.plot(
        np.arange(len(best_found))[idx:],
        best_found[idx:],
        color=color,
        label=optimizer,
        drawstyle="steps-mid",
    )

    ax.set_xlabel("Budget", fontsize=10)
    ax.set_ylabel(f"Validation {metric.upper()}", fontsize=10)
    ax.grid(alpha=0.2)


def build_color_dict(names: Any) -> Any:

    # FIXME type hints
    norm = colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(norm, cm.tab20)
    color_dict = m.to_rgba(np.linspace(0, 1, len(names)))
    color_dict = dict(zip(names, color_dict))

    return color_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--model", type=str, default="kNN")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sampler-list", type=str, default="RandomSampler TPESampler")
    parser.add_argument(
        "--sampler-kwargs-list",
        type=str,
        default='{} {"multivariate":true,"constant_liar":true}',
    )
    parser.add_argument("--pruner-list", type=str, default="NopPruner")
    parser.add_argument("--pruner-kwargs-list", type=str, default="{}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-warmup", type=str)

    args = parser.parse_args()
    run(args)
    partial_report(args)
    visuals(args)
