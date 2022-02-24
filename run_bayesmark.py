import argparse
import json
import os
import subprocess
from typing import Any

import bayesmark.constants as cc
import matplotlib.pyplot as plt
import numpy as np
from bayesmark.serialize import XRSerializer
from matplotlib import cm, colors
from matplotlib.axes import Axes
from xarray import Dataset


_RUN_NAME = "bo_optuna_run"


def run(args: argparse.Namespace) -> None:

    sampler_list = args.sampler_list.split()
    sampler_kwargs_list = args.sampler_kwargs_list.split()
    pruner_list = args.pruner_list.split()
    pruner_kwargs_list = args.pruner_kwargs_list.split()

    config = {}
    for sampler, sampler_kwargs in zip(sampler_list, sampler_kwargs_list):
        for pruner, pruner_kwargs in zip(pruner_list, pruner_kwargs_list):
            optimizer_name = f"{sampler}-{pruner}-Optuna"
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
        f"-o RandomSearch {samplers} "
        f"-c {args.model} -d {args.dataset} -m {metric} --opt-root . -v"
    )
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-agg -dir runs -b {_RUN_NAME}"
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-anal -dir runs -b {_RUN_NAME} -v"
    subprocess.run(cmd, shell=True)


def visuals(args: argparse.Namespace) -> None:

    # https://github.com/uber/bayesmark/tree/master/notebooks
    db_root = os.path.abspath("runs")
    summary, _ = XRSerializer.load_derived(db_root, db=_RUN_NAME, key=cc.PERF_RESULTS)

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    for benchmark in summary.coords["function"].values:
        for metric, ax in zip(["mean", "median"], axs):
            make_plot(summary, ax, benchmark, metric)

    dataset = args.dataset
    model = args.model
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(benchmark)
    fig.savefig(f"out/optuna-{dataset}-{model}-sumamry.png")


def make_plot(summary: Dataset, ax: Axes, func: str, metric: str) -> None:

    color = build_color_dict(summary.coords["optimizer"].values.tolist())
    optimizers = summary.coords["optimizer"].values

    for optimizer in optimizers:
        curr_ds = summary.sel(
            {"function": func, "optimizer": optimizer, "objective": cc.VISIBLE_TO_OPT}
        )

        ax.fill_between(
            curr_ds.coords[cc.ITER].values,
            curr_ds[f"{metric} LB"].values,
            curr_ds[f"{metric} UB"].values,
            color=color[optimizer],
            alpha=0.5,
        )
        ax.plot(
            curr_ds.coords["iter"].values,
            curr_ds[metric].values,
            color=color[optimizer],
            label=optimizer,
        )

    ax.set_xlabel("Budget", fontsize=10)
    ax.set_ylabel(f"{metric.capitalize()} score", fontsize=10)
    ax.grid(alpha=0.2)
    ax.label_outer()


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

    args = parser.parse_args()
    run(args)
    visuals(args)
