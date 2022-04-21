import argparse
import json
import os
import subprocess
from typing import Any

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
        f"-o {samplers} "
        f"-c {args.model} -d {args.dataset} -m {metric} --opt-root . -v"
    )
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-agg -dir runs -b {_RUN_NAME}"
    subprocess.run(cmd, shell=True)

def partial_report(args: argparse.Namespace) -> None:

    eval_path = os.path.join("runs", _RUN_NAME, "eval")
    time_path = os.path.join("runs", _RUN_NAME, "time")
    studies = os.listdir(eval_path)
    summaries = []

    for study in studies:
        buff = []
        for rundata in [eval_path, time_path]:
            with open(os.path.join(rundata, study), "r") as file:
                data = json.load(file)
                df = (
                    xarray.Dataset.from_dict(data["data"])
                    .to_dataframe()
                    .droplevel("suggestion")
                )

            for k, v in data["meta"]["args"].items():
                colname = k[2:] if k.startswith("--") else k
                df[colname] = v

            buff.append(df)

        # FIXME No need to append meta cols twice.
        df = pd.merge(*buff, left_index=True, right_index=True, suffixes=("", "_drop"))
        to_drop = [col for col in df.columns if col.endswith("_drop")]
        df = df.drop(to_drop, axis=1).reset_index()
        summaries.append(df)

    filename = f"{args.dataset}-{args.model}-partial-report.json"
    (
        pd.concat(summaries)
        .reset_index(drop=True)
        .to_json(os.path.join("partial", filename))
    )


def visuals(args: argparse.Namespace) -> None:

    # https://github.com/uber/bayesmark/tree/master/notebooks
    db_root = os.path.abspath("runs")
    summary, _ = XRSerializer.load_derived(db_root, db=_RUN_NAME, key=cc.PERF_RESULTS)

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots()
    warmup = json.loads(args.plot_warmup)

    for benchmark in summary.coords["function"].values:
        for metric, ax in zip(["mean", "median"], axs):
            make_plot(summary, ax, benchmark, metric, warmup)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(benchmark)
    fig.savefig(f"plots/optuna-{args.dataset}-{args.model}-sumamry.png")


def make_plot(summary: Dataset, ax: Axes, func: str, metric: str, plot_warmup: bool) -> None:

    color = build_color_dict(summary.coords["optimizer"].values.tolist())
    optimizers = summary.coords["optimizer"].values
    idx = 0 if plot_warmup else 10

    for optimizer in optimizers:
        curr_ds = summary.sel(
            {"function": func, "optimizer": optimizer, "objective": cc.VISIBLE_TO_OPT}
        )

        if len(curr_ds.coords[cc.ITER].values) <= idx:
            continue

        ax.fill_between(
            curr_ds.coords[cc.ITER].values[idx:],
            curr_ds[f"{metric} LB"].values[idx:],
            curr_ds[f"{metric} UB"].values[idx:],
            color=color[optimizer],
            alpha=0.5,
        )
        ax.plot(
            curr_ds.coords["iter"].values[idx:],
            curr_ds[metric].values[idx:],
            color=color[optimizer],
            label=optimizer,
        )

    ax.set_xlabel("Budget", fontsize=10)
    ax.set_ylabel(f"{metric.capitalize()} score", fontsize=10)
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
