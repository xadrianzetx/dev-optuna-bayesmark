import argparse
import json
import os
import subprocess
from typing import Any

import bayesmark.constants as cc
import matplotlib.pyplot as plt
import numpy as np
from bayesmark.constants import (ITER, METHOD, OBJECTIVE, TEST_CASE,
                                 VISIBLE_TO_OPT)
from bayesmark.serialize import XRSerializer
from matplotlib import cm, colors


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
    cmd = (
        f"bayesmark-launch -n {args.budget} -r {args.repeat} -dir runs -b bo_debug_run "
        f"-o RandomSearch {samplers} "
        f"-c kNN -d {args.dataset} -m acc --opt-root . -v"
    )
    subprocess.run(cmd, shell=True)

    cmd = "bayesmark-agg -dir runs -b bo_debug_run"
    subprocess.run(cmd, shell=True)

    cmd = "bayesmark-anal -dir runs -b bo_debug_run -v"
    subprocess.run(cmd, shell=True)


def visuals(args: argparse.Namespace) -> None:

    # https://github.com/uber/bayesmark/tree/master/notebooks
    db_root = os.path.abspath("runs")
    dbid = "bo_debug_run"

    # FIXME correct key?
    summary_ds, _ = XRSerializer.load_derived(db_root, db=dbid, key=cc.PERF_RESULTS)
    method_to_rgba = build_color_dict(summary_ds.coords[METHOD].values.tolist())
    method_list = summary_ds.coords[METHOD].values

    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
    for func_name in summary_ds.coords[TEST_CASE].values:
        plt.sca(axarr[0])
        for method_name in method_list:
            curr_ds = summary_ds.sel(
                {TEST_CASE: func_name, METHOD: method_name, OBJECTIVE: VISIBLE_TO_OPT}
            )

            plt.fill_between(
                curr_ds.coords[ITER].values,
                curr_ds[cc.LB_MED].values,
                curr_ds[cc.UB_MED].values,
                color=method_to_rgba[method_name],
                alpha=0.5,
            )
            plt.plot(
                curr_ds.coords[ITER].values,
                curr_ds[cc.PERF_MED].values,
                color=method_to_rgba[method_name],
                label=method_name,
            )
        plt.xlabel("evaluation", fontsize=10)
        plt.ylabel("median score", fontsize=10)
        plt.title(func_name)
        plt.legend()

        plt.sca(axarr[1])
        for method_name in method_list:
            curr_ds = summary_ds.sel(
                {TEST_CASE: func_name, METHOD: method_name, OBJECTIVE: VISIBLE_TO_OPT}
            )

            plt.fill_between(
                curr_ds.coords[ITER].values,
                curr_ds[cc.LB_MEAN].values,
                curr_ds[cc.UB_MEAN].values,
                color=method_to_rgba[method_name],
                alpha=0.5,
            )
            plt.plot(
                curr_ds.coords[ITER].values,
                curr_ds[cc.PERF_MEAN].values,
                color=method_to_rgba[method_name],
                label=method_name,
            )
        plt.xlabel("evaluation", fontsize=10)
        plt.ylabel("mean score", fontsize=10)
        plt.title(func_name)
        plt.legend()

    dataset = args.dataset
    model = "knn"  # TODO add to matrix
    fig.savefig(f"optuna-{dataset}-{model}-sumamry.png")


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
    try:
        visuals()
    except Exception as e:
        print(f"Caught: {str(e)}")
