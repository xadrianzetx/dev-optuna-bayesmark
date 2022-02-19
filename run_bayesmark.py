import argparse
import json
import subprocess


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
    subprocess.run(cmd)

    cmd = "bayesmark-agg -dir runs -b bo_debug_run"
    subprocess.run(cmd)

    cmd = "bayesmark-anal -dir runs -b bo_debug_run -v"
    subprocess.run(cmd)


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
