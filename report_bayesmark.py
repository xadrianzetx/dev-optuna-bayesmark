import json
import os
from collections import Counter
from typing import Dict

# TODO(xadrianzetx) Consider proper templating engine.
_LINE_BREAK = "\n"
_TABLE_HEADER = "|Ranking|Solver|Score|\n|---:|:---|---:|"
_REPORT_TEMPLATE = """
# Benchmark Result Report

* Number of Solvers: {num_solvers}
* Number of Models: {num_models}
* Number of Datasets: {num_datasets}
* Number of Problems: {num_problems}

Final score for each problem is calculated as `100 x (1-loss)`. Solver with lowest score in each problem wins. For more details visit [bayesmark docs.](https://bayesmark.readthedocs.io/en/stable/scoring.html)

## Table of Contents

1. [Problem Leaderboards](#problem-leaderboards)
2. [Datasets](#datasets)
3. [Models](#models)

## Problem Leaderboards

{leaderboards}

## Datasets

{datasets}

## Models

{models}
"""


class BayesmarkReportBuilder:
    def __init__(self) -> None:

        self._solvers = Counter()
        self._datasets = Counter()
        self._models = Counter()
        self._problems_counter = 1
        self._problems_body = ""

    def add_problem(self, name: str, partial: Dict[str, float]) -> None:

        self._problems_body += (
            f"### ({self._problems_counter}) Problem: {name}" + _LINE_BREAK
        )
        self._problems_body += "".join([_LINE_BREAK, _TABLE_HEADER, _LINE_BREAK])

        for idx, (solver, score) in enumerate(partial.items()):
            row = f"|{idx + 1}|{solver}|{score:.5f}|"
            self._problems_body += "".join([row, _LINE_BREAK])

        self._problems_body += _LINE_BREAK
        self._solvers.update(partial.keys())
        self._problems_counter += 1

    def add_dataset(self, dataset: str) -> None:

        self._datasets.update([dataset])

    def add_model(self, model: str) -> None:

        self._models.update([model])

    def assemble_report(self) -> str:

        num_datasets = len(self._datasets)
        num_models = len(self._models)

        report = _REPORT_TEMPLATE.format(
            num_solvers=len(self._solvers),
            num_datasets=num_datasets,
            num_models=num_models,
            num_problems=num_datasets * num_models,
            leaderboards=self._problems_body,
            datasets="foo",
            models="foo",
        )

        return report


def build_report() -> None:

    report_builder = BayesmarkReportBuilder()
    for partial_name in os.listdir("partial"):
        dataset, model, *_ = partial_name.split("-")
        report_builder.add_dataset(dataset)
        report_builder.add_model(model)

        path = os.path.join("partial", partial_name)
        with open(path) as file:
            partial = json.load(file)
            problem_name = f"{dataset.capitalize()}-{model}"
            report_builder.add_problem(problem_name, partial)

    report = report_builder.assemble_report()
    with open(os.path.join("report", "benchmark-report.md"), "w") as file:
        file.write(report)


if __name__ == "__main__":
    build_report()
