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

* [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)
* [Diabetes Data Set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)
* [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
* [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)
* [Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)

## Models

* [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [Decistion Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Linear Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Multi-layer Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
"""


class BayesmarkReportBuilder:
    def __init__(self) -> None:

        self._solvers = Counter()
        self._datasets = Counter()
        self._models = Counter()
        self._problems_counter = 1
        self._problems_body = ""

    def add_problem(
        self, name: str, scores: Dict[str, float]
    ) -> "BayesmarkReportBuilder":

        self._problems_body += (
            f"### ({self._problems_counter}) Problem: {name}" + _LINE_BREAK
        )
        self._problems_body += "".join([_LINE_BREAK, _TABLE_HEADER, _LINE_BREAK])

        for idx, (solver, score) in enumerate(scores.items()):
            row = f"|{idx + 1}|{solver}|{score:.5f}|"
            self._problems_body += "".join([row, _LINE_BREAK])

        self._problems_body += _LINE_BREAK
        self._solvers.update(scores.keys())
        self._problems_counter += 1

        return self

    def add_dataset(self, dataset: str) -> "BayesmarkReportBuilder":

        self._datasets.update([dataset])
        return self

    def add_model(self, model: str) -> "BayesmarkReportBuilder":

        self._models.update([model])
        return self

    def assemble_report(self) -> str:

        num_datasets = len(self._datasets)
        num_models = len(self._models)

        report = _REPORT_TEMPLATE.format(
            num_solvers=len(self._solvers),
            num_datasets=num_datasets,
            num_models=num_models,
            num_problems=num_datasets * num_models,
            leaderboards=self._problems_body,
        )

        return report


def build_report() -> None:

    report_builder = BayesmarkReportBuilder()
    for partial_name in os.listdir("partial"):
        dataset, model, *_ = partial_name.split("-")
        path = os.path.join("partial", partial_name)

        with open(path) as file:
            scores = json.load(file)
            problem_name = f"{dataset.capitalize()}-{model}"
            report_builder = (
                report_builder.add_problem(problem_name, scores)
                .add_dataset(dataset)
                .add_model(model)
            )

    report = report_builder.assemble_report()
    with open(os.path.join("report", "benchmark-report.md"), "w") as file:
        file.write(report)


if __name__ == "__main__":
    build_report()
