import io
import itertools
import os
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

_LINE_BREAK = "\n"

Moments = Tuple[float, float]


class BaseMetric(ABC):
    @property
    def name(self) -> str:
        ...

    def calculate(self, data: pd.DataFrame) -> List[float]:
        ...


class BestValueMetric(BaseMetric):
    name = "Best value"

    def calculate(self, data: pd.DataFrame) -> Moments:
        return super().calculate(data)


class AUCMetric(BaseMetric):
    name = "AUC"

    def calculate(self, data: pd.DataFrame) -> List[float]:
        return super().calculate(data)


class ElapsedMetric(BaseMetric):
    name = "Elapsed"

    def calculate(self, data: pd.DataFrame) -> List[float]:
        return super().calculate(data)


class PartialReport:
    def __init__(self, data: pd.DataFrame, metrics: List[BaseMetric]) -> None:
        self._data = data
        self._metrics = metrics
        # TODO This class should also be able to provide report
        # metadate for recipes and stuff.

    @classmethod
    def from_json(cls, path: str, metrics: List[BaseMetric]) -> "PartialReport":

        data = pd.read_json(path)
        return cls(data, metrics)

    def add_metric(self, metric: BaseMetric) -> None:

        self._metrics.extend(metric)

    def summarize_solver(self, solver: str, metric: BaseMetric) -> Moments:

        solver_data = self._data[self._data.opt == solver]
        if solver_data.shape[0] == 0:
            raise ValueError(f"{solver} not found in report.")

        run_metrics = metric.calculate(solver_data)
        return np.mean(run_metrics), np.var(run_metrics)

    def average_performance(self, metric: BaseMetric) -> Dict[str, float]:

        performance: Dict[str, float] = {}
        for solver, data in self._data.groupby("opt"):
            run_metrics = metric.calculate(data)
            performance[solver] = run_metrics
        return performance


class DewanckerRanker:
    def __init__(self, metrics: List[BaseMetric]) -> None:
        self._metrics = metrics
        self._ranking = None
        self._borda = None

    @property
    def solvers(self) -> List[str]:

        if self._ranking is None:
            raise ValueError("Call rank first.")
        return self._ranking

    @property
    def borda(self) -> np.ndarray:

        if self._borda is None:
            raise ValueError("Call rank first.")
        return self._borda

    def rank(self, report: PartialReport) -> None:

        wins = defaultdict(int)
        for metric in self._metrics:
            summaries = report.average_performance(metric)
            for a, b in itertools.permutations(summaries.keys(), 2):
                _, p_val = mannwhitneyu(summaries[a], summaries[b], alternative="less")
                # FIXME alpha should be determined by num optimizers - see section 2.1
                if p_val < 0.05:
                    wins[a] += 1

            all_wins = [wins[optimizer] for optimizer in summaries]
            no_ties = len(all_wins) == len(np.unique(all_wins))
            if no_ties:
                break

        wins = {optimzier: wins[optimzier] for optimzier in summaries}
        sorted_wins = {k: v for k, v in sorted(wins.items(), key=lambda x: x[1])}
        self._ranking = list(reversed(sorted_wins.keys()))

        prev_wins = -1
        borda: List[int] = []
        for points, num_wins in enumerate(sorted_wins.values()):
            if num_wins == prev_wins:
                borda.append(borda[-1])
            else:
                borda.append(points)
            prev_wins = num_wins
        self._borda = np.array(borda)[::-1]


# TODO(xadrianzetx) Consider proper templating engine.
class BayesmarkReportBuilder:
    def __init__(self) -> None:

        self._solvers = Counter()
        self._datasets = Counter()
        self._models = Counter()
        self._problems_counter = 1
        self._problems_body = io.StringIO()

    def add_problem(
        self, name: str, scores: Dict[str, float]
    ) -> "BayesmarkReportBuilder":

        if self._problems_body.closed:
            self._problems_body = io.StringIO()

        problem_header = f"### ({self._problems_counter}) Problem: {name}" + _LINE_BREAK
        self._problems_body.write(problem_header)
        self._problems_body.write("".join([_LINE_BREAK, _TABLE_HEADER, _LINE_BREAK]))

        for idx, (solver, score) in enumerate(scores.items()):
            row = f"|{idx + 1}|{solver}|{score:.5f}|"
            self._problems_body.write("".join([row, _LINE_BREAK]))

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

        with open("report_template.md") as file:
            report_template = file.read()

        report = report_template.format(
            num_solvers=len(self._solvers),
            num_datasets=num_datasets,
            num_models=num_models,
            num_problems=num_datasets * num_models,
            leaderboards=self._problems_body.getvalue(),
        )

        self._problems_body.close()
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
