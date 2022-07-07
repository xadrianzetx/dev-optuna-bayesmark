[Format reference.](https://github.com/optuna/optuna/issues/3464#issue-1199704879)
# Benchmark Result Report

* Number of Solvers: {{ report.solvers|length }}
* Number of Models: 7
* Number of Datasets: 5
* Number of Problems: {{ report.problems|length }}
* Metrics Precedence: {{ report.metric_precedence }}

Please refer to ["A Strategy for Ranking Optimizers using Multiple Criteria"][Dewancker, Ian, et al., 2016] for the ranking strategy used in this report.

[Dewancker, Ian, et al., 2016]: http://proceedings.mlr.press/v64/dewancker_strategy_2016.pdf

## Table of Contents

1. [Overall Results](#overall-results)
2. [Problem Leaderboards](#problem-leaderboards)
3. [Datasets](#datasets)
4. [Models](#models)

## Overall Results

|Solver|Borda|Firsts|
|:---|---:|---:|
{% for solver in report.solvers -%}
|{{ solver }}|{{ report.borda[solver] }}|{{ report.firsts[solver] }}|
{% endfor %}

## Problem Leaderboards

{% for problem in report.problems %}
{% set dataset, model = problem.name.split("-") %}
### ({{ problem.number }}) Problem: [{{ dataset }}](#{{ dataset|lower }})-[{{ model }}](#{{ model|lower }})

|Ranking|Solver|{%- for metric in problem.metrics -%}{{ metric.name }} (avg +- std)|{% endfor %}
|---:|:---|{%- for _ in range(problem.metrics|length) -%}---:|{% endfor %}
{% for study in problem.studies -%}
|{{ study.results.rank }}|[{{ study.solver.name }}](#id-{{ study.solver.id }}) ([study](#id-{{ study.id }}))|{{ study.results.values|join('|') }}|
{% endfor -%}
{% endfor %}
## Solvers
{% for _, solver in report.solvers.items() %}
### ID: {{ solver.id }}

recipe:
```json
{
    name: "{{ solver.name }}"
    optuna: {{ solver.args }}
}
```

specification:
```json
{
  "name": "{{ solver.name }}",
  "attrs": {
    "github": "https://github.com/optuna/optuna",
    "paper": "Akiba, Takuya, et al. \"Optuna: A next-generation hyperparameter optimization framework.\" Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019.",
    "version": "{{ solver.version }}"
  },
  "capabilities": [
    "UNIFORM_CONTINUOUS",
    "UNIFORM_DISCRETE",
    "LOG_UNIFORM_CONTINUOUS",
    "LOG_UNIFORM_DISCRETE",
    "CATEGORICAL",
    "CONDITIONAL",
    "MULTI_OBJECTIVE",
    "CONCURRENT"
  ]
}
```
{% endfor %}
## Studies
{% for study in report.studies %}
### ID: {{ study.id }}
{%- set problem_name = (report.problems|selectattr("id", "eq", study.problem_id)|list)[0].name %}
{% set dataset, model = problem_name.split("-") %}
- problem: [{{ dataset }}](#{{ dataset|lower }})-[{{ model }}](#{{ model|lower }})
- solver: [{{ study.solver.name }}](#id-{{ study.solver.id }})
- budget: {{ study.budget }}
- repeats: {{ study.repeats }}
- concurrency: 1
{% endfor %}
## Datasets
<!-- FIXME Reference user guide under full dataset name. -->
* #### [Breast](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)
* #### [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)
* #### [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
* #### [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)
* #### [Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)

## Models
<!-- FIXME Reference under full model name. -->
* #### [Ada](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* #### [DT](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* #### [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* #### [Linear](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* #### [MLP-sgd](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* #### [RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* #### [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
