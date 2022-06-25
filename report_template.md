# Benchmark Result Report

* Number of Solvers: {{ report._solvers|length }}
* Number of Models: {{ report._models|length }}
* Number of Datasets: {{ report._datasets|length }}
* Number of Problems: {{ report._problems|length }}
* Metrics Precedence: {{ report._metric_precedence }}

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
{% for solver in report._solvers -%}
|{{ solver }}|{{ report._borda[solver] }}|{{ report._firsts[solver] }}|
{% endfor %}

## Problem Leaderboards

{% for problem in report._problems %}
### ({{ problem.number }}) Problem: {{ problem.name }}

|Ranking|Solver|{%- for metric in problem.metrics -%}{{ metric.name }} (avg +- std)|{% endfor %}
|:---|---:|{%- for _ in range(problem.metrics|length) -%}---:|{% endfor %}
{% for row in problem.rows -%}
|{{ row.pos }}|{{ row.solver }}|{%- for moments in row.moments -%}{{ moments[0] }} +- {{ moments[1] }}|{% endfor %}
{% endfor -%}
{% endfor %}

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
