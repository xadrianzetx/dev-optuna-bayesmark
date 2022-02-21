#!/bin/bash

bayesmark-launch -n 15 -r 3 -dir runs -o RandomSearch TPESampler-NopPruner-Optuna -c kNN -d iris -m nll --opt-root . -v -b bo_debug_run
bayesmark-agg -dir runs -b bo_debug_run
bayesmark-anal -dir runs -b bo_debug_run -v
