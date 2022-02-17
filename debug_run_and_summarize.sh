#!/bin/bash

bayesmark-launch -n 80 -r 30 -dir runs -o RandomSearch OptunaSearch -c kNN -d iris -m acc --opt-root . -v -b bo_debug_run
bayesmark-agg -dir runs -b bo_debug_run
bayesmark-anal -dir runs -b bo_debug_run -v
