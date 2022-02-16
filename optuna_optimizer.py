from typing import Dict, List, Union
from collections import deque

import optuna
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


X = Dict[str, Union[int, float]]
optuna.logging.disable_default_handler()


class OptunaOptimizer(AbstractOptimizer):

    primary_import = "optuna"

    def __init__(self, api_config, **kwargs):

        super().__init__(api_config, **kwargs)
        # FIXME We don't have info about metric being optimized
        # so we can't determine direction.
        self.study = optuna.create_study(direction="maximize")
        self.current_trials = deque()

    def _suggest(self, trial: optuna.trial.Trial) -> X:

        suggestions: X = {}
        for name, config in self.api_config.items():
            low, high = config["range"]
            log = config["space"] == "log"  # FIXME What about logit space?

            if config["type"] == "real":
                param = trial.suggest_float(name, low, high, log=log)
            else:
                param = trial.suggest_int(name, low, high, log=log)

            suggestions[name] = param

        return suggestions

    def suggest(self, n_suggestions: int) -> List[X]:

        # n_suggestions is controlled by -p or --suggestions flag and defaults
        # to 1 then we can ask n_suggestions trials and run optimization for
        # them in parralel ... maybe?
        next_guess: List[X] = []
        for _ in range(n_suggestions):
            trial = self.study.ask()
            # FIXME: I'm not sure if order in `observe` is guaranteed
            # to be preserved when n_suggestions != 1.
            self.current_trials.append(trial.number)
            suggestions = self._suggest(trial)
            next_guess.append(suggestions)

        return next_guess

    def observe(self, X: List[X], y: List[float]) -> None:

        for objective_value in y:
            trial = self.current_trials.popleft()
            self.study.tell(trial, objective_value)


if __name__ == "__main__":
    experiment_main(OptunaOptimizer)
