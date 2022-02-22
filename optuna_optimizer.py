from typing import Dict, List, Union

import optuna
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from optuna.pruners import NopPruner
from optuna.samplers import RandomSampler, TPESampler, CmaEsSampler

X = Dict[str, Union[int, float]]
SAMPLERS = {
    "RandomSampler": RandomSampler,
    "TPESampler": TPESampler,
    "CmaEsSampler": CmaEsSampler,
}
PRUNERS = {"NopPruner": NopPruner}


class OptunaOptimizer(AbstractOptimizer):

    primary_import = "optuna"

    def __init__(self, api_config, **kwargs):

        super().__init__(api_config, **kwargs)

        sampler = SAMPLERS[kwargs.get("sampler")]
        sampler_kwargs = kwargs.get("sampler_kwargs")
        pruner = PRUNERS[kwargs.get("pruner")]
        pruner_kwargs = kwargs.get("pruner_kwargs")

        # We are using negative log-likelihood for classification
        # and MSE for regression. Both are minimized.
        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler(**sampler_kwargs),
            pruner=pruner(**pruner_kwargs),
        )
        self.current_trials: Dict[int, int] = {}

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

        next_guess: List[X] = []
        for _ in range(n_suggestions):
            trial = self.study.ask()
            suggestions = self._suggest(trial)
            sid = hash(frozenset(suggestions.items()))
            self.current_trials[sid] = trial.number
            next_guess.append(suggestions)

        return next_guess

    def observe(self, X: List[X], y: List[float]) -> None:

        for params, objective_value in zip(X, y):
            sid = hash(frozenset(params.items()))
            trial = self.current_trials.pop(sid)
            self.study.tell(trial, objective_value)


if __name__ == "__main__":
    optuna.logging.disable_default_handler()
    experiment_main(OptunaOptimizer)
