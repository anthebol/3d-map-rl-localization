from typing import Any

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.study import Study


class SimulatedAnnealingSampler(BaseSampler):
    def __init__(self, temperature=100):
        self._rng = np.random.RandomState()
        self._temperature = temperature
        self._current_trial = None

    def infer_relative_search_space(self, study, trial):
        # create an IntersectionSearchSpace instance and calculate the search space
        search_space_calculator = optuna.search_space.IntersectionSearchSpace(
            include_pruned=False
        )
        search_space = search_space_calculator.calculate(study)
        return search_space

    def sample_relative(self, study: Study, trial, search_space) -> dict[str, Any]:
        if not search_space:
            return {}

        if len(study.trials) < 2:
            # use random sampling for the first trial
            params = {}
            for param_name in search_space:
                params[param_name] = self.sample_independent(
                    study, trial, param_name, search_space[param_name]
                )
            return params

        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            probability = 1.0
        else:
            probability = np.exp(
                (self._current_trial.value - prev_trial.value) / self._temperature
            )
        self._temperature *= 0.9

        if self._rng.uniform(0, 1) < probability:
            self._current_trial = prev_trial

        params = {}
        for param_name, param_distribution in search_space.items():
            if not isinstance(
                param_distribution, optuna.distributions.UniformDistribution
            ):
                raise NotImplementedError("Only suggest_float() is supported")

            current_value = self._current_trial.params[param_name]
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
