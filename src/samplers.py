from typing import Any

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import Trial


class SimulatedAnnealingSampler(BaseSampler):
    def __init__(self, temperature=100):
        self._rng = np.random.RandomState()
        self._temperature = temperature
        self._current_trial = None

    def sample_relative(
        self, study: Study, trial: Trial, search_space: dict
    ) -> dict[str, Any]:
        if not search_space:
            return {}

        if len(study.trials) < 2:
            # use random sampling for the first trial
            return self._sample_randomly(study, trial, search_space)

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
            if isinstance(param_distribution, optuna.distributions.UniformDistribution):
                current_value = self._current_trial.params[param_name]
                width = (param_distribution.high - param_distribution.low) * 0.1
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)
            elif isinstance(
                param_distribution, optuna.distributions.LogUniformDistribution
            ):
                current_value = self._current_trial.params[param_name]
                log_current = np.log(current_value)
                log_low, log_high = np.log(param_distribution.low), np.log(
                    param_distribution.high
                )
                width = (log_high - log_low) * 0.1
                neighbor_low = max(log_current - width, log_low)
                neighbor_high = min(log_current + width, log_high)
                params[param_name] = np.exp(
                    self._rng.uniform(neighbor_low, neighbor_high)
                )
            elif isinstance(
                param_distribution, optuna.distributions.IntUniformDistribution
            ):
                current_value = self._current_trial.params[param_name]
                width = max(
                    1, int((param_distribution.high - param_distribution.low) * 0.1)
                )
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.randint(neighbor_low, neighbor_high + 1)
            elif isinstance(
                param_distribution, optuna.distributions.CategoricalDistribution
            ):
                params[param_name] = self._rng.choice(param_distribution.choices)
            else:
                params[param_name] = self.sample_independent(
                    study, trial, param_name, param_distribution
                )

        return params

    def _sample_randomly(
        self, study: Study, trial: Trial, search_space: dict
    ) -> dict[str, Any]:
        params = {}
        for param_name, param_distribution in search_space.items():
            params[param_name] = self.sample_independent(
                study, trial, param_name, param_distribution
            )
        return params

    def infer_relative_search_space(self, study: Study, trial: Trial) -> dict[str, Any]:
        search_space_calculator = optuna.search_space.IntersectionSearchSpace(
            include_pruned=False
        )
        return search_space_calculator.calculate(study)

    def sample_independent(
        self, study: Study, trial: Trial, param_name: str, param_distribution: Any
    ) -> Any:
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
