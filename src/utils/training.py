from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from envs.satellite_env import SatelliteEnv
from envs.map_rl_network import MapRLPolicy
from evaluate import evaluate_policy


def objective(trial):
    env = SatelliteEnv()
    model = PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="./tensorboard_logs/",
        learning_rate=trial.suggest_uniform("learning_rate", 1e-4, 1e-2),
        gamma=trial.suggest_float("gamma", 0.5, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.5),
        ent_coef=trial.suggest_float("ent_coef", 0., 0.5),
        vf_coef=trial.suggest_float("vf_coef", 0.3, 0.8),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.8),
        batch_size=16,
        n_steps=100,
        stats_window_size=1,
        n_epochs=1,
    )

    model.learn(total_timesteps=100)

    episode_rewards, _, _, _ = evaluate_policy(model, env, n_eval_episodes=1, total_timestep=100)
    mean_reward = sum(episode_rewards) / len(episode_rewards)

    return mean_reward


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率调度函数。
    :param initial_value: 初始学习率
    :return: 一个函数,该函数接受当前进度(0到1之间的浮点数)作为输入,返回对应的学习率
    """
    def func(progress_remaining: float) -> float:
        """
        进度从1(开始)线性减少到0(结束)。
        """
        return progress_remaining * initial_value

    return func


def define_objective(trial):
    env = SatelliteEnv()

    initial_learning_rate = 0.001
    lr_schedule = linear_schedule(initial_learning_rate)

    model = PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="./tensorboard_logs/",
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2,),
        gamma=trial.suggest_float("gamma", 0.9, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.5),
        ent_coef=trial.suggest_float("ent_coef", 0., 0.5),
        vf_coef=trial.suggest_float("vf_coef", 0.3, 0.8),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.8),
        batch_size=16,
        n_steps=100,
        stats_window_size=1,
        n_epochs=1,
    )

    return model


class LoggerRewardCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super(LoggerRewardCallback, self).__init__(verbose)
        self.rewards = []
        self._logger = logger

    def _on_step(self):
        # get reward for the current step
        reward = self.locals["rewards"]
        self.rewards.append(np.sum(reward))
        return True

    def _on_rollout_end(self):
        episode_reward = np.sum(self.rewards)
        
        self._logger.info(f"Episode reward: {episode_reward}")
        self.rewards = []

