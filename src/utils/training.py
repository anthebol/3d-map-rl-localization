import logging
import os
from datetime import datetime
from typing import Callable

import gymnasium as gym
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from envs.map_rl_network import MapRLPolicy
from envs.satellite_env import SatelliteEnv
from utils.eval_callback import PeriodicEvalCallback

TENSORBOARD_LOG_DIR = "./tensorboard_logs/"
CHECKPOINT_DIR = "./checkpoints/"
FINAL_MODEL_PATH = "./final_model.zip"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate
    :return: A function that computes the learning rate given the remaining progress
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def create_ppo_model(env: SatelliteEnv, trial: optuna.Trial) -> PPO:
    return PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 5e-3),
        gamma=trial.suggest_float("gamma", 0.9, 0.9999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.0),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        ent_coef=trial.suggest_loguniform("ent_coef", 1e-8, 0.1),
        vf_coef=trial.suggest_float("vf_coef", 0.5, 1.0),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 1.0),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
    )


def objective(trial):
    env = SatelliteEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env)

    model = PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard_logs/",
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.9, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.999),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        ent_coef=trial.suggest_float("ent_coef", 1e-5, 0.1, log=True),
        vf_coef=trial.suggest_float("vf_coef", 0.5, 0.9),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.5),
        batch_size=32,
        n_steps=128,
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
    )

    logger_callback = LoggerRewardCallback()
    eval_callback = PeriodicEvalCallback(env, eval_freq=10000, n_eval_episodes=5)

    model.learn(total_timesteps=100000, callback=[logger_callback, eval_callback])

    print("Training completed. Starting evaluation...")

    # Create a separate environment for evaluation
    eval_env = SatelliteEnv()
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=1000)
    eval_env = Monitor(eval_env)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
    print(f"Evaluation completed: Mean reward: {mean_reward}, Std: {std_reward}")

    return mean_reward


class LoggerRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggerRewardCallback, self).__init__(verbose)
        self.episode_reward = 0
        self.episode_count = 0

    def _init_callback(self) -> None:
        # Initialize the variables when the training starts
        self.episode_reward = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Get the reward and cosine similarity for the current step
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]  # Get the info dictionary
        cosine_similarity = info.get("cosine_similarity", None)

        self.episode_reward += reward

        # Log the current reward and cosine similarity
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        self.logger.record("reward", reward)
        self.logger.record("cosine_similarity", cosine_similarity)
        print(
            f"{current_time} - INFO - Step reward: {reward:.8f}, Cosine similarity: {cosine_similarity:.8f}, Cumalative reward: {self.episode_reward:.8f}"
        )

        # Check if the episode has ended
        if self.locals["dones"][0]:
            self.logger.record("episode_reward", self.episode_reward)
            print(
                f"{current_time} - INFO - Episode {self.episode_count} finished. Total reward: {self.episode_reward:.8f}"
            )

            self.episode_count += 1
            self.episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        self.logger.dump(step=self.num_timesteps)


def train_model(model: PPO, env: SatelliteEnv, total_timesteps) -> PPO:
    """Train the model and log results."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger_callback = LoggerRewardCallback()
    eval_callback = PeriodicEvalCallback(env, eval_freq=10000, n_eval_episodes=5)

    model.learn(total_timesteps=100000, callback=[logger_callback, eval_callback])

    return model


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best reward: {study.best_trial.value}")

    # Train the best model
    env = SatelliteEnv()
    best_model = create_ppo_model(env, study.best_trial)
    final_model = train_model(best_model, env, logger)

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        final_model,
        env,
        n_eval_episodes=1,
        deterministic=True,
    )
    logger.info(f"Final model performance: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
