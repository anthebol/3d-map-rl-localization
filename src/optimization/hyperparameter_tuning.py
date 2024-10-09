import os

import gymnasium as gym
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from src.callbacks.eval_callback import EvalCallback
from src.callbacks.logger_reward_callback import LoggerRewardCallback
from src.callbacks.test_callback import TestCallback
from src.envs.satellite_env import SatelliteEnv
from src.models.rl_policy import SatelliteRLPolicy
from src.utils.image_loader import TENSORBOARD_LOG_DIR, env_image, single_target


def create_ppo_model(env: SatelliteEnv, trial: optuna.Trial) -> PPO:
    initial_learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    end_learning_rate = initial_learning_rate / 10

    lr_schedule = get_linear_fn(initial_learning_rate, end_learning_rate, 1.0)

    return PPO(
        SatelliteRLPolicy,
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=lr_schedule,
        gamma=trial.suggest_float("gamma", 0.9, 0.9999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.999),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        ent_coef=trial.suggest_loguniform("ent_coef", 1e-8, 0.01),
        vf_coef=trial.suggest_float("vf_coef", 0.5, 0.9),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 1.0),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        n_epochs=trial.suggest_int("n_epochs", 5, 15),
    )


def objective(trial):

    env = SatelliteEnv(env_image=env_image, target_images=single_target)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2500)
    env = Monitor(env)

    eval_env = SatelliteEnv(env_image=env_image, target_images=single_target)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=2500)
    eval_env = Monitor(eval_env)

    test_env = SatelliteEnv(env_image=env_image, target_images=single_target)
    test_env = gym.wrappers.TimeLimit(test_env, max_episode_steps=2500)
    test_env = Monitor(test_env)

    model = create_ppo_model(env, trial)

    logger_callback = LoggerRewardCallback()
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=25,
    )
    test_callback = TestCallback(
        test_env,
        test_freq=25000,
        n_test_episodes=14,
    )

    model.learn(
        total_timesteps=50000,
        callback=[logger_callback, eval_callback, test_callback],
    )

    print("Training completed. Starting final evaluation...")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
    )
    print(f"Final evaluation completed: Mean reward: {mean_reward}, Std: {std_reward}")

    return mean_reward
