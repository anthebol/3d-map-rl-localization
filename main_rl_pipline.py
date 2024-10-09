import logging

import gymnasium as gym
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from tensorboardX import SummaryWriter

from src.callbacks.eval_callback import EvalCallback
from src.callbacks.logger_reward_callback import LoggerRewardCallback
from src.callbacks.test_callback import TestCallback
from src.envs.satellite_env import SatelliteEnv
from src.models.rl_policy import SatelliteRLPolicy
from src.optimization.hyperparameter_tuning import objective
from src.optimization.optuna_sampler import SimulatedAnnealingSampler
from src.utils.image_loader import env_image, single_target, test_targets
from src.utils.logging_setup import setup_logging, setup_run_directories

"""
This script implements the complete reinforcement learning pipeline for training
and evaluating a model to locate specific targets within satellite imagery. It includes:
- Hyperparameter optimization using Optuna
- Model training using PPO (Proximal Policy Optimization)
- Evaluation on training and test sets
- Logging and TensorBoard integration

The pipeline is designed for the SatelliteEnv environment and uses a custom
SatelliteRLPolicy. It has been most successful with the following target images:
- target_009_imperial_main_entrance
- target_003_statue
- test_target_006_victoria_albert_museum_rooftop

These targets consistently achieved a final cosine similarity of over 0.90,
indicating highly accurate localization.

The script handles the entire process from environment setup and hyperparameter
tuning to model training, saving, and final evaluation on both training and test sets.

Usage:
    Run this script directly to execute the full RL pipeline.
"""


def main():
    base_log_dir = "./logs"
    run_dirs = setup_run_directories(base_log_dir)

    setup_logging(run_dirs["reward_log_path"])
    logger = logging.getLogger(__name__)

    logger.info("Starting the main process")
    logger.info(f"Reward log file: {run_dirs['reward_log_path']}")
    logger.info(f"TensorBoard log directory: {run_dirs['tensorboard_dir']}")
    logger.info(f"Model save path: {run_dirs['model_save_path']}")

    writer = SummaryWriter(log_dir=run_dirs["tensorboard_dir"])
    logger.info("TensorBoard writer initialized")

    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(
        storage="sqlite:///rl.db",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=2)

    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best reward: {study.best_trial.value}")

    env = SatelliteEnv(env_image=env_image, target_images=single_target)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2500)
    env = Monitor(env)

    eval_env = SatelliteEnv(env_image=env_image, target_images=single_target)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=2500)
    eval_env = Monitor(eval_env)

    test_env = SatelliteEnv(env_image=env_image, target_images=test_targets)
    test_env = gym.wrappers.TimeLimit(test_env, max_episode_steps=2500)
    test_env = Monitor(test_env)

    initial_learning_rate = study.best_trial.params["learning_rate"]
    end_learning_rate = initial_learning_rate / 10

    lr_schedule = get_linear_fn(initial_learning_rate, end_learning_rate, 1.0)

    model = PPO(
        SatelliteRLPolicy,
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=run_dirs["tensorboard_dir"],
        learning_rate=lr_schedule,
        gamma=study.best_trial.params["gamma"],
        gae_lambda=study.best_trial.params["gae_lambda"],
        clip_range=study.best_trial.params["clip_range"],
        ent_coef=study.best_trial.params["ent_coef"],
        vf_coef=study.best_trial.params["vf_coef"],
        max_grad_norm=study.best_trial.params["max_grad_norm"],
        batch_size=study.best_trial.params["batch_size"],
        n_steps=study.best_trial.params["n_steps"],
        n_epochs=study.best_trial.params["n_epochs"],
    )

    logger_callback = LoggerRewardCallback()
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=10,
        tensorboard_log=writer,
    )
    test_callback = TestCallback(
        test_env,
        test_freq=50000,
        n_test_episodes=25,
        tensorboard_log=writer,
    )

    logger.info("Starting model training")
    model.learn(
        total_timesteps=100000,
        callback=[logger_callback, eval_callback, test_callback],
    )
    logger.info("Model training completed")

    logger.info("Evaluating model immediately after training...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=25,
        deterministic=True,
    )
    logger.info(
        f"Model performance on train/eval set: {mean_reward:.2f} +/- {std_reward:.2f}"
    )

    model.save(run_dirs["model_save_path"])
    logger.info(f"Final model saved to {run_dirs['model_save_path']}")

    loaded_model = PPO.load(run_dirs["model_save_path"])

    logger.info("Evaluating model on test set...")
    mean_reward, std_reward = evaluate_policy(
        loaded_model,
        test_env,
        n_eval_episodes=25,
        deterministic=True,
    )
    logger.info(
        f"Model performance on test set: {mean_reward:.2f} +/- {std_reward:.2f}"
    )

    writer.close()
    logger.info("TensorBoard writer closed")
    logger.info("Main process completed")


if __name__ == "__main__":
    main()
