import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import gymnasium as gym
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from tensorboardX import SummaryWriter

from envs.map_rl_network import MapRLPolicy
from envs.satellite_env import SatelliteEnv
from samplers import SimulatedAnnealingSampler
from utils.eval_callback import PeriodicEvalCallback
from utils.load_data import load_image, load_target_images
from utils.test_callback import TestCallback
from utils.training import LoggerRewardCallback, objective

env_image_path = os.path.join("data", "env", "env_image_exibition_road.jpg")
env_image = load_image(env_image_path)
train_eval_targets = load_target_images(os.path.join("data", "train_eval"))
test_targets = load_target_images(os.path.join("data", "test"))
single_target = {"high_performing_image": train_eval_targets["target_003_statue"]}
# target_009_imperial_main_entrance
# target_003_statue
# test_target_006_victoria_albert_museum_rooftop


def setup_logging(log_file):
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=20,
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    sys.stdout = LoggerWriter(root_logger.info)
    sys.stderr = LoggerWriter(root_logger.error)


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != "\n":
            self.level(message)

    def flush(self):
        pass


def get_next_run_number(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    if not existing_runs:
        return 1
    return max([int(d.split("_")[1]) for d in existing_runs]) + 1


def setup_run_directories(base_log_dir):
    run_number = get_next_run_number(base_log_dir)
    run_dir = os.path.join(base_log_dir, f"run_{run_number}")

    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    rewards_dir = os.path.join(run_dir, "rewards")
    models_dir = os.path.join(run_dir, "models")

    for directory in [tensorboard_dir, rewards_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)

    reward_log_path = os.path.join(rewards_dir, "reward_logs.txt")
    model_save_path = os.path.join(models_dir, "ppo_satellite_final.zip")

    return {
        "run_number": run_number,
        "tensorboard_dir": tensorboard_dir,
        "reward_log_path": reward_log_path,
        "model_save_path": model_save_path,
    }


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
        MapRLPolicy,
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
    eval_callback = PeriodicEvalCallback(
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
