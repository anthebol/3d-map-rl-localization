import logging
import os
from datetime import datetime

import gymnasium as gym
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from envs.map_rl_network import MapRLPolicy
from envs.satellite_env import SatelliteEnv
from utils.eval_callback import PeriodicEvalCallback
from utils.load_data import load_image, load_target_images
from utils.test_callback import TestCallback

TENSORBOARD_LOG_DIR = "./tensorboard_logs/"
CHECKPOINT_DIR = "./checkpoints/"
FINAL_MODEL_PATH = "./final_model.zip"

env_image_path = os.path.join("data", "env", "env_image_exibition_road.jpg")
env_image = load_image(env_image_path)
train_eval_targets = load_target_images(os.path.join("data", "train_eval"))
test_targets = load_target_images(os.path.join("data", "test"))
single_target = {"high_performing_image": train_eval_targets["target_003_statue"]}


def create_ppo_model(env: SatelliteEnv, trial: optuna.Trial) -> PPO:
    initial_learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    end_learning_rate = initial_learning_rate / 10

    lr_schedule = get_linear_fn(initial_learning_rate, end_learning_rate, 1.0)

    return PPO(
        MapRLPolicy,
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
    eval_callback = PeriodicEvalCallback(
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


class LoggerRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggerRewardCallback, self).__init__(verbose)
        self.episode_reward = 0
        self.episode_count = 0

    def _init_callback(self) -> None:
        self.episode_reward = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        cosine_similarity = info.get("cosine_similarity", None)
        current_target = info.get("current_target", None)

        self.episode_reward += reward

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        self.logger.record("reward", reward)
        self.logger.record("cosine_similarity", cosine_similarity)
        self.logger.record("current_target", current_target)

        print(
            f"{current_time} - INFO - Step reward: {reward:.8f}, "
            f"Cosine similarity: {cosine_similarity:.8f}, "
            f"Cumulative reward: {self.episode_reward:.8f}, "
            f"Current target: {current_target}"
        )

        if self.locals["dones"][0]:
            self.logger.record("episode_reward", self.episode_reward)
            print(
                f"{current_time} - INFO - Episode {self.episode_count} finished. "
                f"Total reward: {self.episode_reward:.8f}"
            )
            self.episode_count += 1
            self.episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        self.logger.dump(step=self.num_timesteps)


def train_model(
    model: PPO,
    env: SatelliteEnv,
    eval_env: SatelliteEnv,
    test_env: SatelliteEnv,
    total_timesteps: int,
) -> PPO:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger_callback = LoggerRewardCallback()
    eval_callback = PeriodicEvalCallback(
        eval_env,
        eval_freq=50000,
        n_eval_episodes=len(train_eval_targets),
    )
    test_callback = TestCallback(
        test_env,
        test_freq=50000,
        n_test_episodes=len(test_targets),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger_callback, eval_callback, test_callback],
    )

    return model


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best reward: {study.best_trial.value}")

    env = SatelliteEnv(env_image=env_image, target_images=train_eval_targets)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1500)
    env = Monitor(env)

    eval_env = SatelliteEnv(env_image=env_image, target_images=train_eval_targets)
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=1500)
    eval_env = Monitor(eval_env)

    test_env = SatelliteEnv(env_image=env_image, target_images=test_targets)
    test_env = gym.wrappers.TimeLimit(test_env, max_episode_steps=1500)
    test_env = Monitor(test_env)

    best_model = create_ppo_model(env, study.best_trial)
    final_model = train_model(
        best_model,
        env,
        eval_env,
        test_env,
        total_timesteps=200000,
    )

    final_model.save(FINAL_MODEL_PATH)
    logger.info(f"Final model saved to {FINAL_MODEL_PATH}")

    mean_reward, std_reward = evaluate_policy(
        final_model,
        eval_env,
        n_eval_episodes=len(train_eval_targets),
        deterministic=True,
    )
    logger.info(
        f"Final model performance on train/eval set: {mean_reward:.2f} +/- {std_reward:.2f}"
    )

    loaded_model = PPO.load(FINAL_MODEL_PATH)
    mean_reward, std_reward = evaluate_policy(
        loaded_model,
        test_env,
        n_eval_episodes=len(test_targets),
        deterministic=True,
    )
    logger.info(
        f"Final model performance on test set: {mean_reward:.2f} +/- {std_reward:.2f}"
    )


if __name__ == "__main__":
    main()
