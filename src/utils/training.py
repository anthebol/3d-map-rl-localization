import os
import optuna
import torch
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.satellite_env import SatelliteEnv
from envs.map_rl_network import MapRLPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboardX import SummaryWriter

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
    """Create a PPO model with hyperparameters suggested by Optuna."""
    return PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=TENSORBOARD_LOG_DIR,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2),
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


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization."""
    env = SatelliteEnv()
    model = create_ppo_model(env, trial)
    
    model.learn(total_timesteps=100)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    return mean_reward


class LoggerRewardCallback(BaseCallback):
    def __init__(self, logger, writer: SummaryWriter, save_freq: int = 1000):
        super(LoggerRewardCallback, self).__init__()
        self.logger = logger
        self.writer = writer
        self.save_freq = save_freq
        self.episode_count = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ppo_satellite_step_{self.n_calls}.zip")
            self.model.save(path)
            self.logger.info(f"Model checkpoint saved to {path}")
        return True

    def _on_rollout_end(self) -> None:
        self.logger.info(f"Episode reward: {self.episode_reward}")
        self.writer.add_scalar('Training/Episode_Reward', self.episode_reward, self.episode_count)
        self.episode_count += 1
        self.episode_reward = 0

    def _on_training_end(self) -> None:
        self.writer.close()
        self.model.save(FINAL_MODEL_PATH)
        self.logger.info(f"Final model saved to {FINAL_MODEL_PATH}")


def train_model(model: PPO, env: SatelliteEnv, logger, total_timesteps: int = 20000) -> PPO:
    """Train the model and log results."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    callback = LoggerRewardCallback(logger, writer)
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=10, deterministic=True)
    logger.info(f"Final model performance: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()