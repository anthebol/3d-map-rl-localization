import logging
import os

import gymnasium as gym
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tensorboardX import SummaryWriter

from envs.map_rl_network import MapRLPolicy
from envs.satellite_env import SatelliteEnv
from samplers import SimulatedAnnealingSampler
from utils.training import objective


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
    eval_dir = os.path.join(run_dir, "eval")
    eval_plots_dir = os.path.join(eval_dir, "plots")
    rewards_dir = os.path.join(run_dir, "rewards")
    models_dir = os.path.join(run_dir, "models")
    checkpoints_dir = os.path.join(models_dir, "checkpoints")

    for directory in [
        tensorboard_dir,
        eval_dir,
        eval_plots_dir,
        rewards_dir,
        models_dir,
        checkpoints_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    eval_results_path = os.path.join(eval_dir, "eval_results.txt")
    reward_log_path = os.path.join(rewards_dir, "reward_logs.txt")
    model_save_path = os.path.join(models_dir, "ppo_satellite_final.zip")

    return {
        "run_number": run_number,
        "tensorboard_dir": tensorboard_dir,
        "eval_dir": eval_dir,
        "eval_plots_dir": eval_plots_dir,
        "eval_results_path": eval_results_path,
        "reward_log_path": reward_log_path,
        "model_save_path": model_save_path,
        "checkpoints_dir": checkpoints_dir,
    }


if __name__ == "__main__":
    base_log_dir = "./logs"
    run_dirs = setup_run_directories(base_log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(run_dirs["reward_log_path"], mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    writer = SummaryWriter(log_dir=run_dirs["tensorboard_dir"])

    # Optuna study
    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(
        storage="sqlite:///rl.db", direction="maximize", sampler=sampler
    )
    study.optimize(objective, n_trials=1)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best reward: {study.best_trial.value}")

    # create and wrap the environment with Monitor
    env = SatelliteEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)  # Add this line
    env = Monitor(env)

    model = PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=run_dirs["tensorboard_dir"],
        learning_rate=study.best_trial.params["learning_rate"],
        gamma=study.best_trial.params["gamma"],
        gae_lambda=study.best_trial.params["gae_lambda"],
        clip_range=study.best_trial.params["clip_range"],
        ent_coef=study.best_trial.params["ent_coef"],
        vf_coef=study.best_trial.params["vf_coef"],
        max_grad_norm=study.best_trial.params["max_grad_norm"],
        batch_size=16,
        n_steps=100,
        stats_window_size=1,
        n_epochs=1,
    )

    model.learn(total_timesteps=1000)

    eval_env = SatelliteEnv()
    eval_env = gym.wrappers.TimeLimit(
        eval_env, max_episode_steps=1000
    )  # Gymnasium's timeLimit
    eval_env = Monitor(eval_env)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
    print(f"Evaluation complete. Mean reward: {mean_reward}, Std reward: {std_reward}")
    # # Save final model
    # model.save(run_dirs["model_save_path"])
    # print(f"Final model saved to {run_dirs['model_save_path']}")

    writer.close()
