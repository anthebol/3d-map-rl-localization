import optuna
import torch
from stable_baselines3 import PPO
from envs.satellite_env import SatelliteEnv
from envs.map_rl_network import MapRLNetwork, MapRLPolicy
from evaluate import evaluate_policy


def objective(trial):
    env = SatelliteEnv()

    model = PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="./tensorboard_logs/",
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        gamma=trial.suggest_float("gamma", 0.5, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.5),
        ent_coef=trial.suggest_float("ent_coef", 0., 0.5),
        vf_coef=trial.suggest_float("vf_coef", 0.3, 0.8),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.8),
        batch_size=64,
        n_steps=2048,
        stats_window_size=100,
        n_epochs=4,
    )

    model.learn(total_timesteps=25000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, total_timestep=25000)

    return mean_reward


def define_objective(trial):
    env = SatelliteEnv()

    model = PPO(
        MapRLPolicy,
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="./tensorboard_logs/",
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        gamma=trial.suggest_float("gamma", 0.9, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.5),
        ent_coef=trial.suggest_float("ent_coef", 0., 0.5),
        vf_coef=trial.suggest_float("vf_coef", 0.3, 0.8),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.8),
        batch_size=64,
        n_steps=2048,
        stats_window_size=100,
        n_epochs=4,
    )

    return model
