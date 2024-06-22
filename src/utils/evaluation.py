import numpy as np
import torch
from tensorboardX import SummaryWriter
from evaluate import evaluate_policy
from envs.satellite_env import SatelliteEnv


def log_evaluation(model, env, log_dir="./tensorboard_logs/eval", n_eval_episodes=5, total_timestep=25000):
    eval_writer = SummaryWriter(log_dir=log_dir)

    print("Starting evaluation...")
    episode_rewards, episode_lengths = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True, total_timestep=total_timestep
    )
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"{mean_reward=};{std_reward=}")

    print("Logging to TensorBoard...")
    for idx, (reward_item, episode_length) in enumerate(zip(episode_rewards, episode_lengths)):
        eval_writer.add_scalar('Eval Reward', reward_item / episode_length, global_step=idx)
    eval_writer.flush()
    eval_writer.close()
    print("Logged successfully.")
