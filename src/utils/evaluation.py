import numpy as np
import matplotlib.pyplot as plt

from evaluate import evaluate_policy
from tensorboardX import SummaryWriter


def log_evaluation(model, env, log_dir="./tensorboard_logs/eval", n_eval_episodes=10, total_timestep=1000):
    eval_writer = SummaryWriter(log_dir=log_dir)

    print("Starting evaluation...")
    episode_rewards, episode_lengths, final_distances, success_rate = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, total_timestep=total_timestep
    )
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_final_distance = np.mean(final_distances)

    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Mean Final Distance to Target: {mean_final_distance:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

    eval_writer.add_scalar('Eval/Mean_Reward', mean_reward)
    eval_writer.add_scalar('Eval/Mean_Episode_Length', mean_length)
    eval_writer.add_scalar('Eval/Mean_Final_Distance', mean_final_distance)
    eval_writer.add_scalar('Eval/Success_Rate', success_rate)

    plot_evaluation_results(episode_rewards, episode_lengths, final_distances)

    eval_writer.close()


def plot_evaluation_results(rewards, lengths, distances):
    (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    ax2.plot(lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')

    ax3.plot(distances)
    ax3.set_title('Final Distance to Target')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Distance')

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()
