import numpy as np
from tensorboardX import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os


class CustomEvalCallback:
    def __init__(self):
        self.final_distances = []

    def __call__(self, locals_, globals_):
        if locals_['dones'][0]:
            info = locals_['infos'][0]
            if 'final_distance' in info:
                self.final_distances.append(info['final_distance'])


def log_evaluation(model, env, log_dir, plots_dir, results_path, run_number, n_eval_episodes=1):
    # eval_writer = SummaryWriter(log_dir=log_dir)

    print("Starting evaluation...")
    # callback = CustomEvalCallback()
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )

    # episode_rewards, episode_lengths = mean_reward, std_reward  # evaluate_policy returns these when return_episode_rewards=True
    # mean_reward = np.mean(episode_rewards)
    # std_reward = np.std(episode_rewards)
    # mean_length = np.mean(episode_lengths)
    # mean_final_distance = np.mean(callback.final_distances) if callback.final_distances else None
    # success_rate = sum(r > 0 for r in episode_rewards) / len(episode_rewards)  # Assuming success is when reward > 0

    # with open(results_path, 'w') as f:
    #     f.write(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    #     f.write(f"Mean Episode Length: {mean_length:.2f}\n")
    #     if mean_final_distance is not None:
    #         f.write(f"Mean Final Distance to Target: {mean_final_distance:.2f}\n")
    #     f.write(f"Success Rate: {success_rate:.2%}\n")

    # print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    # print(f"Mean Episode Length: {mean_length:.2f}")
    # if mean_final_distance is not None:
    #     print(f"Mean Final Distance to Target: {mean_final_distance:.2f}")
    # print(f"Success Rate: {success_rate:.2%}")

    # eval_writer.add_scalar('Eval/Mean_Reward', mean_reward)
    # eval_writer.add_scalar('Eval/Mean_Episode_Length', mean_length)
    # if mean_final_distance is not None:
    #     eval_writer.add_scalar('Eval/Mean_Final_Distance', mean_final_distance)
    # eval_writer.add_scalar('Eval/Success_Rate', success_rate)

    # plot_evaluation_results(episode_rewards, episode_lengths, callback.final_distances, plots_dir, run_number)

    return mean_reward, std_reward

def plot_evaluation_results(rewards, lengths, distances, plots_dir, run_number):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # Plot episode lengths
    ax2.plot(lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')

    # Plot final distances to target
    if distances:
        ax3.plot(distances)
        ax3.set_title('Final Distance to Target')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Distance')
    else:
        ax3.set_title('Final Distance to Target (Not Available)')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'evaluation_results_{run_number}.png'))
    plt.close()