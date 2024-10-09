from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class TestCallback(BaseCallback):
    def __init__(
        self,
        test_env,
        test_freq,
        n_test_episodes,
        tensorboard_log=None,
        verbose=0,
    ):
        super(TestCallback, self).__init__(verbose)
        self.test_env = test_env
        self.test_freq = test_freq
        self.n_test_episodes = n_test_episodes
        self.tensorboard_log = tensorboard_log

    def _on_step(self):
        if self.n_calls % self.test_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.test_env,
                n_eval_episodes=self.n_test_episodes,
                return_episode_rewards=True,
            )
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)

            if hasattr(self.model, "logger"):
                train_info = self.model.logger.name_to_value
                policy_loss = train_info.get("train/policy_loss", None)
                value_loss = train_info.get("train/value_loss", None)
                entropy_loss = train_info.get("train/entropy_loss", None)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            print(f"{current_time} - INFO - Test Results:")
            print(f"  Mean Reward: {mean_reward:.4f}")
            print(f"  Std Reward: {std_reward:.4f}")
            print(f"  Mean Episode Length: {mean_length:.2f}")
            if policy_loss is not None:
                print(f"  Current Policy Loss: {policy_loss:.4f}")
            if value_loss is not None:
                print(f"  Current Value Loss: {value_loss:.4f}")
            if entropy_loss is not None:
                print(f"  Current Entropy Loss: {entropy_loss:.4f}")

            if self.tensorboard_log:
                self.tensorboard_log.add_scalar(
                    "test/mean_reward", mean_reward, self.num_timesteps
                )
                self.tensorboard_log.add_scalar(
                    "test/std_reward", std_reward, self.num_timesteps
                )
                self.tensorboard_log.add_scalar(
                    "test/mean_episode_length", mean_length, self.num_timesteps
                )
                if policy_loss is not None:
                    self.tensorboard_log.add_scalar(
                        "test/policy_loss", policy_loss, self.num_timesteps
                    )
                if value_loss is not None:
                    self.tensorboard_log.add_scalar(
                        "test/value_loss", value_loss, self.num_timesteps
                    )
                if entropy_loss is not None:
                    self.tensorboard_log.add_scalar(
                        "test/entropy_loss", entropy_loss, self.num_timesteps
                    )

            self.logger.record("test/mean_reward", mean_reward)
            self.logger.record("test/std_reward", std_reward)
            self.logger.record("test/mean_episode_length", mean_length)
            if policy_loss is not None:
                self.logger.record("test/policy_loss", policy_loss)
            if value_loss is not None:
                self.logger.record("test/value_loss", value_loss)
            if entropy_loss is not None:
                self.logger.record("test/entropy_loss", entropy_loss)

        return True
