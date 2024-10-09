from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback


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
