from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self, eval_env, eval_freq=10000, n_eval_episodes=5, tensorboard_log=None
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.tensorboard_log = tensorboard_log

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            if self.tensorboard_log:
                self.tensorboard_log.add_scalar(
                    "eval/mean_reward", mean_reward, self.num_timesteps
                )
                self.tensorboard_log.add_scalar(
                    "eval/std_reward", std_reward, self.num_timesteps
                )

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)

        return True
