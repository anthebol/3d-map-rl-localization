import optuna
import logging

from utils.evaluation_logger import log_evaluation
from utils.training import objective, define_objective, LoggerRewardCallback
from envs.satellite_env import SatelliteEnv
from samplers import SimulatedAnnealingSampler

log_file = 'rewards_log.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # overwrite the log file each run
        logging.StreamHandler()  # optionally log to console as well
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(storage="sqlite:///rl.db", direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=2)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best reward: {study.best_trial.value}")

    model = define_objective(study.best_trial)

    env = SatelliteEnv()
    logger_reward_callback = LoggerRewardCallback(logger)
    model.learn(total_timesteps=200*100,callback=logger_reward_callback)
    model.save("ppo_satellite")

    log_evaluation(model, env, n_eval_episodes=50, total_timestep=1000)