import optuna
from samplers import SimulatedAnnealingSampler
from utils.training import objective, define_objective
from utils.evaluation import log_evaluation
from envs.satellite_env import SatelliteEnv

if __name__ == "__main__":
    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(storage="sqlite:///rl.db", direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=600)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best reward: {study.best_trial.value}")

    model = define_objective(study.best_trial)

    env = SatelliteEnv()
    model.learn(total_timesteps=25000)
    model.save("ppo_satellite")

    log_evaluation(model, env)
