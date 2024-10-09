import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != "\n":
            self.level(message)

    def flush(self):
        pass


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
    rewards_dir = os.path.join(run_dir, "rewards")
    models_dir = os.path.join(run_dir, "models")

    for directory in [tensorboard_dir, rewards_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)

    reward_log_path = os.path.join(rewards_dir, "reward_logs.txt")
    model_save_path = os.path.join(models_dir, "ppo_satellite_final.zip")

    return {
        "run_number": run_number,
        "tensorboard_dir": tensorboard_dir,
        "reward_log_path": reward_log_path,
        "model_save_path": model_save_path,
    }


def setup_logging(log_file):
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,
        backupCount=20,
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    sys.stdout = LoggerWriter(root_logger.info)
    sys.stderr = LoggerWriter(root_logger.error)
