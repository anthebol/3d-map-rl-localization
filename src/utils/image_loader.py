import os

import numpy as np
from PIL import Image


def load_image(path):
    with Image.open(path) as img:
        return np.array(img)


def load_target_images(directory):
    images = {}

    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(filename)[0]
            path = os.path.join(directory, filename)
            images[name] = load_image(path)

    return images


TENSORBOARD_LOG_DIR = "./tensorboard_logs/"
CHECKPOINT_DIR = "./checkpoints/"
FINAL_MODEL_PATH = "./final_model.zip"

env_image = load_image(os.path.join("data", "env", "south_kenstington.jpg"))
train_eval_targets = load_target_images(os.path.join("data", "train_eval"))
test_targets = load_target_images(os.path.join("data", "test"))
single_target = {"high_performing_image": train_eval_targets["target_003_statue"]}
