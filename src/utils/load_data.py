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
