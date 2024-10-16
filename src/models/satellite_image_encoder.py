import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatelliteFeatureExtractor(BaseFeaturesExtractor):
    """
    Uses a pre-trained MobileNetV2 model to extract features from satellite images。
    The extractor processes both the target image and the current state image, concatenates
    their features, and passes them through a small neural network for further processing.
    """

    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 1280):
        super(SatelliteFeatureExtractor, self).__init__(observation_space, features_dim)

        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier = nn.Identity()
        self.mobilenet = self.mobilenet.to(device)

        # update the input size of the first linear layer
        self.network = nn.Sequential(
            nn.Linear(features_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

    def forward(self, obs):
        # assume input is already in [batch_size, channels, height, width] format
        target_image = obs["target_image"].float()
        current_state_image = obs["current_state_image"].float()

        self.mobilenet.eval()

        target_features = self.mobilenet(target_image)
        current_state_features = self.mobilenet(current_state_image)

        target_features = torch.flatten(target_features, start_dim=1)
        current_state_features = torch.flatten(current_state_features, start_dim=1)

        features = torch.cat([target_features, current_state_features], dim=-1)

        return self.network(features)
