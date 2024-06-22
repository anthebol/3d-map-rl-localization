import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MapRLNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 1280):
        super(MapRLNetwork, self).__init__(observation_space, features_dim)

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        self.mobilenet = self.mobilenet.to(device)

        self.network = nn.Sequential(
            nn.Linear(features_dim * 2 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, obs):
        target_image = obs["target_image"].permute(0, 3, 1, 2).float()
        current_state_image = obs["current_state_image"].permute(0, 3, 1, 2).float()
        coordinates = obs["coordinates"].float()

        self.mobilenet.eval()
        target_features = self.mobilenet(target_image)
        current_state_features = self.mobilenet(current_state_image)

        target_features = torch.flatten(target_features, start_dim=1)
        current_state_features = torch.flatten(current_state_features, start_dim=1)
        coordinates = coordinates.view(coordinates.shape[0], -1)

        features = torch.cat(
            [target_features, current_state_features, coordinates], dim=-1
        )

        return self.network(features)
