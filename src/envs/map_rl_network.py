import gym
import torch
import torch.nn as nn
import torchvision.models as models

from stable_baselines3.common.policies import  ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MapRLNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64):
        super(MapRLNetwork, self).__init__(observation_space, features_dim)
        
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        self.mobilenet = self.mobilenet.to(device)
        
        self.network = nn.Sequential(
            nn.Linear(features_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
        
    def forward(self, obs):
        target_image = obs["target_image"].permute(0, 3, 1, 2).float()
        current_state_image = obs["current_state_image"].permute(0, 3, 1, 2).float()
        
        self.mobilenet.eval()
        
        target_features = self.mobilenet(target_image)

        current_state_features = self.mobilenet(current_state_image)
        
        target_features = torch.flatten(target_features, start_dim=1)
        current_state_features = torch.flatten(current_state_features, start_dim=1)

        features = torch.cat(
            [target_features, current_state_features], dim=-1
        )

        return self.network(features)
    

class MapRLPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MapRLPolicy, self).__init__(
            *args, **kwargs,
            features_extractor_class=MapRLNetwork,
            features_extractor_kwargs={'features_dim': 1280},
        )
