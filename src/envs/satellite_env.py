import os
import gym
import torch

from gym import spaces
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor


class SatelliteEnv(gym.Env):
    def __init__(self, action_step=0.05, image_size=(224, 224), device='cpu'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.action_step = action_step
        self.episode_reward = 0.0
        self.episode_length = 0
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, 1))
        self.observation_space = spaces.Dict(
            {
                "target_image": spaces.Box(low=0, high=255, shape=(224, 224, 3)),
                "current_state_image": spaces.Box(low=0, high=255, shape=(224, 224, 3)),
            }
        )

        environment_path = os.path.join('../data', 'exibition_road_1.jpg')
        self.environment_image_pil = Image.open(environment_path).convert("RGB")

        self.transform = Compose(
            [
                Resize(self.image_size),
                CenterCrop(self.image_size),
                ToTensor(),
                lambda x: x.to(device),
            ]
        )

        target_image_path = os.path.join('../data', 'queens_tower.jpg')
        self.target_image_pil = Image.open(target_image_path).convert("RGB")
        self.target_image = self.transform(self.target_image_pil).to(device)
        
        self.agent_point = torch.rand(2, device=device)

        self.time_penalty = -0.01  # small time penalty for each step
        self.success_reward = 100  # large reward for finding the target
        self.success_threshold = 0.90  # cosine similarity threshold for "finding" the target


    def get_image(self, point):
        offset = torch.tensor([self.environment_image_pil.width, self.environment_image_pil.height], device=self.device)
        scaled_point = point * offset
        scaled_point = scaled_point.to(torch.int64)
        crop_x1 = max(0, scaled_point[0].item())
        crop_y1 = max(0, scaled_point[1].item())
        crop_x2 = min(self.environment_image_pil.width, crop_x1 + 224)
        crop_y2 = min(self.environment_image_pil.height, crop_y1 + 224)

        cropped_image = self.environment_image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        image = self.transform(cropped_image)
        image_np = image.cpu().numpy()

        return image_np


    def step(self, action):
        action_scale = 0.01
        
        self.agent_point[0] += action[0] * action_scale 
        self.agent_point[1] += action[1] * action_scale 

        self.agent_point = torch.clamp(self.agent_point, 0, 0.9)

        current_state_image = self.get_image(self.agent_point)
        current_state_image_flat = torch.tensor(current_state_image, dtype=torch.float32).flatten().to(self.device)
        target_image_flat = self.target_image.flatten()

        assert current_state_image_flat.shape == target_image_flat.shape, f"Shape mismatch: {current_state_image_flat.shape} vs {target_image_flat.shape}"

        cosine_similarity = torch.nn.functional.cosine_similarity(target_image_flat, current_state_image_flat, dim=0).item()

        reward = self.time_penalty

        done = cosine_similarity > self.success_threshold
        if done:
            reward += self.success_reward
        else:
            reward += cosine_similarity 

        info = {
            'cosine_similarity': cosine_similarity,
            'success': done,
        }

        if done:
            info['agent_coordinates'] = self.agent_point.cpu().numpy()
            print("Episode terminated!")

        obs = {
            "target_image": self.target_image.cpu().numpy().transpose(1, 2, 0),
            "current_state_image": current_state_image.transpose(1, 2, 0),
        }

        return obs, reward, done, info


    def reset(self):
        self.agent_point = torch.rand(2).to(self.device) * 0.9 
        self.target_image = self.transform(self.target_image_pil).to(self.device)
        
        obs = {
            "target_image": self.target_image.cpu().numpy().transpose(1, 2, 0),
            "current_state_image": self.get_image(self.agent_point).transpose(1, 2, 0),
        }

        return obs
