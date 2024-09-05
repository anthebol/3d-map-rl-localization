import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import os

class SatelliteEnv(gym.Env):
    def __init__(self, action_step=0.05, image_size=(224, 224), device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.image_size = image_size
        self.action_step = action_step
        self.episode_reward = 0.0
        self.episode_length = 0
        self.max_episode_length = 1000
        
        # Use numpy for internal state
        self.agent_point = np.random.rand(2)

        # Use Gymnasium spaces (which use numpy) for action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "target_image": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "current_state_image": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        })
        print(f"SatelliteEnv observation space: {self.observation_space}")

        self.transform = Compose([
            Resize(self.image_size),
            CenterCrop(self.image_size),
            ToTensor(),
        ])

        environment_path = os.path.join('../data', 'exibition_road_1.jpg')
        self.environment_image = self.load_image(environment_path)

        target_image_path = os.path.join('../data', 'queens_tower.jpg')
        self.target_image = self.load_image(target_image_path)

        self.time_penalty = -0.01
        self.success_reward = 100
        self.success_threshold = 0.90

    def step(self, action):
        self.episode_length += 1
        
        # Convert action to numpy and ensure it's a 1D array with 2 elements
        action_np = np.array(action).flatten()[:2]
        
        # Update agent position
        self.agent_point += action_np * self.action_step
        self.agent_point = np.clip(self.agent_point, 0, 0.9)
        
        # Get new state image
        current_state_image = self.get_image(self.agent_point)
        
        # Calculate similarity
        cosine_similarity = self.calculate_similarity(current_state_image)

        # Calculate reward
        reward = cosine_similarity
        terminated = cosine_similarity > self.success_threshold
        truncated = self.episode_length >= self.max_episode_length

        if terminated:
            reward += self.success_reward
        
        # Prepare info dictionary
        info = {
            'cosine_similarity': cosine_similarity,
            'success': terminated,
            'agent_coordinates': self.agent_point.copy(),
            'episode_length': self.episode_length
        }
        
        if terminated or truncated:
            print(f"Episode ended! Reason: {'Success' if terminated else 'Max steps reached'}")
            print(f"Final similarity: {cosine_similarity:.4f}, Steps taken: {self.episode_length}")
        
        obs = {
            "target_image": self.target_image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8),
            "current_state_image": current_state_image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8),
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_point = np.random.rand(2) * 0.9
        
        obs = {
            "target_image": self.target_image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8),
            "current_state_image": self.get_image(self.agent_point).cpu().numpy().transpose(1, 2, 0).astype(np.uint8),
        }


        print(f"Observation shape in SatelliteEnv reset: {obs['current_state_image'].shape}")
        print(f"Reset: Initial agent position: {self.agent_point}")

        return obs, {}

    def render(self):
        pass

    def close(self):
        pass

    def calculate_similarity(self, current_image):
        current_image_flat = current_image.flatten()
        target_image_flat = self.target_image.flatten()
        return torch.nn.functional.cosine_similarity(current_image_flat, target_image_flat, dim=0).item()

    def load_image(self, path):
        with Image.open(path).convert("RGB") as img:
            return self.transform(img).to(self.device)
        
    def get_image(self, point):
        point_tensor = torch.tensor(point, device=self.device)
        offset = torch.tensor([self.environment_image.shape[2], self.environment_image.shape[1]], device=self.device)
        scaled_point = point_tensor * offset
        scaled_point = scaled_point.to(torch.int64)

        crop_x1 = max(0, scaled_point[0].item())
        crop_y1 = max(0, scaled_point[1].item())
        crop_x2 = min(self.environment_image.shape[2], crop_x1 + self.image_size[0])
        crop_y2 = min(self.environment_image.shape[1], crop_y1 + self.image_size[1])

        # Ensure the cropped area is valid and non-empty
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(f"Invalid crop area: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")

        cropped_image = self.environment_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Ensure the cropped image is the correct size
        if cropped_image.shape[1:] != self.image_size:
            cropped_image = torch.nn.functional.interpolate(
                cropped_image.unsqueeze(0), 
                size=self.image_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

        return cropped_image