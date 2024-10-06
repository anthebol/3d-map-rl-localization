import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


class SatelliteEnv(gym.Env):
    def __init__(
        self,
        env_image,
        target_images,
        action_step=0.05,
        image_size=(224, 224),
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.image_size = image_size
        self.action_step = action_step
        self.episode_reward = 0.0
        self.episode_length = 0
        self.max_episode_length = 2500
        self.success_reward = 100.0
        self.success_threshold = 0.95
        self.bonus_threshold = 0.90
        self.max_bonus_reward = 10.0
        self.target_found = False

        # prepare environment and target images
        self.environment_image = self.prepare_image(env_image)
        self.target_images = {
            name: self.prepare_image(img) for name, img in target_images.items()
        }
        self.target_names = list(self.target_images.keys())
        self.current_target_name = None
        self.current_target = None

        # use numpy for internal state
        self.agent_point = np.random.rand(2)

        # Use Gymnasium spaces (which use numpy) for action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "target_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(224, 224, 3),
                    dtype=np.uint8,
                ),
                "current_state_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(224, 224, 3),
                    dtype=np.uint8,
                ),
            }
        )
        print(f"SatelliteEnv observation space: {self.observation_space}")

        self.transform = Compose(
            [
                Resize(self.image_size),
                CenterCrop(self.image_size),
                ToTensor(),
            ]
        )

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
        reward = self._calculate_reward(cosine_similarity)

        # Check for success, but don't terminate the episode
        if cosine_similarity > self.success_threshold and not self.target_found:
            reward += self.success_reward
            self.target_found = True
            print(
                f"Target found! Cosine similarity: {cosine_similarity:.4f}, Steps taken: {self.episode_length}"
            )

        truncated = self.episode_length >= self.max_episode_length

        # if terminated:
        #     reward += self.success_reward

        # terminated = cosine_similarity > self.success_threshold

        # Prepare info dictionary
        info = {
            "cosine_similarity": cosine_similarity,
            "target_found": self.target_found,
            # "success": terminated,
            "agent_coordinates": self.agent_point.copy(),
            "episode_length": self.episode_length,
            "current_target": self.current_target_name,
        }

        # if terminated or truncated:
        #     print(
        #         f"Episode ended! Reason: {'Success' if terminated else 'Max steps reached'}"
        #     )
        #     print(
        #         f"Final similarity: {cosine_similarity:.4f}, Steps taken: {self.episode_length}"
        #     )

        if truncated:
            print(f"Episode ended due to max steps reached")
            print(
                f"Final similarity: {cosine_similarity:.4f}, Steps taken: {self.episode_length}"
            )

        obs = {
            "target_image": self.get_observation(self.current_target),
            "current_state_image": self.get_observation(current_state_image),
        }

        return obs, reward, False, truncated, info

    def _calculate_reward(self, cosine_similarity):
        if cosine_similarity < self.bonus_threshold:
            return cosine_similarity
        else:
            bonus_factor = (cosine_similarity - self.bonus_threshold) / (
                1 - self.bonus_threshold
            )
            bonus = self.max_bonus_reward * bonus_factor
            return cosine_similarity + bonus

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_point = np.random.rand(2) * 0.9
        self.episode_length = 0
        self.episode_reward = 0.0
        self.target_found = False

        self.current_target_name = np.random.choice(self.target_names)
        self.current_target = self.target_images[self.current_target_name]

        obs = {
            "target_image": self.get_observation(self.current_target),
            "current_state_image": self.get_observation(
                self.get_image(self.agent_point)
            ),
        }

        print(
            f"Observation shape in SatelliteEnv reset: {obs['current_state_image'].shape}"
        )
        print(f"Reset: Initial agent position: {self.agent_point}")
        print(f"Current target: {self.current_target_name}")

        return obs, {}

    def render(self):
        pass

    def close(self):
        pass

    def calculate_similarity(self, current_image):
        current_image_flat = current_image.flatten()
        target_image_flat = self.current_target.flatten()
        return F.cosine_similarity(current_image_flat, target_image_flat, dim=0).item()

    def prepare_image(self, image):
        if isinstance(image, np.ndarray):
            # Convert numpy array (H, W, C) to tensor (C, H, W) and normalize to [0, 1]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor):
            # If it's already a tensor, ensure it's in the right format
            if image.dim() == 3:
                image_tensor = image.float() / 255.0 if image.max() > 1 else image
            else:
                raise ValueError("Tensor must have 3 dimensions (C, H, W)")
        else:
            raise ValueError("Unsupported image type")

        # Resize if necessary
        if image_tensor.shape[1:] != self.image_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return image_tensor.to(self.device)

    def get_image(self, point):
        point_tensor = torch.tensor(point, device=self.device)
        offset = torch.tensor(
            [self.environment_image.shape[2], self.environment_image.shape[1]],
            device=self.device,
        )
        scaled_point = point_tensor * offset
        scaled_point = scaled_point.to(torch.int64)

        crop_x1 = max(0, scaled_point[0].item())
        crop_y1 = max(0, scaled_point[1].item())
        crop_x2 = min(self.environment_image.shape[2], crop_x1 + self.image_size[0])
        crop_y2 = min(self.environment_image.shape[1], crop_y1 + self.image_size[1])

        # Ensure the cropped area is valid and non-empty
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(
                f"Invalid crop area: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})"
            )

        cropped_image = self.environment_image[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # Ensure the cropped image is the correct size
        if cropped_image.shape[1:] != self.image_size:
            cropped_image = torch.nn.functional.interpolate(
                cropped_image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return cropped_image

    def get_observation(self, image_tensor):
        # Convert from [0, 1] float tensor to [0, 255] uint8 numpy array
        return (image_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)
