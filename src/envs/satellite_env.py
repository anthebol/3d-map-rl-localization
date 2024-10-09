import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from src.utils.helpers import prepare_image_as_tensor


class SatelliteEnv(gym.Env):
    """
    This environment simulates an agent navigating a large satellite image to find a specific target location.
    The agent receives observations in the form of image patches and must learn to navigate to the target location.

    Args:
        env_image (np.ndarray): The full satellite image representing the environment.
        target_images (dict): A dictionary of target images to locate within the environment.
        action_step (float): The step size for each action.
        image_size (tuple): The size of the image patches (height, width). Defaults to (224, 224).
        device (str): The device to use for tensor operations ('cpu' or 'cuda').
    """

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
        self.success_threshold = 0.90
        self.bonus_threshold = 0.85
        self.max_bonus_reward = 10.0

        self.environment_image = prepare_image_as_tensor(
            env_image,
            image_size,
            device,
        )
        self.target_images = {
            name: prepare_image_as_tensor(img, image_size, device)
            for name, img in target_images.items()
        }
        self.target_names = list(self.target_images.keys())
        self.current_target_name = None
        self.current_target = None

        self.agent_point = np.random.rand(2)

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
        """
        Take a step in the environment based on the given action.

        Args:
            action (np.ndarray): The action to take, represented as a 2D movement vector.

        Returns:
            tuple: A tuple containing:
                - obs (dict): The new observation after taking the action.
                - reward (float): The reward received for taking the action.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
                - info (dict): Additional information about the step.

        """
        self.episode_length += 1

        # convert action to numpy and ensure it's a 1D array with 2 elements
        action_np = np.array(action).flatten()[:2]

        self.agent_point += action_np * self.action_step
        self.agent_point = np.clip(self.agent_point, 0, 0.9)

        current_state_image = self.get_current_image(self.agent_point)

        cosine_similarity = self.calculate_similarity(current_state_image)
        reward = self.calculate_reward(cosine_similarity)

        truncated = self.episode_length >= self.max_episode_length
        terminated = cosine_similarity > self.success_threshold

        if terminated:
            reward += self.success_reward

        info = {
            "cosine_similarity": cosine_similarity,
            "target_found": self.target_found,
            "success": terminated,
            "agent_coordinates": self.agent_point,
            "episode_length": self.episode_length,
            "current_target": self.current_target_name,
        }

        if terminated or truncated:
            print(
                f"Episode ended! Reason: {'Success' if terminated else 'Max steps reached'}"
            )
            print(
                f"Final similarity: {cosine_similarity:.4f}, Steps taken: {self.episode_length}"
            )

        obs = {
            "target_image": self.get_observation(self.current_target),
            "current_state_image": self.get_observation(current_state_image),
        }

        return obs, reward, False, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            tuple: A tuple containing:
                - obs (dict): The initial observation of the reset environment.
                - info (dict): Additional information about the reset.

        """

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
                self.get_current_image(self.agent_point)
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

    def calculate_reward(self, cosine_similarity):
        """
        Calculate the reward based on the cosine similarity between the current state and target image.

        The reward system is designed to guide the agent towards finding the target location in the satellite image.
        It uses cosine similarity as the primary metric for several reasons:
        1. Cosine similarity measures the angle between two vectors, making it invariant to the magnitude of the vectors.
        2. It's effective for comparing image features, as it captures the overall similarity in structure and content.
        3. It ranges from -1 to 1, where 1 indicates perfect similarity, providing a natural scale for rewards.

        The reward mechanism works as follows:
        1. Basic Reward: The cosine similarity itself serves as the basic reward, encouraging the agent to move towards
        more similar image patches.
        2. Bonus Mechanism: When the cosine similarity exceeds the bonus threshold, an additional bonus is added to the reward.
        This creates a stronger incentive for the agent to fine-tune its position once it's in the general vicinity of the target.
        3. Success Reward: When the agent finds the target (cosine similarity > success_threshold), it receives a large
        additional reward (self.success_reward), significantly reinforcing the correct behavior.

        The bonus is calculated as:
        bonus = max_bonus_reward * (cosine_similarity - bonus_threshold) / (1 - bonus_threshold)

        This formula ensures that:
        - The bonus starts at 0 when cosine_similarity equals bonus_threshold
        - The bonus increases linearly, reaching max_bonus_reward when cosine_similarity is 1

        Args:
            cosine_similarity (float): The cosine similarity between the current state and target image.

        Returns:
            float: The calculated reward. This will be:
                - cosine_similarity if below bonus_threshold
                - cosine_similarity + bonus if above bonus_threshold but below success_threshold
                - cosine_similarity + bonus + success_reward if above success_threshold

        Note:
            The success_reward is not added in this function but in the step() method when termination conditions are checked.
        """
        if cosine_similarity < self.bonus_threshold:
            return cosine_similarity
        else:
            bonus_factor = (cosine_similarity - self.bonus_threshold) / (
                1 - self.bonus_threshold
            )
            bonus = self.max_bonus_reward * bonus_factor
            return cosine_similarity + bonus

    def calculate_similarity(self, current_image):
        current_image_flat = current_image.flatten()
        target_image_flat = self.current_target.flatten()
        return F.cosine_similarity(current_image_flat, target_image_flat, dim=0).item()

    def get_current_image(self, point):
        """
        Extract a portion of the environment image at the agent's position, matching the target image dimensions.

        This function ensures that the agent's current view is of the same size as the target image,
        enabling pixel-by-pixel comparison during navigation. It scales the agent's position,
        crops the relevant section of the environment image, and resizes if necessary.

        Args:
            point (numpy.ndarray): The agent's current position in normalized coordinates (0 to 1).

        Returns:
            torch.Tensor: An image tensor of the agent's current view, dimensionally consistent with the target image.
        """
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

        # ensure the cropped area is valid and non-empty
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(
                f"Invalid crop area: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})"
            )

        cropped_image = self.environment_image[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # ensure the cropped image is the correct size
        if cropped_image.shape[1:] != self.image_size:
            cropped_image = torch.nn.functional.interpolate(
                cropped_image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return cropped_image

    def get_observation(self, image_tensor):
        return (image_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)
