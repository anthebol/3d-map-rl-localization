import gym
from gym import spaces
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatelliteEnv(gym.Env):
    def __init__(self, action_step=0.05, image_size=(224, 224)):
        super().__init__()

        self.image_size = image_size
        self.action_step = action_step
        self.episode_reward = 0.0
        self.episode_length = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = spaces.Dict(
            {
                "target_image": spaces.Box(low=0, high=255, shape=(224, 224, 3)),
                "coordinates": spaces.Box(low=-1, high=1, shape=(2,)),
                "current_state_image": spaces.Box(low=0, high=255, shape=(224, 224, 3)),
            }
        )

        environment_path = os.path.join(
            './', 'exibition_road_1.jpg'
        )

        self.environment_image_pil = Image.open(environment_path).convert("RGB")

        self.transform = Compose(
            [
                Resize((int(self.environment_image_pil.height * 0.05), int(self.environment_image_pil.width * 0.05))),
                CenterCrop(self.image_size),
                ToTensor(),
                lambda x: x.to(device),
            ]
        )

        target_image_path = os.path.join(
            './', 'exibition_road_target.jpg'
        )

        target_image_pil = Image.open(target_image_path).convert("RGB").resize((224, 224))
        self.target_image = self.transform(target_image_pil).to(device)

        self.agent_point = torch.rand(2, device=device) * 1.8 - 0.9

    def get_image(self, point):
        offset = torch.tensor([self.environment_image_pil.width, self.environment_image_pil.height],
                              device=device)
        scaled_point = (point + 1) / 2 * offset
        scaled_point = scaled_point.to(torch.int64)
        crop_x1 = max(0, scaled_point[0].item() - self.environment_image_pil.width // 40)
        crop_y1 = max(0, scaled_point[1].item() - self.environment_image_pil.height // 40)
        crop_x2 = min(self.environment_image_pil.width, scaled_point[0].item() + self.environment_image_pil.width // 40)
        crop_y2 = min(self.environment_image_pil.height,
                      scaled_point[1].item() + self.environment_image_pil.height // 40)

        cropped_image = self.environment_image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        image = self.transform(cropped_image)
        image_np = image.cpu().numpy()
        return image_np

    def step(self, action):
        action_scale = 0.01
        self.agent_point[0] += action[2] * action_scale - action[3] * action_scale
        self.agent_point[1] += action[0] * action_scale - action[1] * action_scale

        self.agent_point = torch.clamp(self.agent_point, -0.9, 0.9)

        current_state_image = self.get_image(self.agent_point)

        reward = -torch.norm(
            self.target_image - torch.tensor(current_state_image, dtype=torch.float32).to(device)
        ).item()

        done = np.linalg.norm(self.agent_point.cpu().numpy()) < 0.7

        if done:
            print("Episode terminated!")

        obs = {
            "target_image": self.target_image.cpu().numpy().transpose(1, 2, 0),
            "coordinates": self.agent_point.cpu().numpy(),
            "current_state_image": current_state_image.transpose(1, 2, 0),
        }

        return obs, reward, done, {}

    def reset(self):
        self.agent_point = (torch.rand(2).to(device) * 1.8) - 0.9
        obs = {
            "target_image": self.target_image.cpu().numpy().transpose(1, 2, 0),
            "coordinates": self.agent_point.cpu().numpy(),
            "current_state_image": self.get_image(self.agent_point).transpose(1, 2, 0),
        }

        return obs
