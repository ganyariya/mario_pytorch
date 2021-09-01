import gym
import numpy as np
import torch
from gym.spaces import Box
from torchvision import transforms as T


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # (240, 256) (H, W)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation: np.ndarray) -> torch.Tensor:
        # permute [H, W, C] array to [C, H, W] tensor
        observation: np.ndarray = np.transpose(observation, (2, 0, 1))
        observation: torch.Tensor = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation: np.ndarray) -> torch.Tensor:
        observation: torch.Tensor = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
