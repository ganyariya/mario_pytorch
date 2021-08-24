import gym
import numpy as np
import torch

from torchvision import transforms as T
from gym.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # (aimed_shape, aimed_shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: torch.Tensor) -> torch.Tensor:
        # argument observation (1, 240, 256)
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        # transformed observation (84, 84)
        observation = transforms(observation).squeeze(0)
        return observation
