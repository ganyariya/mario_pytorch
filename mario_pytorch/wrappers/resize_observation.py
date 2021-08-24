import gym
import numpy as np

from torchvision import transforms as T
from gym.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # (shape, shape, 3)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation
