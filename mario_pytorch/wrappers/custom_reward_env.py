from typing import Tuple

import gym
import numpy as np


# https://zakopilo.hatenablog.jp/entry/2021/01/30/214806
class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super(CustomRewardEnv, self).__init__(env)
        self.reward = 0
        self.prev_state = env.reset()

    def reset(self, **kwargs) -> np.ndarray:
        self.reward = 0
        self.prev_state = self.env.reset(**kwargs)
        return self.prev_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        return state, self.reward, done, info
