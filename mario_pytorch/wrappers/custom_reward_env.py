from typing import Tuple

import gym
import numpy as np


# https://zakopilo.hatenablog.jp/entry/2021/01/30/214806
class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super(CustomRewardEnv, self).__init__(env)
        self.reward = 0

    def reset(self, **kwargs) -> np.ndarray:
        self.reward = 0
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        self.reward = reward
        # ここにカスタム Reward を計算する
        return state, self.reward, done, info
