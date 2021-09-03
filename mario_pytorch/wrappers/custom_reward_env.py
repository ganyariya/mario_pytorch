from typing import Tuple

import gym
import numpy as np


# https://zakopilo.hatenablog.jp/entry/2021/01/30/214806
class CustomRewardEnv(gym.Wrapper):
    """カスタム報酬関数を実装する.

    Notes
    -----
    - state.shape: (240, 256, 3)
    - 重複を避けるために `__` とする
    """

    def __init__(self, env: gym.Env) -> None:
        super(CustomRewardEnv, self).__init__(env)
        self.__reward = 0
        self.__prev_state = env.reset()

    def reset(self, **kwargs) -> np.ndarray:
        self.__reward = 0
        self.__prev_state = self.env.reset(**kwargs)
        return self.__prev_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        self.__reward = 0
        return state, reward, done, info
