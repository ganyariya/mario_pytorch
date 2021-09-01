from typing import Any, Tuple

import gym

StepRet = Tuple[Any, float, bool, dict]


class SkipFrame(gym.Wrapper):
    """skipフレーム分報酬を貯める

    Notes
    -----
    skipFrame 分同じ action を実行する
    """

    def __init__(self, env: gym.Env, skip: int) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> StepRet:
        total_reward = 0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
