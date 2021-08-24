import gym
from typing import Tuple, Any

StepRet = Tuple[Any, float, bool, dict]


class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> StepRet:
        total_reward = 0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
