from typing import Tuple, Dict, Final, Literal

import gym
import numpy as np

from mario_pytorch.util.config import RewardConfig

STATUS_LITERAL = Literal("small", "tall", "fireball")
STATUS_TO_INT: Final[Dict[STATUS_LITERAL, int]] = {
    "small": 0,
    "tall": 1,
    "fireball": 2,
}

# https://zakopilo.hatenablog.jp/entry/2021/01/30/214806
class CustomRewardEnv(gym.Wrapper):
    """カスタム報酬関数を実装する.

    TODO
    ----
    - マリオが死んだとき，位置を0に戻す処理がおそらく必要
    - 時間も同様である

    Notes
    -----
    - state.shape: (240, 256, 3)
    - 重複を避けるために `__` とする
    """

    def __init__(self, env: gym.Env, reward_config: RewardConfig) -> None:
        super(CustomRewardEnv, self).__init__(env)
        self.__reward_config = reward_config
        self.__prev_state = env.reset()

        self.__prev_x = 0
        self.__prev_coin = 0
        self.__prev_life = 0
        self.__prev_time = 0
        self.__prev_score = 0
        self.__prev_status = STATUS_TO_INT["small"]

    def reset(self, **kwargs) -> np.ndarray:
        self.__prev_state = self.env.reset(**kwargs)
        return self.__prev_state

    def change_reward_config(self, reward_config: RewardConfig) -> None:
        self.__reward_config = reward_config

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)

        reward_x = self.__process_reward_x(info)
        reward_coin = self.__process_reward_coin(info)
        reward_life = self.__process_reward_life(info)
        reward_goal = self.__process_reward_goal(info)
        reward_item = self.__process_reward_item(info)
        reward_time = self.__process_reward_time(info)
        reward_score = self.__process_reward_score(info)

        return state, reward, done, info

    def __process_reward_x(self, info: Dict) -> int:
        x = info["x_pos"]
        w = self.__reward_config.POSITION
        ret = (x - self.__prev_x) * w
        self.__prev_x = x
        return ret

    def __process_reward_coin(self, info: Dict) -> int:
        c = info["coins"]
        w = self.__reward_config.COIN
        if self.__prev_coin <= c:
            ret = (c - self.__prev_coin) * w
        else:
            ret = ((100 + c) - self.__prev_coin) * w
        self.__prev_coin = c
        return ret

    def __process_reward_life(self, info: Dict) -> int:
        l = info["life"]
        w = self.__reward_config.LIFE
        ret = (l - self.__prev_life) * w
        self.__prev_life = l
        return ret

    def __process_reward_goal(self, info: Dict) -> int:
        f = int(info["flag_get"])
        w = self.__reward_config.GOAL
        ret = f * w
        return ret

    def __process_reward_item(self, info: Dict) -> int:
        s = STATUS_TO_INT[info["status"]]
        w = self.__reward_config.ITEM
        d = max(0, (s - self.__prev_status))
        ret = d * w
        self.__prev_status = s
        return ret

    def __process_reward_time(self, info: Dict) -> int:
        t = info["time"]
        w = self.__reward_config.TIME
        d = max(0, self.__prev_time - t)
        ret = d * w
        self.__prev_time = t
        return ret

    def __process_reward_score(self, info: Dict) -> int:
        s = info["score"]
        w = self.__reward_config.SCORE
        ret = (s - self.__prev_score) * w
        self.__prev_score = s
        return ret
