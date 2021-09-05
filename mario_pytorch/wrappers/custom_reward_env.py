from typing import Tuple, Dict, Final
from logging import getLogger

import gym
import numpy as np

from mario_pytorch.util.config import RewardConfig

STATUS_TO_INT: Final[Dict[str, int]] = {
    "small": 0,
    "tall": 1,
    "fireball": 2,
}
logger = getLogger(__name__)


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

        self.pprev_x = 0
        self.pprev_coin = 0
        self.pprev_life = 2
        self.pprev_time = 0
        self.pprev_score = 0
        self.pprev_kills = 0
        self.pprev_status = STATUS_TO_INT["small"]

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        _, _, _, info = self.env.step(0)

        self.__prev_state = self.env.reset(**kwargs)
        self.pprev_x = info["x_pos"]
        self.pprev_coin = 0
        self.pprev_life = 2
        self.pprev_time = info["time"]
        self.pprev_score = 0
        self.pprev_kills = 0
        self.pprev_status = STATUS_TO_INT["small"]
        return self.__prev_state

    def change_reward_config(self, reward_config: RewardConfig) -> None:
        self.__reward_config = reward_config
        logger.info(f"[CHANGED] {reward_config}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)

        self.reset_on_each_life(info)

        reward_x = self.process_reward_x(info)
        reward_coin = self.process_reward_coin(info)
        reward_life = self.process_reward_life(info)
        reward_goal = self.process_reward_goal(info)
        reward_item = self.process_reward_item(info)
        reward_time = self.process_reward_time(info)
        reward_score = self.process_reward_score(info)
        reward_kills = self.process_reward_kills(info)
        custom_reward = (
            reward_x
            + reward_coin
            + reward_life
            + reward_goal
            + reward_item
            + reward_time
            + reward_score
            + reward_kills
        )

        return state, custom_reward, done, info

    def reset_on_each_life(self, info: Dict) -> None:
        """ライフが減少したときの reset 処理.

        Notes
        -----
        バグっている可能性は高い
        """
        l = info["life"].item()
        if self.pprev_life - l > 0:
            self.pprev_x = info["x_pos"].item()
            self.pprev_status = STATUS_TO_INT["small"]
            self.pprev_time = info["time"]

    def process_reward_x(self, info: Dict) -> int:
        x = info["x_pos"].item()
        w = self.__reward_config.POSITION
        ret = (x - self.pprev_x) * w
        self.pprev_x = x
        return ret

    def process_reward_coin(self, info: Dict) -> int:
        c = info["coins"]
        w = self.__reward_config.COIN
        if self.pprev_coin <= c:
            ret = (c - self.pprev_coin) * w
        else:
            ret = ((100 + c) - self.pprev_coin) * w
        self.pprev_coin = c
        return ret

    def process_reward_life(self, info: Dict) -> int:
        l = info["life"].item()
        if l == 255:
            l = -1
        w = self.__reward_config.LIFE
        ret = (self.pprev_life - l) * w
        self.pprev_life = l
        return ret

    def process_reward_goal(self, info: Dict) -> int:
        f = int(info["flag_get"])
        w = self.__reward_config.GOAL
        ret = f * w
        return ret

    def process_reward_item(self, info: Dict) -> int:
        s = STATUS_TO_INT[info["status"]]
        w = self.__reward_config.ITEM
        d = max(0, (s - self.pprev_status))
        ret = d * w
        self.pprev_status = s
        return ret

    def process_reward_time(self, info: Dict) -> int:
        t = info["time"]
        w = self.__reward_config.TIME
        d = max(0, self.pprev_time - t)
        ret = d * w
        self.pprev_time = t
        return ret

    def process_reward_score(self, info: Dict) -> int:
        s = info["score"]
        w = self.__reward_config.SCORE
        ret = (s - self.pprev_score) * w
        self.pprev_score = s
        return ret

    def process_reward_kills(self, info: Dict) -> int:
        k = info["kills"]
        w = self.__reward_config.ENEMY
        ret = (k - self.pprev_kills) * w
        self.pprev_kills = k
        return ret
