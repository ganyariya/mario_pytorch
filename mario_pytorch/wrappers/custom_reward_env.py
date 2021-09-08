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
        self.playlog = {}

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
        self.playlog = {}
        return self.__prev_state

    def change_reward_config(self, reward_config: RewardConfig) -> None:
        self.__reward_config = reward_config
        logger.info(f"[CHANGED] {reward_config}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        self.reset_on_each_life(info)

        diff_info = self.get_diff_info(info)
        reward_x = self.process_reward_x(diff_info)
        reward_coin = self.process_reward_coin(diff_info)
        reward_life = self.process_reward_life(diff_info)
        reward_goal = self.process_reward_goal(diff_info)
        reward_item = self.process_reward_item(diff_info)
        reward_time = self.process_reward_time(diff_info)
        reward_score = self.process_reward_score(diff_info)
        reward_kills = self.process_reward_kills(diff_info)
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

    def process_reward_x(self, diff_info: Dict) -> int:
        return diff_info["x_pos"] * self.__reward_config.POSITION

    def process_reward_coin(self, diff_info: Dict) -> int:
        return diff_info["coins"] * self.__reward_config.COIN

    def process_reward_life(self, diff_info: Dict) -> int:
        return diff_info["life"] * self.__reward_config.LIFE

    def process_reward_goal(self, diff_info: Dict) -> int:
        return diff_info["goal"] * self.__reward_config.GOAL

    def process_reward_item(self, diff_info: Dict) -> int:
        return diff_info["item"] * self.__reward_config.ITEM

    def process_reward_time(self, diff_info: Dict) -> int:
        return diff_info["time"] * self.__reward_config.TIME

    def process_reward_score(self, diff_info: Dict) -> int:
        return diff_info["score"] * self.__reward_config.SCORE

    def process_reward_kills(self, diff_info: Dict) -> int:
        return diff_info["kills"] * self.__reward_config.ENEMY

    def get_diff_info(self, info: Dict) -> Dict:
        return {
            "x_pos": self.get_diff_x(info),
            "coins": self.get_diff_coins(info),
            "life": self.get_diff_life(info),
            "goal": self.get_diff_goal(info),
            "item": self.get_diff_item(info),
            "time": self.get_diff_time(info),
            "score": self.get_diff_score(info),
            "kills": self.get_diff_kills(info),
        }

    def get_diff_x(self, info: Dict) -> int:
        return info["x_pos"].item() - self.pprev_x

    def get_diff_coins(self, info: Dict) -> int:
        c = info["coins"]
        if self.pprev_coin <= c:
            ret = c - self.pprev_coin
        else:
            ret = (100 + c) - self.pprev_coin
        return ret

    def get_diff_life(self, info: Dict) -> int:
        l = info["life"].item()
        if l == 255:
            l = -1
        return l - self.pprev_life

    def get_diff_goal(self, info: Dict) -> int:
        return int(info["flag_get"])

    def get_diff_item(self, info: Dict) -> int:
        return STATUS_TO_INT[info["status"]] - self.pprev_status

    def get_diff_time(self, info: Dict) -> int:
        return info["time"] - self.pprev_time

    def get_diff_score(self, info: Dict) -> int:
        return info["score"] - self.pprev_score

    def get_diff_kills(self, info: Dict) -> int:
        return info["kills"] - self.pprev_kills

    def get_playlog_info(self, info: Dict) -> None:
        """プレイログ用の log を返す"""
        pass
