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

        # マリオが1機失ったらリセット
        self.reset_on_each_life(info)

        # 差分を計算する
        diff_info = self.get_diff_info(info)

        # カスタム報酬と内訳を計算する
        custom_reward, custom_reward_info = self.process_reward(diff_info)

        # 差分用変数を更新する
        self.update_pprev(info)

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

    # *--------------------------------------------*
    # * update
    # *--------------------------------------------*

    def update_pprev(self, info: Dict) -> None:
        self.update_pprev_x(info)
        self.update_pprev_coin(info)
        self.update_pprev_life(info)
        self.update_pprev_status(info)
        self.update_pprev_time(info)
        self.update_pprev_score(info)
        self.update_pprev_kills(info)

    def update_pprev_x(self, info: Dict) -> None:
        self.pprev_x = info["x_pos"]

    def update_pprev_coin(self, info: Dict) -> None:
        self.pprev_coin = info["coins"]

    def update_pprev_life(self, info: Dict) -> None:
        l = info["life"].item()
        if l == 255:
            l = -1
        self.pprev_life = l

    def update_pprev_status(self, info: Dict) -> None:
        self.pprev_status = STATUS_TO_INT[info["status"]]

    def update_pprev_time(self, info: Dict) -> None:
        self.pprev_time = info["time"]

    def update_pprev_score(self, info: Dict) -> None:
        self.pprev_score = info["score"]

    def update_pprev_kills(self, info: Dict) -> None:
        self.pprev_kills = info["kills"]

    # *--------------------------------------------*
    # * diff
    # *--------------------------------------------*

    def get_diff_info(self, info: Dict) -> Dict:
        """差分を計算する.

        Notes
        -----
        now - prev を返す.
        """
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

    # *--------------------------------------------*
    # * process
    # *--------------------------------------------*

    def process_reward(self, diff_info: Dict) -> Tuple[int, Dict]:
        x_pos = self.process_reward_x(diff_info)
        coins = self.process_reward_coin(diff_info)
        life = self.process_reward_life(diff_info)
        goal = self.process_reward_goal(diff_info)
        item = self.process_reward_item(diff_info)
        time = self.process_reward_time(diff_info)
        score = self.process_reward_score(diff_info)
        kills = self.process_reward_kills(diff_info)
        reward = x_pos + coins + life + goal + item + time + score + kills
        reward_dict = {
            "x_pos": x_pos,
            "coins": coins,
            "life": life,
            "goal": goal,
            "item": item,
            "time": time,
            "score": score,
            "kills": kills,
        }
        return reward, reward_dict

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
