from typing import Tuple, Dict, Final
from logging import getLogger

import gym
import numpy as np

from mario_pytorch.util.config import RewardConfig
from mario_pytorch.wrappers.custom.custom_info_model import InfoModel

STATUS_TO_INT: Final[Dict[str, int]] = {
    "small": 0,
    "tall": 1,
    "fireball": 2,
}
logger = getLogger(__name__)


def initPlayLog() -> Dict:
    return {
        "x_pos": 0,
        "x_abs": 0,
        "x_plus": 0,
        "x_minus": 0,
        "coins": 0,
        "life": 0,
        "life_plus": 0,
        "life_minus": 0,
        "goal": 0,
        "item": 0,
        "item_plus": 0,
        "item_minus": 0,
        "elapsed": 0,
        "score": 0,
        "kills": 0,
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
        self.playlog = initPlayLog()

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        _, _, _, info = self.env.step(0)

        self.__prev_state = self.env.reset(**kwargs)
        self.pprev_x = info["x_pos"]
        self.pprev_coin = 0
        self.pprev_life = info["life"]
        self.pprev_time = info["time"]
        self.pprev_score = 0
        self.pprev_kills = 0
        self.pprev_status = STATUS_TO_INT["small"]

        self.playlog = initPlayLog()
        self.playlog["x_pos"] = info["x_pos"]
        self.playlog["life"] = info["life"]

        return self.__prev_state

    def change_reward_config(self, reward_config: RewardConfig) -> None:
        self.__reward_config = reward_config
        logger.info(f"[CHANGED] {reward_config}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        info_model = InfoModel.create(info)
        print(info_model)

        # マリオが1機失ったらリセット
        self.reset_on_each_life(info_model)

        # ゴールについたらリセット処理
        self.reset_on_arrival_to_goal(info_model)

        # 差分を計算する
        diff_info = self.get_diff_info(info_model)

        # カスタム報酬と内訳を計算する
        custom_reward, custom_reward_info = self.process_reward(diff_info)

        # 差分用変数を更新する
        self.update_pprev(info_model)

        # プレイログを累積する
        self.accumulate_playlog(info, diff_info)

        return (
            state,
            custom_reward,
            done,
            {
                "default": info,
                "diff_info": diff_info,
                "custom_reward_info": custom_reward_info,
                "playlog": self.playlog,
            },
        )

    def reset_on_each_life(self, info_model: InfoModel) -> None:
        """ライフが減少したときの reset 処理.

        Notes
        -----
        バグっている可能性は高い
        """
        l = info_model.life
        if self.pprev_life - l > 0:
            self.pprev_x = info_model.x_pos
            self.pprev_status = STATUS_TO_INT["small"]
            self.pprev_time = info_model.time

    def reset_on_arrival_to_goal(self, info_model: InfoModel) -> None:
        x = info_model.x_pos
        if self.pprev_x - x > 50:
            self.pprev_x = info_model.x_pos
            self.pprev_time = info_model.time

    def accumulate_playlog(self, info: Dict, diff_info: Dict) -> None:
        self.accumulate_x(info, diff_info)
        self.accumulate_coins(diff_info)
        self.accumulate_kills(diff_info)
        self.accumulate_life(diff_info)
        self.accumulate_goal(diff_info)
        self.accumulate_item(diff_info)
        self.accumulate_elapsed(diff_info)
        self.accumulate_score(diff_info)

    # *--------------------------------------------*
    # * accumulate
    # *--------------------------------------------*

    def accumulate_x(self, info: Dict, diff_info: Dict) -> None:
        self.playlog["x_pos"] = info["x_pos"]
        self.playlog["x_abs"] += abs(diff_info["x_pos"])
        if diff_info["x_pos"] > 0:
            self.playlog["x_plus"] += diff_info["x_pos"]
        if diff_info["x_pos"] < 0:
            self.playlog["x_minus"] += diff_info["x_pos"]

    def accumulate_coins(self, diff_info: Dict) -> None:
        self.playlog["coins"] += diff_info["coins"]

    def accumulate_kills(self, diff_info: Dict) -> None:
        self.playlog["kills"] += diff_info["kills"]

    def accumulate_life(self, diff_info: Dict) -> None:
        self.playlog["life"] += diff_info["life"]
        if diff_info["life"] > 0:
            self.playlog["life_plus"] += diff_info["life"]
        if diff_info["life"] < 0:
            self.playlog["life_minus"] += diff_info["life"]

    def accumulate_goal(self, diff_info: Dict) -> None:
        self.playlog["goal"] += diff_info["goal"]

    def accumulate_item(self, diff_info: Dict) -> None:
        self.playlog["item"] += diff_info["item"]
        if diff_info["item"] > 0:
            self.playlog["item_plus"] += diff_info["item"]
        if diff_info["item"] < 0:
            self.playlog["item_minus"] += diff_info["item"]

    def accumulate_elapsed(self, diff_info: Dict) -> None:
        self.playlog["elapsed"] += abs(diff_info["elapsed"])

    def accumulate_score(self, diff_info: Dict) -> None:
        self.playlog["score"] += diff_info["score"]

    # *--------------------------------------------*
    # * update
    # *--------------------------------------------*

    def update_pprev(self, info_model: InfoModel) -> None:
        self.update_pprev_x(info_model)
        self.update_pprev_coin(info_model)
        self.update_pprev_life(info_model)
        self.update_pprev_status(info_model)
        self.update_pprev_time(info_model)
        self.update_pprev_score(info_model)
        self.update_pprev_kills(info_model)

    def update_pprev_x(self, info_model: InfoModel) -> None:
        self.pprev_x = info_model.x_pos

    def update_pprev_coin(self, info_model: InfoModel) -> None:
        self.pprev_coin = info_model.coins

    def update_pprev_life(self, info_model: InfoModel) -> None:
        l = info_model.life
        if l == 255:
            l = -1
        self.pprev_life = l

    def update_pprev_status(self, info_model: InfoModel) -> None:
        self.pprev_status = STATUS_TO_INT[info_model.status]

    def update_pprev_time(self, info_model: InfoModel) -> None:
        self.pprev_time = info_model.time

    def update_pprev_score(self, info_model: InfoModel) -> None:
        self.pprev_score = info_model.score

    def update_pprev_kills(self, info_model: InfoModel) -> None:
        self.pprev_kills = info_model.kills

    # *--------------------------------------------*
    # * diff
    # *--------------------------------------------*

    def get_diff_info(self, info_model: InfoModel) -> Dict:
        """差分を計算する.

        Notes
        -----
        now - prev を返す.
        """
        return {
            "x_pos": self.get_diff_x(info_model),
            "coins": self.get_diff_coins(info_model),
            "life": self.get_diff_life(info_model),
            "goal": self.get_diff_goal(info_model),
            "item": self.get_diff_item(info_model),
            "elapsed": self.get_diff_time(info_model),
            "score": self.get_diff_score(info_model),
            "kills": self.get_diff_kills(info_model),
        }

    def get_diff_x(self, info_model: InfoModel) -> int:
        return info_model.x_pos - self.pprev_x

    def get_diff_coins(self, info_model: InfoModel) -> int:
        c = info_model.coins
        if self.pprev_coin <= c:
            ret = c - self.pprev_coin
        else:
            ret = (100 + c) - self.pprev_coin
        return ret

    def get_diff_life(self, info_model: InfoModel) -> int:
        l = info_model.life
        if l == 255:
            l = -1
        return l - self.pprev_life

    def get_diff_goal(self, info_model: InfoModel) -> int:
        return int(info_model.flag_get)

    def get_diff_item(self, info_model: InfoModel) -> int:
        return STATUS_TO_INT[info_model.status] - self.pprev_status

    def get_diff_time(self, info_model: InfoModel) -> int:
        return abs(info_model.time - self.pprev_time)

    def get_diff_score(self, info_model: InfoModel) -> int:
        return info_model.score - self.pprev_score

    def get_diff_kills(self, info_model: InfoModel) -> int:
        return info_model.kills - self.pprev_kills

    # *--------------------------------------------*
    # * process
    # *--------------------------------------------*

    def process_reward(self, diff_info: Dict) -> Tuple[int, Dict]:
        x_pos = self.process_reward_x(diff_info)
        coins = self.process_reward_coin(diff_info)
        life = self.process_reward_life(diff_info)
        goal = self.process_reward_goal(diff_info)
        item = self.process_reward_item(diff_info)
        elapsed = self.process_reward_elapsed(diff_info)
        score = self.process_reward_score(diff_info)
        kills = self.process_reward_kills(diff_info)
        reward = x_pos + coins + life + goal + item + elapsed + score + kills
        reward_dict = {
            "x_pos": x_pos,
            "coins": coins,
            "life": life,
            "goal": goal,
            "item": item,
            "elapsed": elapsed,
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

    def process_reward_elapsed(self, diff_info: Dict) -> int:
        return diff_info["elapsed"] * self.__reward_config.TIME

    def process_reward_score(self, diff_info: Dict) -> int:
        return diff_info["score"] * self.__reward_config.SCORE

    def process_reward_kills(self, diff_info: Dict) -> int:
        return diff_info["kills"] * self.__reward_config.ENEMY
