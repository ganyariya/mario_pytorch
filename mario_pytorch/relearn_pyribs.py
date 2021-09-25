import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import dill
import torch
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer

from mario_pytorch.agent.mario import Mario, ReLearnMario
from mario_pytorch.metric_logger.metric_logger import MetricLogger, _set_logger
from mario_pytorch.util.config import (
    EnvConfig,
    PlayLogScopeConfig,
    RewardConfig,
    RewardScopeConfig,
)
from mario_pytorch.util.get_env import get_env
from mario_pytorch.util.process_path import (
    get_checkpoint_path,
    get_date_path,
    get_env_config_path,
    get_model_path,
    get_playlog_scope_config_path,
    get_pickles_path,
    get_results_path,
    get_reward_models_path,
    get_reward_scope_config_path,
)
from mario_pytorch.wrappers.custom import CustomRewardEnv
from mario_pytorch.wrappers.custom.custom_info_model import PlayLogModel
from mario_pytorch.util.save_model import save_episode_model

# ----------------------------------------------------------------------


def restore_objects(
    pickles_path: Path, reward_models_path: Path
) -> tuple[GridArchive, ImprovementEmitter, Optimizer, MetricLogger, dict[str, Any]]:
    with open(pickles_path / "archive.pickle", "rb") as f:
        archive: GridArchive = pickle.load(f)
    with open(pickles_path / "emitters.pickle", "rb") as f:
        emitters: ImprovementEmitter = pickle.load(f)
    with open(pickles_path / "optimzer.pickle", "rb") as f:
        optimizer: Optimizer = pickle.load(f)
    with open(pickles_path / "logger.dill", "rb") as f:
        logger: MetricLogger = dill.load(f)
    with open(reward_models_path / "playlog_reward.json", "r") as f:
        playlog_reward_dict = json.load(f)
    return archive, emitters, optimizer, logger, playlog_reward_dict


def save_playlog_reward_dict(
    parameter: np.ndarray,
    episode_serial: int,
    behavior: list[int],
    playlogs: list[PlayLogModel],
    rewards: list[float],
    reward_models_path: Path,
    playlog_reward_dict: dict[str, Any],
) -> None:
    """プレイログの値ごとに，報酬やエピソード数などを保存する.

    Notes
    -----
    モデルも保存するかは今後しだい．
    リファクタリングしたほうがいいかも.
    """
    playlog_key = ",".join(map(lambda x: str(x), behavior))
    if playlog_key not in playlog_reward_dict:
        playlog_reward_dict[playlog_key] = []
    playlog_reward_dict[playlog_key].append(
        {
            "parameter": parameter.tolist(),
            "episode_serial": episode_serial,
            "reward": rewards,
            "playlog": list(map(lambda x: x.dict(), playlogs)),
        }
    )
    with open(reward_models_path / "playlog_reward.json", "w") as f:
        json.dump(playlog_reward_dict, f, indent=2)


def simulate(
    env: CustomRewardEnv,
    train_on_custom_reward: Callable[
        [CustomRewardEnv, np.ndarray], tuple[int, list[PlayLogModel], list[float]]
    ],
    solutions: np.ndarray,
    reward_keys: list[str],
    playlog_keys: list[str],
    reward_models_path: Path,
    playlog_reward_dict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """報酬重み（パラメータ）をもとにマリオをプレイさせる.

    Attributes
    ----------
    solutions: np.ndarray
        報酬重みのパラメータ (n, d)
        n はデータ数  d は重みの数
    reward_keys: list[str]
        利用する報酬重みの key リスト
    playlog_keys: list[str]
        利用するプレイログの key リスト

    Returns
    -------
    objectives: np.ndarray
        適応度 (n,)
    behaviors: np.ndarray
        特徴量（プレイログ） (n, r)  rはプレイログの要素の個数
        おそらくプレイログ空間作ったときの順番で入れる必要がある
    """
    objectives, behaviors = [], []

    for reward_parameter in solutions:
        reward_config = RewardConfig.init_with_keys(reward_parameter, reward_keys)
        reward_config.POSITION = 0.001  # TODO: 将来ここなんとかする
        reward_config.TIME = -0.001
        env.change_reward_config(reward_config)

        print("here")
        episode_serial, playlogs, rewards = train_on_custom_reward(
            env, reward_parameter
        )
        objective = playlogs[-1].goal
        behavior = PlayLogModel.select_with_keys(playlogs[-1], playlog_keys)
        save_playlog_reward_dict(
            reward_parameter,
            episode_serial,
            behavior,
            playlogs,
            rewards,
            reward_models_path,
            playlog_reward_dict,
        )

        objectives.append(objective)
        behaviors.append(behavior)
    return np.array(objectives), np.array(behaviors)


def get_train_on_custom_reward(
    env_config: EnvConfig,
    mario: Mario,
    logger: MetricLogger,
    checkpoint_path: Path,
    episode: int,
) -> Callable[
    [CustomRewardEnv, np.ndarray], tuple[int, list[PlayLogModel], list[float]]
]:
    """学習を行うコールバックを返す.

    報酬重みが変更された環境を与えると学習を行うコールバックを返す．
    env_config, mario, logger など固定される変数を先にキャプチャしてスッキリさせるためである．

    Notes
    -----
    callback を呼び出す前に報酬重みを変更する必要がある
    """
    episode_serial = episode

    def callback(
        env: CustomRewardEnv, reward_weights: np.ndarray
    ) -> tuple[int, list[PlayLogModel], list[float]]:
        nonlocal episode_serial
        rewards = []
        playlogs = []

        # TODO: ログ出力の内容を調整する
        for _ in range(env_config.EPISODES):
            state = env.reset()  # (4, 84, 84) LazyFrames

            sum_reward = 0
            while True:
                action = mario.act(state, reward_weights)
                if (
                    env_config.IS_RENDER
                    and episode_serial % env_config.EVERY_RENDER == 0
                ):
                    env.render()

                next_state, reward, done, info = env.step(action)
                sum_reward += reward

                mario.cache(state, next_state, action, reward, done, reward_weights)
                q, loss = mario.learn()

                logger.log_step(reward, loss, q)

                state = next_state

                if done:
                    break

            rewards.append(sum_reward)
            playlogs.append(info["playlog"])

            episode_serial += 1
            logger.log_episode()

            if episode_serial % env_config.EVERY_RECORD == 0:
                logger.record(
                    episode=episode_serial,
                    epsilon=mario.exploration_rate,
                    step=mario.curr_step,
                )

            if episode_serial % env_config.EVERY_EPISODE_SAVE == 0:
                save_episode_model(
                    mario,
                    checkpoint_path,
                    episode_serial,
                    mario.curr_step,
                    env_config.EVERY_EPISODE_SAVE,
                )

        # TODO: Average?
        return episode_serial, playlogs, rewards

    return callback


def relearn_pyribs(
    env_config_name: str,
    reward_scope_config_name: str,
    playlog_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
) -> None:
    # Config
    env_config_path = get_env_config_path(env_config_name)
    env_config = EnvConfig.create(str(env_config_path))
    reward_scope_config_path = get_reward_scope_config_path(reward_scope_config_name)
    reward_scope_config = RewardScopeConfig.create(str(reward_scope_config_path))
    playlog_scope_config_path = get_playlog_scope_config_path(playlog_scope_config_name)
    playlog_scope_config = PlayLogScopeConfig.create(str(playlog_scope_config_path))

    # Path
    results_path = get_results_path()
    date_path = get_date_path(results_path, date_str)
    checkpoint_path = get_checkpoint_path(date_path)
    episode_model_path = get_model_path(checkpoint_path, checkpoint_idx, "episode")
    reward_models_path = get_reward_models_path(date_path)
    pickles_path = get_pickles_path(date_path)
    _set_logger(date_path)

    # Restore
    loaded = torch.load(episode_model_path)
    model = loaded["model"]
    exploration_rate = loaded["exploration_rate"]
    episode = loaded["episode"]
    step = loaded["step"]
    archive, emitters, optimizer, logger, playlog_reward_dict = restore_objects(
        pickles_path, reward_models_path
    )
    print(f"exploration_rate: {exploration_rate} episode: {episode} step: {step}")

    # Pyribs
    playlog_ranges, playlog_bins, playlog_keys = PlayLogScopeConfig.take_out_use(
        playlog_scope_config
    )
    reward_bounds, reward_keys = RewardScopeConfig.take_out_use(reward_scope_config)
    print(
        f"[PLAYLOG] keys:{playlog_keys} ranges: {playlog_ranges} bins: {playlog_bins}"
    )
    print(f"[REWARD] keys:{reward_keys} bounds:{reward_bounds}")

    # Components
    env = get_env(env_config, RewardConfig.init())
    mario = ReLearnMario(
        state_dim=(env_config.NUM_STACK, env_config.SHAPE, env_config.SHAPE),
        action_dim=env.action_space.n,
        reward_dim=len(reward_keys),
        model=model,
        exploration_rate=exploration_rate,
        step=step,
    )
    # logger = MetricLogger(date_path)
    train_callback = get_train_on_custom_reward(
        env_config, mario, logger, checkpoint_path, episode
    )

    for _ in range(10000000):
        # パラメータ(報酬重み)空間 (データ数, 重み要素)
        solutions = optimizer.ask()

        objectives, behaviors = simulate(
            env,
            train_callback,
            solutions,
            reward_keys,
            playlog_keys,
            reward_models_path,
            playlog_reward_dict,
        )
        optimizer.tell(objectives, behaviors)

        with open(pickles_path / "archive.pickle", "wb") as f:
            pickle.dump(archive, f)
        with open(pickles_path / "emitters.pickle", "wb") as f:
            pickle.dump(emitters, f)
        with open(pickles_path / "optimzer.pickle", "wb") as f:
            pickle.dump(optimizer, f)
