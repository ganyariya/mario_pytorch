from typing import Callable
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer

from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger
from mario_pytorch.util.config import (
    EnvConfig,
    RewardConfig,
    RewardScopeConfig,
    PlayLogScopeConfig,
)
from mario_pytorch.util.export_onnx import export_onnx, transform_mario_input
from mario_pytorch.util.get_env import get_env
from mario_pytorch.util.process_path import (
    copy_and_backup_env_files,
    generate_README_file,
    get_checkpoint_path,
    get_env_config_path,
    get_reward_scope_config_path,
    get_playlog_scope_config_path,
    get_results_path,
    get_save_path,
)
from mario_pytorch.wrappers.custom import CustomRewardEnv
from mario_pytorch.wrappers.custom.custom_info_model import PlayLogModel


def tmp_create_reward_config() -> RewardConfig:
    return RewardConfig(
        **{
            "POSITION": 1,
            "ENEMY": 50,
            "COIN": 30,
            "GOAL": 500,
            "LIFE": -200,
            "ITEM": 200,
            "TIME": -1,
            "SCORE": 0,
        }
    )


# ----------------------------------------------------------------------


def simulate(
    env: CustomRewardEnv,
    train_on_custom_reward: Callable[[CustomRewardEnv], PlayLogModel],
    solutions: np.ndarray,
    reward_keys: list[str],
    playlog_keys: list[str],
) -> np.ndarray:
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

    Notes
    -----
    behaviors は入れるかわからん
    ただ，多分入れたほうが楽な感じする
    """
    for parameter in solutions:
        reward_config = RewardConfig.init_with_keys(parameter, reward_keys)
        env.change_reward_config(reward_config)
        env.confirm_reward_config()
        playlog = train_on_custom_reward(env)


def get_train_on_custom_reward(
    env_config: EnvConfig, mario: Mario, logger: MetricLogger
) -> Callable[[CustomRewardEnv], PlayLogModel]:
    def callback(env: CustomRewardEnv) -> PlayLogModel:
        # TODO: ログ出力の内容を調整する
        for e in range(env_config.EPISODES):
            state = env.reset()

            while True:
                action = mario.act(state)
                if env_config.IS_RENDER and e % env_config.EVERY_RENDER == 0:
                    env.render()

                next_state, reward, done, info = env.step(action)

                mario.cache(state, next_state, action, reward, done)
                q, loss = mario.learn()

                logger.log_step(reward, loss, q)

                state = next_state

                if done or info["default"].flag_get:
                    break

            # TODO: このあたりもうまく調整する
            logger.log_episode()
            if e % env_config.EVERY_RECORD == 0:
                logger.record(
                    episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
                )

        return info["playlog"]

    return callback


def learn_pyribs(
    env_config_name: str, reward_scope_config_name: str, playlog_scope_config_name: str
) -> None:
    # パス・設定ファイル
    env_config_path = get_env_config_path(env_config_name)
    env_config = EnvConfig.create(str(env_config_path))
    reward_scope_config_path = get_reward_scope_config_path(reward_scope_config_name)
    reward_scope_config = RewardScopeConfig.create(str(reward_scope_config_path))
    playlog_scope_config_path = get_playlog_scope_config_path(playlog_scope_config_name)
    playlog_scope_config = PlayLogScopeConfig.create(str(playlog_scope_config_path))

    results_path = get_results_path()
    save_path = get_save_path(results_path)
    checkpoint_path = get_checkpoint_path(save_path)
    copy_and_backup_env_files(
        save_path, env_config, reward_scope_config, playlog_scope_config
    )
    generate_README_file(save_path)

    # 行動空間
    playlog_ranges, playlog_bins, playlog_keys = PlayLogScopeConfig.take_out_use(
        playlog_scope_config
    )
    archive = GridArchive(dims=playlog_bins, ranges=playlog_ranges)
    print(
        f"[PLAYLOG] keys:{playlog_keys} ranges: {playlog_ranges} bins: {playlog_bins}"
    )

    # パラメータ空間
    reward_bounds, reward_keys = RewardScopeConfig.take_out_use(reward_scope_config)
    emitters = [
        ImprovementEmitter(
            archive, x0=[0] * len(reward_bounds), sigma0=1, bounds=reward_bounds
        )
    ]
    print(f"[REWARD] keys:{reward_keys} bounds:{reward_bounds}")

    optimizer = Optimizer(archive=archive, emitters=emitters)

    env = get_env(env_config, RewardConfig.init())
    mario = Mario(
        state_dim=(env_config.NUM_STACK, env_config.SHAPE, env_config.SHAPE),
        action_dim=env.action_space.n,
        checkpoint_path=checkpoint_path,
    )
    export_onnx(mario.online_net, env.reset(), transform_mario_input, save_path)
    logger = MetricLogger(save_path)
    train_on_custom_reward = get_train_on_custom_reward(env_config, mario, logger)

    for _ in range(2):
        # パラメータ(報酬重み)空間 (データ数, 重み要素)
        solutions = optimizer.ask()
        print(solutions.shape)
        print(solutions)

        objectives = simulate(
            env, train_on_custom_reward, solutions, reward_keys, playlog_keys
        )
