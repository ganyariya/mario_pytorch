"""
ある固定された報酬関数で学習できるかテストする用

pyribs 側の対応もあってかなりぐちゃぐちゃしている
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""

import numpy as np

from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger
from mario_pytorch.util.config import (
    EnvConfig,
    PlayLogScopeConfig,
    RewardConfig,
    RewardScopeConfig,
)
from mario_pytorch.util.export_onnx import export_onnx, transform_mario_input
from mario_pytorch.util.get_env import get_env
from mario_pytorch.util.process_path import (
    copy_and_backup_env_files,
    generate_README_file,
    get_checkpoint_path,
    get_env_config_path,
    get_playlog_scope_config_path,
    get_results_path,
    get_reward_scope_config_path,
    get_save_path,
)


def tmp_create_reward_config() -> RewardConfig:
    return RewardConfig(
        **{
            "POSITION": 1,
            "ENEMY": 50,
            "COIN": 30,
            "GOAL": 500,
            "LIFE": 200,
            "ITEM": 200,
            "TIME": -1,
            "SCORE": 0,
        }
    )


def revise_reward_weights(reward_config: RewardConfig) -> RewardConfig:
    for x in reward_config:
        setattr(reward_config, x[0], x[1] / 1000)
    return reward_config


# ----------------------------------------------------------------------


def get_reward_weights(reward_config: RewardConfig) -> list[float]:
    ret = []
    for x in reward_config:
        ret.append(x[1])
    return ret


def learn(env_config_name: str, reward_scope_config_name: str) -> None:
    env_config_path = get_env_config_path(env_config_name)
    env_config = EnvConfig.create(str(env_config_path))

    # とりあえず config を仮置して対応する
    reward_scope_config_path = get_reward_scope_config_path(reward_scope_config_name)
    playlog_scope_config_path = get_playlog_scope_config_path(reward_scope_config_name)
    reward_scope_config = RewardScopeConfig.create(str(reward_scope_config_path))
    playlog_scope_config = PlayLogScopeConfig.create(str(playlog_scope_config_path))

    reward_config = tmp_create_reward_config()
    reward_config = revise_reward_weights(reward_config)
    reward_weights = get_reward_weights(reward_config)
    print(reward_weights)
    print(reward_config)

    results_path = get_results_path()
    save_path = get_save_path(results_path)
    checkpoint_path = get_checkpoint_path(save_path)
    copy_and_backup_env_files(
        save_path, env_config, reward_scope_config, playlog_scope_config
    )
    generate_README_file(save_path)

    env = get_env(env_config, reward_config)
    logger = MetricLogger(save_path)
    mario = Mario(
        state_dim=(env_config.NUM_STACK, env_config.SHAPE, env_config.SHAPE),
        action_dim=env.action_space.n,
        reward_dim=len(reward_weights),
        checkpoint_path=checkpoint_path,
    )
    export_onnx(
        mario.online_net,
        env.reset(),
        np.zeros(len(reward_weights)),
        transform_mario_input,
        save_path,
    )

    for e in range(env_config.EPISODES):

        state = env.reset()

        while True:
            action = mario.act(state, reward_weights)
            if env_config.IS_RENDER and e % env_config.EVERY_RENDER == 0:
                env.render()

            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done, reward_weights)
            q, loss = mario.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done or info["default"].flag_get:
                break

        logger.log_episode()
        if e % env_config.EVERY_RECORD == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )
