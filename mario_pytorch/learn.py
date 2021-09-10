"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""

import time

from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger
from mario_pytorch.util.config import EnvConfig, RewardConfig, RewardScopeConfig
from mario_pytorch.util.export_onnx import export_onnx, transform_mario_input
from mario_pytorch.util.get_env import get_env
from mario_pytorch.util.process_path import (
    copy_and_save_env_files,
    generate_README_file,
    get_checkpoint_path,
    get_env_config_path,
    get_results_path,
    get_save_path,
)


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


def learn(env_config_name: str, reward_scope_config_name: str) -> None:
    env_config_path = get_env_config_path(env_config_name)
    env_config = EnvConfig.create(str(env_config_path))
    reward_config = tmp_create_reward_config()

    results_path = get_results_path()
    save_path = get_save_path(results_path)
    checkpoint_path = get_checkpoint_path(save_path)
    copy_and_save_env_files(save_path, env_config, reward_config)
    generate_README_file(save_path)

    env = get_env(env_config, reward_config)
    logger = MetricLogger(save_path)
    mario = Mario(
        state_dim=(env_config.NUM_STACK, env_config.SHAPE, env_config.SHAPE),
        action_dim=env.action_space.n,
        checkpoint_path=checkpoint_path,
    )
    export_onnx(mario.online_net, env.reset(), transform_mario_input, save_path)

    for e in range(env_config.EPISODES):

        # state.shape (4, 84, 84)
        # state.frame_shape (84, 84)
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

            if done or info["default"]["flag_get"]:
                break

        logger.log_episode()
        if e % env_config.EVERY_RECORD == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )
