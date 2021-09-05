"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
https://github.com/YuansongFeng/MadMario/blob/master/agent.py
"""

import torch

from pathlib import Path
from mario_pytorch.agent.mario import LearnedMario
from mario_pytorch.util.config import EnvConfig, RewardConfig
from mario_pytorch.util.get_env import get_env

from mario_pytorch.util.process_path import (
    get_date_path,
    get_env_config_path,
    get_results_path,
    get_date_path,
    get_model_path,
)


def tmp_create_reward_config() -> RewardConfig:
    return RewardConfig(
        **{
            "POSITION": 1,
            "ENEMY": 1,
            "COIN": 1,
            "GOAL": 1,
            "LIFE": 1,
            "ITEM": 1,
            "TIME": 1,
            "SCORE": 1,
        }
    )


# ----------------------------------------------------------------------


def play(env_config_name: str, date_str: str, checkpoint_idx: int) -> None:
    env_config_path = get_env_config_path(env_config_name)
    env_config = EnvConfig.create(str(env_config_path))
    reward_config = tmp_create_reward_config()

    results_path = get_results_path()
    date_path = get_date_path(results_path, date_str)
    model_path = get_model_path(date_path, checkpoint_idx)
    model = torch.load(model_path)["model"]

    env = get_env(env_config, reward_config)

    mario = LearnedMario(
        state_dim=(env_config.NUM_STACK, env_config.SHAPE, env_config.SHAPE),
        action_dim=env.action_space.n,
        model=model,
    )

    for e in range(env_config.EPISODES):

        # state.shape (4, 84, 84)  state.frame_shape (84, 84)
        state = env.reset()
        reward_sum = 0

        while True:
            action = mario.act(state)
            env.render()

            next_state, reward, done, info = env.step(action)
            reward_sum += reward

            state = next_state

            if done or info["flag_get"]:
                break

        print(reward_sum)


if __name__ == "__main__":
    play("base.yaml", "2021-08-29T15-32-18", 10)
