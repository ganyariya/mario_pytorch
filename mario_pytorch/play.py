"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
https://github.com/YuansongFeng/MadMario/blob/master/agent.py
"""
import time

import torch

from mario_pytorch.agent.mario import LearnedMario
from mario_pytorch.util.config import EnvConfig, RewardConfig, RewardScopeConfig
from mario_pytorch.util.get_env import get_env
from mario_pytorch.util.process_path import (
    get_checkpoint_path,
    get_date_path,
    get_env_config_path,
    get_model_path,
    get_results_path,
    get_reward_scope_config_path,
)

# ----------------------------------------------------------------------

EPISODE_LENGTH = 5


def play(
    env_config_name: str,
    reward_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
    exploration_rate: float,
) -> None:
    # コンフィグ
    env_config_path = get_env_config_path(env_config_name)
    env_config = EnvConfig.create(str(env_config_path))
    reward_scope_config_path = get_reward_scope_config_path(reward_scope_config_name)
    reward_scope_config = RewardScopeConfig.create(str(reward_scope_config_path))
    reward_bounds, reward_keys = RewardScopeConfig.take_out_use(reward_scope_config)
    print(f"[REWARD] keys:{reward_keys} bounds:{reward_bounds}")

    # パス
    results_path = get_results_path()
    date_path = get_date_path(results_path, date_str)
    checkpoint_path = get_checkpoint_path(date_path)
    model_path = get_model_path(checkpoint_path, checkpoint_idx)

    # 環境
    model = torch.load(model_path)["model"]
    env = get_env(env_config, RewardConfig.init())

    mario = LearnedMario(
        state_dim=(env_config.NUM_STACK, env_config.SHAPE, env_config.SHAPE),
        action_dim=env.action_space.n,
        reward_dim=len(reward_keys),
        model=model,
        exploration_rate=exploration_rate,
    )

    for _ in range(10000):
        print(f"[REWARD] keys:{reward_keys} bounds:{reward_bounds}")
        reward_weights = list(map(float, input(f"Rewards {reward_keys}:").split()))
        reward_config = RewardConfig.init_with_keys(reward_weights, reward_keys)

        # カスタム
        # reward_config.POSITION = 0.001
        # reward_config.TIME = -0.001

        env.change_reward_config(reward_config)

        for e in range(EPISODE_LENGTH):

            state = env.reset()
            reward_sum = 0

            while True:
                action = mario.act(state, reward_weights)
                env.render()

                next_state, reward, done, info = env.step(action)
                reward_sum += reward
                state = next_state

                time.sleep(0.001)
                # print(info["playlog"])

                if done or info["default"].flag_get:
                    break

            print(reward_sum)
            print(info["playlog"])
