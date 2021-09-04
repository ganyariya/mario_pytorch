import numpy as np
from mario_pytorch.wrappers.custom_reward_env import CustomRewardEnv


def test_reward_x(make_env: CustomRewardEnv):
    info = {"x_pos": np.array(10)}
    reward = make_env.process_reward_x(info)
    assert reward == 10
    assert make_env.pprev_x == 10


def test_reward_coin_normal(make_env: CustomRewardEnv):
    info = {"coins": 10}
    reward = make_env.process_reward_coin(info)
    assert reward == 10
    assert make_env.pprev_coin == 10


def test_reward_coin_over(make_env: CustomRewardEnv):
    info = {"coins": 98}
    reward = make_env.process_reward_coin(info)
    assert reward == 98
    info = {"coins": 5}
    reward = make_env.process_reward_coin(info)
    assert reward == 7
    assert make_env.pprev_coin == 5


def test_reward_life(make_env: CustomRewardEnv):
    info = {"life": np.array(2)}
    reward = make_env.process_reward_life(info)
    assert reward == 0
    info = {"life": np.array(255)}
    reward = make_env.process_reward_life(info)
    assert reward == 3
    assert make_env.pprev_life == -1
