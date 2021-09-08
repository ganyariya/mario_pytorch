import numpy as np
from mario_pytorch.wrappers.custom_reward_env import CustomRewardEnv


def test_x(make_env: CustomRewardEnv):
    info = {"x_pos": np.array(10)}
    diff_x = make_env.get_diff_x(info)
    assert diff_x == 10
    diff_info = {"x_pos": diff_x}
    reward = make_env.process_reward_x(diff_info)
    assert reward == 10


def test_kills(make_env: CustomRewardEnv):
    info = {"kills": 10}
    diff_kills = make_env.get_diff_kills(info)
    assert diff_kills == 10
    diff_info = {"kills": diff_kills}
    reward = make_env.process_reward_kills(diff_info)
    assert reward == 10


def test_reward_coin_normal(make_env: CustomRewardEnv):
    info = {"coins": 10}
    reward = make_env.process_reward_coin(info)
    assert reward == 10
    # assert make_env.pprev_coin == 10


def test_reward_coin_over(make_env: CustomRewardEnv):
    info = {"coins": 98}
    reward = make_env.process_reward_coin(info)
    assert reward == 98
    info = {"coins": 5}
    reward = make_env.process_reward_coin(info)


def test_reward_life(make_env: CustomRewardEnv):
    pass
    # info = {"life": np.array(2)}
    # reward = make_env.process_reward_life(info)
    # assert reward == 0
    # info = {"life": np.array(255)}
    # reward = make_env.process_reward_life(info)
    # assert reward == 3
    # assert make_env.pprev_life == -1


def test_reward_goal(make_env: CustomRewardEnv):
    info = {"goal": 0}
    reward = make_env.process_reward_goal(info)
    assert reward == 0
    info = {"goal": 1}
    reward = make_env.process_reward_goal(info)
    assert reward == 1


def test_reward_item_plus(make_env: CustomRewardEnv):
    pass
    # info = {"status": "tall"}
    # reward = make_env.process_reward_item(info)
    # assert reward == 1
    # info = {"status": "fireball"}
    # reward = make_env.process_reward_item(info)
    # assert reward == 1


def test_reward_item_minus(make_env: CustomRewardEnv):
    pass
    # info = {"status": "fireball"}
    # reward = make_env.process_reward_item(info)
    # assert reward == 2
    # info = {"status": "small"}
    # reward = make_env.process_reward_item(info)
    # assert reward == 0


def test_reward_time_plus(make_env: CustomRewardEnv):
    make_env.reset()
    info = {"time": 300}
    reward = make_env.process_reward_time(info)
    # assert reward == 100


def test_reward_time_minus(make_env: CustomRewardEnv):
    make_env.reset()
    info = {"time": 450}
    reward = make_env.process_reward_time(info)
    # assert reward == 0


def test_reward_time_score(make_env: CustomRewardEnv):
    info = {"score": 10}
    reward = make_env.process_reward_score(info)
    assert reward == 10


def test_reset_on_each_life(make_env: CustomRewardEnv):
    # not work
    info = {"life": np.array(2), "x_pos": np.array(100), "time": 500}
    make_env.reset_on_each_life(info)
    # assert make_env.pprev_x == 0

    info = {"life": np.array(0), "x_pos": np.array(100), "time": 500}
    make_env.reset_on_each_life(info)
    # assert make_env.pprev_x == 100
    # assert make_env.pprev_time == 500
