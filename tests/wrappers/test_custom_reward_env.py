import numpy as np
from mario_pytorch.wrappers.custom_reward_env import CustomRewardEnv


def test_reward_x(make_env: CustomRewardEnv):
    info = {"x_pos": np.array(10)}
    reward = make_env.process_reward_x(info)
    assert reward == 10
    assert make_env.pprev_x == 10
