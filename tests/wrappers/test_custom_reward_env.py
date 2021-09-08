import numpy as np
from mario_pytorch.wrappers.custom_reward_env import CustomRewardEnv

# *--------------------------------------------*
# * Main
# *--------------------------------------------*


def test_process_reward(make_env: CustomRewardEnv):
    diff_info = {
        "x_pos": 1,
        "coins": 2,
        "life": 3,
        "goal": 4,
        "item": 5,
        "elapsed": 6,
        "score": 7,
        "kills": 8,
    }
    reward, reward_dict = make_env.process_reward(diff_info)
    assert reward == 36
    assert reward_dict["x_pos"] == 1
    assert reward_dict["coins"] == 2
    assert reward_dict["life"] == 3
    assert reward_dict["goal"] == 4
    assert reward_dict["item"] == 5
    assert reward_dict["elapsed"] == 6
    assert reward_dict["score"] == 7
    assert reward_dict["kills"] == 8


# *--------------------------------------------*
# * Each
# *--------------------------------------------*


def test_x(make_env: CustomRewardEnv):
    info = {"x_pos": np.array(10)}
    diff_x = make_env.get_diff_x(info)
    assert diff_x == 10
    diff_info = {"x_pos": diff_x}
    reward = make_env.process_reward_x(diff_info)
    assert reward == 10
    make_env.update_pprev_x(info)
    assert make_env.pprev_x == 10


def test_kills(make_env: CustomRewardEnv):
    info = {"kills": 10}
    diff_kills = make_env.get_diff_kills(info)
    assert diff_kills == 10
    diff_info = {"kills": diff_kills}
    reward = make_env.process_reward_kills(diff_info)
    assert reward == 10
    make_env.update_pprev_kills(info)
    assert make_env.pprev_kills == 10


def test_normal_coins(make_env: CustomRewardEnv):
    info = {"coins": 10}
    diff_coins = make_env.get_diff_coins(info)
    assert diff_coins == 10
    diff_info = {"coins": diff_coins}
    reward = make_env.process_reward_coin(diff_info)
    assert reward == 10
    make_env.update_pprev_coin(info)
    assert make_env.pprev_coin == 10


def test_over_coins(make_env: CustomRewardEnv):
    make_env.update_pprev_coin({"coins": 95})
    info = {"coins": 10}
    diff_coins = make_env.get_diff_coins(info)
    assert diff_coins == 15
    diff_info = {"coins": diff_coins}
    reward = make_env.process_reward_coin(diff_info)
    assert reward == 15
    make_env.update_pprev_coin(info)
    assert make_env.pprev_coin == 10


def test_reward_life_add(make_env: CustomRewardEnv):
    info = {"life": np.array(3)}
    diff_life = make_env.get_diff_life(info)
    assert diff_life == 1
    diff_info = {"life": diff_life}
    reward = make_env.process_reward_life(diff_info)
    assert reward == 1
    make_env.update_pprev_life(info)
    assert make_env.pprev_life == 3


def test_reward_life_dec(make_env: CustomRewardEnv):
    info = {"life": np.array(1)}
    diff_life = make_env.get_diff_life(info)
    assert diff_life == -1
    diff_info = {"life": diff_life}
    reward = make_env.process_reward_life(diff_info)
    assert reward == -1
    make_env.update_pprev_life(info)
    assert make_env.pprev_life == 1


def test_reward_life_minus(make_env: CustomRewardEnv):
    info = {"life": np.array(255)}
    diff_life = make_env.get_diff_life(info)
    assert diff_life == -3
    diff_info = {"life": diff_life}
    reward = make_env.process_reward_life(diff_info)
    assert reward == -3
    make_env.update_pprev_life(info)
    assert make_env.pprev_life == -1


def test_reward_not_goal(make_env: CustomRewardEnv):
    info = {"flag_get": 0}
    diff_goal = make_env.get_diff_goal(info)
    assert diff_goal == 0
    diff_info = {"goal": 0}
    reward = make_env.process_reward_goal(diff_info)
    assert reward == 0


def test_reward_goal(make_env: CustomRewardEnv):
    info = {"flag_get": 1}
    diff_goal = make_env.get_diff_goal(info)
    assert diff_goal == 1
    diff_info = {"goal": 1}
    reward = make_env.process_reward_goal(diff_info)
    assert reward == 1


def test_reward_item_plus(make_env: CustomRewardEnv):
    info = {"status": "tall"}
    diff_item = make_env.get_diff_item(info)
    assert diff_item == 1
    diff_info = {"item": 1}
    reward = make_env.process_reward_item(diff_info)
    assert reward == 1
    make_env.update_pprev_status(info)
    assert make_env.pprev_status == 1


def test_reward_item_minus(make_env: CustomRewardEnv):
    make_env.update_pprev_status({"status": "tall"})
    info = {"status": "small"}
    diff_item = make_env.get_diff_item(info)
    assert diff_item == -1
    diff_info = {"item": -1}
    reward = make_env.process_reward_item(diff_info)
    assert reward == -1
    make_env.update_pprev_status(info)
    assert make_env.pprev_status == 0


def test_reward_time(make_env: CustomRewardEnv):
    make_env.reset()
    info = {"time": 300}
    diff_time = make_env.get_diff_time(info)
    assert diff_time == 100
    diff_info = {"elapsed": diff_time}
    reward = make_env.process_reward_elapsed(diff_info)
    assert reward == 100
    make_env.update_pprev_time(info)
    assert make_env.pprev_time == 300


def test_reward_time_score(make_env: CustomRewardEnv):
    make_env.reset()
    info = {"score": 300}
    diff_score = make_env.get_diff_score(info)
    assert diff_score == 300
    diff_info = {"score": diff_score}
    reward = make_env.process_reward_score(diff_info)
    assert reward == 300
    make_env.update_pprev_score(info)
    assert make_env.pprev_score == 300


def test_reset_on_each_life(make_env: CustomRewardEnv):
    # not work
    info = {"life": np.array(2), "x_pos": np.array(100), "time": 500}
    make_env.reset_on_each_life(info)
    assert make_env.pprev_x == 0

    # work
    info = {"life": np.array(0), "x_pos": np.array(100), "time": 500}
    make_env.reset_on_each_life(info)
    assert make_env.pprev_x == 100
    assert make_env.pprev_time == 500


# *--------------------------------------------*
# * accumulate
# *--------------------------------------------*


def test_accumulate_x(make_env: CustomRewardEnv):
    info = {"x_pos": 20}
    diff_info = {"x_pos": 20}

    make_env.accumulate_x(info, diff_info)
    assert make_env.playlog["x_pos"] == 20
    assert make_env.playlog["x_abs"] == 20
    assert make_env.playlog["x_plus"] == 20
    assert make_env.playlog["x_minus"] == 00

    info = {"x_pos": 10}
    diff_info = {"x_pos": -10}

    make_env.accumulate_x(info, diff_info)
    assert make_env.playlog["x_pos"] == 10
    assert make_env.playlog["x_abs"] == 30
    assert make_env.playlog["x_plus"] == 20
    assert make_env.playlog["x_minus"] == -10


def test_accumulate_coins(make_env: CustomRewardEnv):
    diff_info = {"coins": 10}
    make_env.accumulate_coins(diff_info)
    assert make_env.playlog["coins"] == 10
    diff_info = {"coins": 20}
    make_env.accumulate_coins(diff_info)
    assert make_env.playlog["coins"] == 30


def test_accumulate_kills(make_env: CustomRewardEnv):
    diff_info = {"kills": 10}
    make_env.accumulate_kills(diff_info)
    assert make_env.playlog["kills"] == 10
    diff_info = {"kills": 20}
    make_env.accumulate_kills(diff_info)
    assert make_env.playlog["kills"] == 30


def test_accumulate_life(make_env: CustomRewardEnv):
    diff_info = {"life": 5}
    make_env.accumulate_life(diff_info)
    assert make_env.playlog["life"] == 5
    assert make_env.playlog["life_plus"] == 5
    assert make_env.playlog["life_minus"] == 0

    diff_info = {"life": -2}
    make_env.accumulate_life(diff_info)
    assert make_env.playlog["life"] == 3
    assert make_env.playlog["life_plus"] == 5
    assert make_env.playlog["life_minus"] == -2


def test_accumulate_goal(make_env: CustomRewardEnv):
    diff_info = {"goal": 1}
    for i in range(1, 5):
        make_env.accumulate_goal(diff_info)
        assert make_env.playlog["goal"] == i


def test_accumulate_item(make_env: CustomRewardEnv):
    diff_info = {"item": 2}
    make_env.accumulate_item(diff_info)
    assert make_env.playlog["item"] == 2
    assert make_env.playlog["item_plus"] == 2
    assert make_env.playlog["item_minus"] == 0

    diff_info = {"item": -1}
    make_env.accumulate_item(diff_info)
    assert make_env.playlog["item"] == 1
    assert make_env.playlog["item_plus"] == 2
    assert make_env.playlog["item_minus"] == -1


def test_accumulate_elapsed(make_env: CustomRewardEnv):
    diff_info = {"elapsed": 100}
    make_env.accumulate_elapsed(diff_info)
    assert make_env.playlog["elapsed"] == 100
    diff_info = {"elapsed": -50}
    make_env.accumulate_elapsed(diff_info)
    assert make_env.playlog["elapsed"] == 150


def test_accumulate_score(make_env: CustomRewardEnv):
    diff_info = {"score": 10}
    make_env.accumulate_score(diff_info)
    assert make_env.playlog["score"] == 10
    diff_info = {"score": 30}
    make_env.accumulate_score(diff_info)
    assert make_env.playlog["score"] == 40
