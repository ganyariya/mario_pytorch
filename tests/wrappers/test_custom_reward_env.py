import numpy as np
from mario_pytorch.wrappers.custom import CustomRewardEnv
from mario_pytorch.wrappers.custom.custom_info_model import InfoModel

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
    info_model = InfoModel.init()
    info_model.x_pos = 10
    diff_x = make_env.get_diff_x(info_model)
    assert diff_x == 10
    diff_info = {"x_pos": diff_x}
    reward = make_env.process_reward_x(diff_info)
    assert reward == 10
    make_env.update_pprev_x(info_model)
    assert make_env.pprev_x == 10


def test_kills(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.kills = 10
    diff_kills = make_env.get_diff_kills(info_model)
    assert diff_kills == 10
    diff_info = {"kills": diff_kills}
    reward = make_env.process_reward_kills(diff_info)
    assert reward == 10
    make_env.update_pprev_kills(info_model)
    assert make_env.pprev_kills == 10


def test_normal_coins(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.coins = 10
    diff_coins = make_env.get_diff_coins(info_model)
    assert diff_coins == 10
    diff_info = {"coins": diff_coins}
    reward = make_env.process_reward_coin(diff_info)
    assert reward == 10
    make_env.update_pprev_coin(info_model)
    assert make_env.pprev_coin == 10


def test_over_coins(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.coins = 95
    make_env.update_pprev_coin(info_model)

    info_model.coins = 10
    diff_coins = make_env.get_diff_coins(info_model)
    assert diff_coins == 15
    diff_info = {"coins": diff_coins}
    reward = make_env.process_reward_coin(diff_info)
    assert reward == 15
    make_env.update_pprev_coin(info_model)
    assert make_env.pprev_coin == 10


def test_reward_life_add(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.life = 3
    diff_life = make_env.get_diff_life(info_model)
    assert diff_life == 1
    diff_info = {"life": diff_life}
    reward = make_env.process_reward_life(diff_info)
    assert reward == 1
    make_env.update_pprev_life(info_model)
    assert make_env.pprev_life == 3


def test_reward_life_dec(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.life = 1
    diff_life = make_env.get_diff_life(info_model)
    assert diff_life == -1
    diff_info = {"life": diff_life}
    reward = make_env.process_reward_life(diff_info)
    assert reward == -1
    make_env.update_pprev_life(info_model)
    assert make_env.pprev_life == 1


def test_reward_life_minus(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.life = 255
    diff_life = make_env.get_diff_life(info_model)
    assert diff_life == -3
    diff_info = {"life": diff_life}
    reward = make_env.process_reward_life(diff_info)
    assert reward == -3
    make_env.update_pprev_life(info_model)
    assert make_env.pprev_life == -1


def test_reward_not_goal(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.flag_get = False
    diff_goal = make_env.get_diff_goal(info_model)
    assert diff_goal == 0
    diff_info = {"goal": 0}
    reward = make_env.process_reward_goal(diff_info)
    assert reward == 0


def test_reward_goal(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.flag_get = True
    diff_goal = make_env.get_diff_goal(info_model)
    assert diff_goal == 1
    diff_info = {"goal": 1}
    reward = make_env.process_reward_goal(diff_info)
    assert reward == 1


def test_reward_item_plus(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.status = "tall"
    diff_item = make_env.get_diff_item(info_model)
    assert diff_item == 1
    diff_info = {"item": 1}
    reward = make_env.process_reward_item(diff_info)
    assert reward == 1
    make_env.update_pprev_status(info_model)
    assert make_env.pprev_status == 1


def test_reward_item_minus(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.status = "tall"
    make_env.update_pprev_status(info_model)

    info_model.status = "small"
    diff_item = make_env.get_diff_item(info_model)
    assert diff_item == -1
    diff_info = {"item": -1}
    reward = make_env.process_reward_item(diff_info)
    assert reward == -1
    make_env.update_pprev_status(info_model)
    assert make_env.pprev_status == 0


def test_reward_time(make_env: CustomRewardEnv):
    make_env.reset()
    info_model = InfoModel.init()
    info_model.time = 300
    diff_time = make_env.get_diff_time(info_model)
    assert diff_time == 100
    diff_info = {"elapsed": diff_time}
    reward = make_env.process_reward_elapsed(diff_info)
    assert reward == 100
    make_env.update_pprev_time(info_model)
    assert make_env.pprev_time == 300


def test_reward_time_score(make_env: CustomRewardEnv):
    make_env.reset()
    info_model = InfoModel.init()
    info_model.score = 300
    diff_score = make_env.get_diff_score(info_model)
    assert diff_score == 300
    diff_info = {"score": diff_score}
    reward = make_env.process_reward_score(diff_info)
    assert reward == 300
    make_env.update_pprev_score(info_model)
    assert make_env.pprev_score == 300


def test_reset_on_each_life(make_env: CustomRewardEnv):
    # not work
    info_model = InfoModel.init()
    info_model.life = 2
    info_model.x_pos = 100
    info_model.time = 500
    make_env.reset_on_each_life(info_model)
    assert make_env.pprev_x == 0
    assert make_env.pprev_time == 0
    assert make_env.pprev_life == 2

    # work
    info_model.life = 0
    make_env.reset_on_each_life(info_model)
    assert make_env.pprev_x == 100
    assert make_env.pprev_time == 500
    assert make_env.pprev_life == 2


# *--------------------------------------------*
# * accumulate
# *--------------------------------------------*


def test_accumulate_x(make_env: CustomRewardEnv):
    info_model = InfoModel.init()
    info_model.x_pos = 20
    diff_info = {"x_pos": 20}

    make_env.accumulate_x(info_model, diff_info)
    assert make_env.playlog["x_pos"] == 20
    assert make_env.playlog["x_abs"] == 20
    assert make_env.playlog["x_plus"] == 20
    assert make_env.playlog["x_minus"] == 00

    info_model = InfoModel.init()
    info_model.x_pos = 10
    diff_info = {"x_pos": -10}

    make_env.accumulate_x(info_model, diff_info)
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
