from mario_pytorch.wrappers.custom.custom_info_model import DiffInfoModel, InfoModel
from mario_pytorch.util.config.reward_config import RewardConfig
from mario_pytorch.wrappers.custom import CustomRewardEnv


def test_custom_change_reward(make_env: CustomRewardEnv):
    # 修正前の挙動テスト
    diff_info_model = DiffInfoModel.init()
    diff_info_model.x_pos = 10
    reward = make_env.process_reward_x(diff_info_model)
    assert reward == 10

    # 報酬変更する
    new_reward_config = RewardConfig(
        **{
            "POSITION": 3,
            "ENEMY": 50,
            "COIN": 30,
            "GOAL": 500,
            "LIFE": 100,
            "ITEM": 200,
            "TIME": -1,
            "SCORE": 0,
        }
    )
    make_env.change_reward_config(new_reward_config)
    diff_info_model = DiffInfoModel.init()
    diff_info_model.x_pos = 10
    diff_info_model.coins = 2
    diff_info_model.goal = 3
    diff_info_model.life = 2
    diff_info_model.elapsed = 5

    reward = make_env.process_reward_x(diff_info_model)
    assert reward == 30
    reward = make_env.process_reward_coin(diff_info_model)
    assert reward == 60
    reward = make_env.process_reward_goal(diff_info_model)
    assert reward == 1500
    reward = make_env.process_reward_life(diff_info_model)
    assert reward == 200
    reward = make_env.process_reward_elapsed(diff_info_model)
    assert reward == -5
