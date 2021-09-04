import gym_super_mario_bros
from mario_pytorch.util.config import EnvConfig, RewardConfig


def test_env_config(make_env_config: EnvConfig) -> None:
    assert make_env_config.STAGE == -1


def test_reward_config(make_reward_config: RewardConfig) -> None:
    assert make_reward_config.POSITION == 1


def test_env(make_env: gym_super_mario_bros.SuperMarioBrosEnv):
    assert True
