import pytest

from mario_pytorch.util.config import EnvConfig, RewardConfig
from mario_pytorch.util.get_env import get_env


@pytest.fixture
def make_env_config() -> EnvConfig:
    return EnvConfig(
        **{
            "WORLD": -1,
            "STAGE": -1,
            "VERSION": 0,
            "SHAPE": 84,
            "SKIP_FRAME": 4,
            "NUM_STACK": 4,
            "IS_RENDER": False,
            "EPISODES": 3,
            "EVERY_RECORD": 1,
            "EVERY_RENDER": 1,
            "INTENTION": "Test",
        }
    )


@pytest.fixture
def make_reward_config() -> RewardConfig:
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


@pytest.fixture
def make_env(make_env_config: EnvConfig, make_reward_config: RewardConfig):
    env = get_env(make_env_config, make_reward_config)
    return env
