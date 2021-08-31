import gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from mario_pytorch.util.config import Config
from mario_pytorch.wrappers import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    CustomRewardEnv,
)


def get_env_name(world: int, stage: int, version: int) -> str:
    return f"SuperMarioBros-{world}-{stage}-v{version}"


def get_env(config: Config) -> gym.Env:
    # 4 Consecutive GrayScale Frames Set [4, 84, 84]
    env = gym_super_mario_bros.make(
        get_env_name(config.WORLD, config.STAGE, config.VERSION)
    )
    env = CustomRewardEnv(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=config.SKIP_FRAME)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=config.SHAPE)
    env = FrameStack(env, num_stack=config.NUM_STACK)
    return env
