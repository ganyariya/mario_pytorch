import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_pytorch.util.config import EnvConfig, RewardConfig
from mario_pytorch.wrappers import (
    CustomRewardEnv,
    GrayScaleObservation,
    ResizeObservation,
    SkipFrame,
)


def _get_env_name(world: int, stage: int, version: int) -> str:
    if world == -1 and stage == -1:
        return f"SuperMarioBros-v{version}"
    return f"SuperMarioBros-{world}-{stage}-v{version}"


def get_env(config: EnvConfig, reward_config: RewardConfig) -> gym.Env:
    """

    Notes
    -----
    env.step を 利用側ですると，FrameStack -> ResizeObservation ... -> CustomRewardEnv
    のような順番で呼び出される

    ただし self.env.step(action) で呼び出されるので，結局実行される順番は後がけであるため
    Joypad -> Custom -> SkipFrame -> ... -> FrameStack となる
    """
    # 4 Consecutive GrayScale Frames Set [4, 84, 84]
    env = gym_super_mario_bros.make(
        _get_env_name(config.WORLD, config.STAGE, config.VERSION)
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomRewardEnv(env, reward_config)
    env = SkipFrame(env, skip=config.SKIP_FRAME)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=config.SHAPE)
    env = FrameStack(env, num_stack=config.NUM_STACK)
    return env
