"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""

import datetime

from pathlib import Path
from os import path

import yaml
import gym_super_mario_bros

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from mario_pytorch.wrappers import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    CustomRewardEnv,
)
from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger
from mario_pytorch.util.get_env_name import get_env_name
from mario_pytorch.util.config import Config
from mario_pytorch.util.export_onnx import export_onnx, transform_mario_input

# ----------------------------------------------------------------------

config_path = Path(__file__).parents[1] / "config" / "base.yaml"
config = Config.create(str(config_path))

save_dir = (
    Path(path.dirname(__file__)).parent
    / "checkpoints"
    / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
)
save_dir.mkdir(parents=True)
with open(save_dir / "used_config.yaml", "w") as f:
    yaml.safe_dump(config.dict(), f, encoding="utf-8", allow_unicode=True)

# 4 Consecutive GrayScale Frames Set [4, 84, 84]
env = gym_super_mario_bros.make(
    get_env_name(config.WORLD, config.STAGE, config.VERSION)
)
env = CustomRewardEnv(env)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=config.SKIP_FRAME)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=config.SHAPE)
env = FrameStack(env, num_stack=config.NUM_STACK)  # 4Frame まとめて取り出す

mario = Mario(
    state_dim=(config.NUM_STACK, config.SHAPE, config.SHAPE),
    action_dim=env.action_space.n,
    save_dir=save_dir,
)
logger = MetricLogger(save_dir)
export_onnx(mario.online_net, env.reset(), transform_mario_input)

for e in range(config.EPISODES):

    # state.shape (4, 84, 84)
    # state.frame_shape (84, 84)
    state = env.reset()

    while True:
        action = mario.act(state)
        if config.IS_RENDER and e % config.EVERY_RENDER == 0:
            env.render()

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()
    if e % config.EVERY_RECORD == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
