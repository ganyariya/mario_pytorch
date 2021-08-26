"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
https://github.com/YuansongFeng/MadMario/blob/master/agent.py
"""

import datetime

from pathlib import Path
from os import path

import torch
import gym_super_mario_bros

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from mario_pytorch.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger
from mario_pytorch.util.get_env_name import get_env_name
from mario_pytorch.util.config import Config

# ----------------------------------------------------------------------

config_path = Path(__file__).parents[1] / "config" / "base.yaml"
config = Config.create(str(config_path))

save_dir = (
    Path(path.dirname(__file__)).parent
    / "checkpoints"
    / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
)
save_dir.mkdir(parents=True)

# 4 Consecutive GrayScale Frames Set [4, 84, 84]
env = gym_super_mario_bros.make(
    get_env_name(config.WORLD, config.STAGE, config.VERSION)
)
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
