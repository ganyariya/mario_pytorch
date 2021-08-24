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

# ----------------------------------------------------------------------

WORLD: int = 1
STAGE: int = 1
VERSION: int = 3

USE_CUDA = torch.cuda.is_available()
IS_RENDER = True
EPISODES = 10
EVERY_RECORD = 20
EVERY_RENDER = 20

save_dir = (
    Path(path.dirname(__file__)).parent
    / "checkpoints"
    / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
)
save_dir.mkdir(parents=True)

# 4 Consecutive GrayScale Frames Set [4, 84, 84]
env = gym_super_mario_bros.make(get_env_name(WORLD, STAGE, VERSION))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)  # 4Frame まとめて取り出す

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

for e in range(EPISODES):

    # state.shape (4, 84, 84)
    # state.frame_shape (84, 84)
    state = env.reset()

    while True:
        action = mario.act(state)
        if IS_RENDER and e % EVERY_RENDER == 0:
            env.render()

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()
    if e % EVERY_RECORD == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
