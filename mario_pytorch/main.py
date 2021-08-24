"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
https://github.com/YuansongFeng/MadMario/blob/master/agent.py
"""

import datetime

from pathlib import Path

import torch
import gym_super_mario_bros

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from mario_pytorch.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger

# ----------------------------------------------------------------------

# 4 Consecutive GrayScale Frames Set
# [4, 84, 84]
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)


use_cuda = torch.cuda.is_available()
is_render = True

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):
    state = env.reset()
    while True:
        action = mario.act(state)
        if is_render:
            env.render()
        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()
    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
