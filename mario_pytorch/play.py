"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
https://github.com/YuansongFeng/MadMario/blob/master/agent.py
"""

from pathlib import Path

import torch
import gym_super_mario_bros

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from mario_pytorch.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from mario_pytorch.agent.mario import LearnedMario
from mario_pytorch.util.get_env_name import get_env_name
from mario_pytorch.util.config import Config

# ----------------------------------------------------------------------

config_path = Path(__file__).parents[1] / "config" / "base.yaml"
config = Config.create(str(config_path))

checkpoint_name = "2021-08-29T15-32-18"
checkpoint = Path(__file__).parents[1] / "checkpoints" / checkpoint_name
model_name = "mario_net_10.chkpt"
model_path = checkpoint / model_name
model = torch.load(model_path)["model"]

# 4 Consecutive GrayScale Frames Set [4, 84, 84]
env = gym_super_mario_bros.make(
    get_env_name(config.WORLD, config.STAGE, config.VERSION)
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=config.SKIP_FRAME)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=config.SHAPE)
env = FrameStack(env, num_stack=config.NUM_STACK)  # 4Frame まとめて取り出す

mario = LearnedMario(
    state_dim=(config.NUM_STACK, config.SHAPE, config.SHAPE),
    action_dim=env.action_space.n,
    model=model,
)

for e in range(config.EPISODES):

    # state.shape (4, 84, 84)  state.frame_shape (84, 84)
    state = env.reset()
    reward_sum = 0

    while True:
        action = mario.act(state)
        env.render()

        next_state, reward, done, info = env.step(action)
        reward_sum += reward

        state = next_state

        if done or info["flag_get"]:
            break

    print(reward_sum)
