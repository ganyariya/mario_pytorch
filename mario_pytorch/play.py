"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
https://github.com/YuansongFeng/MadMario/blob/master/agent.py
"""

from pathlib import Path

import torch

from mario_pytorch.agent.mario import LearnedMario
from mario_pytorch.util.config import EnvConfig
from mario_pytorch.util.get_env import get_env

# ----------------------------------------------------------------------

config_path = Path(__file__).parents[1] / "config" / "base.yaml"
config = EnvConfig.create(str(config_path))

checkpoint_name = "2021-08-29T15-32-18"
checkpoint = Path(__file__).parents[1] / "checkpoints" / checkpoint_name
model_name = "mario_net_10.chkpt"
model_path = checkpoint / model_name
model = torch.load(model_path)["model"]

env = get_env(config)

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
