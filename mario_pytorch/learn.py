"""
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""

import datetime
import time
from os import path
from pathlib import Path

import yaml

from mario_pytorch.agent.mario import Mario
from mario_pytorch.metric_logger.metric_logger import MetricLogger
from mario_pytorch.util.config import EnvConfig, RewardScopeConfig, RewardConfig
from mario_pytorch.util.export_onnx import export_onnx, transform_mario_input
from mario_pytorch.util.get_env import get_env


def tmp_create_reward_config() -> RewardConfig:
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


# ----------------------------------------------------------------------

config_path = Path(__file__).parents[1] / "config" / "env" / "base.yaml"
config = EnvConfig.create(str(config_path))

# reward_scope_config_path = Path(__file__).parents[1] / "config" / "reward" / "base.yaml"
# reward_scope_config = RewardScopeConfig.create(str(reward_scope_config_path))
reward_config = tmp_create_reward_config()

save_dir = (
    Path(path.dirname(__file__)).parent
    / "checkpoints"
    / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
)
save_dir.mkdir(parents=True)
with open(save_dir / "used_config.yaml", "w") as f:
    yaml.safe_dump(config.dict(), f, encoding="utf-8", allow_unicode=True)

env = get_env(config, reward_config)

logger = MetricLogger(save_dir)
mario = Mario(
    state_dim=(config.NUM_STACK, config.SHAPE, config.SHAPE),
    action_dim=env.action_space.n,
    save_dir=save_dir,
)
export_onnx(mario.online_net, env.reset(), transform_mario_input, save_dir)

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

        print(reward)
        time.sleep(0.1)

        logger.log_step(reward, loss, q)

        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()
    if e % config.EVERY_RECORD == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
