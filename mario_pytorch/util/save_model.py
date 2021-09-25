from logging import getLogger
from pathlib import Path
from mario_pytorch.agent.mario import BaseMario

import torch

logger = getLogger(__name__)


def save_episode_model(
    mario: BaseMario, checkpoint_path: Path, episode: int, save_every: int
) -> None:
    save_checkpoint_path = (
        checkpoint_path / f"mario_net_{int(episode // save_every)}.chkpt"
    )
    torch.save(
        dict(
            model=mario.online_net.state_dict(),
            exploration_rate=mario.exploration_rate,
            episode=episode,
        ),
        str(save_checkpoint_path),
    )
    logger.info(f"MarioNet saved to {checkpoint_path} at episode {episode}")


def save_step_model(
    mario: BaseMario, checkpoint_path: Path, step: int, save_every: int
) -> None:
    save_checkpoint_path = (
        checkpoint_path / f"mario_net_step_{int(step // save_every)}.chkpt"
    )
    torch.save(
        dict(
            model=mario.online_net.state_dict(),
            exploration_rate=mario.exploration_rate,
            step=step,
        ),
        str(save_checkpoint_path),
    )
    logger.info(f"MarioNet saved to {checkpoint_path} at step {step}")
