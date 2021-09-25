from logging import getLogger
from pathlib import Path
from mario_pytorch.agent.mario import BaseMario

import torch

logger = getLogger(__name__)


def save_episode_model(
    mario: BaseMario, checkpoint_path: Path, episode: int, step: int, save_every: int
) -> None:
    save_checkpoint_path = (
        checkpoint_path / f"episode_mario_net_{int(episode // save_every)}.chkpt"
    )
    torch.save(
        dict(
            model=mario.online_net.state_dict(),
            exploration_rate=mario.exploration_rate,
            episode=episode,
            step=step,
        ),
        str(save_checkpoint_path),
    )
    logger.info(f"Net to {checkpoint_path} at episode / step  {episode} / {step}")


def save_step_model(
    mario: BaseMario, checkpoint_path: Path, episode: int, step: int, save_every: int
) -> None:
    save_checkpoint_path = (
        checkpoint_path / f"step_mario_net_{int(step // save_every)}.chkpt"
    )
    torch.save(
        dict(
            model=mario.online_net.state_dict(),
            exploration_rate=mario.exploration_rate,
            episode=episode,
            step=step,
        ),
        str(save_checkpoint_path),
    )
    logger.info(f"Net to {checkpoint_path} at episode / step  {episode} / {step}")
