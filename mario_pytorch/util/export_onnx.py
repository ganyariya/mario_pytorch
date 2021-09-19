"""
https://github.com/pytorch/pytorch/issues/22488#issuecomment-630140460
http://kaga100man.com/2019/03/25/post-102/
https://dajiro.com/entry/2020/06/27/160255
https://teratail.com/questions/277420
"""
from pathlib import Path
from typing import Callable

import torch
import numpy as np
from torch.nn import Module

from mario_pytorch.agent.merge_reward_to_state import merge_reward_to_state as f

REWARD = torch.Tensor([[0, 0, 0, 0, 0]])


def export_onnx(
    model: Module,
    input_state: torch.Tensor,
    input_reward: np.ndarray,
    transform: Callable,
    save_dir: Path,
    file_name="rnn.onnx",
) -> None:
    input_state = transform(input_state)
    torch.onnx.export(
        model,
        f(input_state, input_reward),
        str(save_dir / file_name),
    )


def transform_mario_input(input: torch.Tensor) -> torch.Tensor:
    input = input.__array__()
    input = torch.tensor(input)
    input = input.unsqueeze(0)
    return input
