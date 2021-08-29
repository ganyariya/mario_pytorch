from typing import Callable, Any

import torch
from torch.nn import Module

REWARD = torch.Tensor([[0, 0, 0, 0, 0]])


def export_onnx(
    model: Module, input: torch.Tensor, transform: Callable, file_name="rnn.onnx"
) -> None:
    input = transform(input)
    r = REWARD.repeat((input.shape[0], 1))
    k = (input, r)
    # torch.onnx.export(model, k, file_name)


def transform_mario_input(input: torch.Tensor) -> torch.Tensor:
    input = input.__array__()
    input = torch.tensor(input)
    input = input.unsqueeze(0)
    return input
