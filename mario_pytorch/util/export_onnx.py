"""
https://github.com/pytorch/pytorch/issues/22488#issuecomment-630140460
http://kaga100man.com/2019/03/25/post-102/
https://dajiro.com/entry/2020/06/27/160255
https://teratail.com/questions/277420
"""
from typing import Callable

import torch
from torch.nn import Module

REWARD = torch.Tensor([[0, 0, 0, 0, 0]])


def export_onnx(
    model: Module, input: torch.Tensor, transform: Callable, file_name="rnn.onnx"
) -> None:
    input = transform(input)
    torch.onnx.export(
        model,
        (input, REWARD.repeat((input.shape[0], 1))),
        file_name,
    )


def transform_mario_input(input: torch.Tensor) -> torch.Tensor:
    input = input.__array__()
    input = torch.tensor(input)
    input = input.unsqueeze(0)
    return input
