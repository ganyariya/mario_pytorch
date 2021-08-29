from typing import Tuple

import torch
from torch import nn

HW_SIZE = 84


class MarioNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int):
        super().__init__()
        c, h, w = input_dim

        if h != HW_SIZE:
            raise ValueError(f"Expecting input height: {HW_SIZE}, got: {h}")
        if w != HW_SIZE:
            raise ValueError(f"Expecting input width: {HW_SIZE}, got: {w}")

        # https://www.koi.mashykom.com/deep_learning.html

        # Image
        self.image_conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4
        )
        self.image_relu1 = nn.ReLU()
        self.image_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        self.image_relu2 = nn.ReLU()
        self.image_conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )
        self.image_relu3 = nn.ReLU()
        self.image_flatten = nn.Flatten()

        # Reward
        self.reward_fc = nn.Linear(5, 20)
        self.reward_relu = nn.ReLU()

        # Merge
        self.merge_fc1 = nn.Linear(3136 + 20, 512)
        self.merge_relu = nn.ReLU()
        self.merge_fc2 = nn.Linear(512, output_dim)

    def forward(self, x, y) -> nn.Module:
        # x image (32, 4, 84, 84) (32 batch_size, 4つの白黒Frameをまとめている, 縦, 横)
        # y reward
        x = self.image_conv1(x)
        x = self.image_relu1(x)
        x = self.image_conv2(x)
        x = self.image_relu2(x)
        x = self.image_conv3(x)
        x = self.image_relu3(x)
        x = self.image_flatten(x)

        y = self.reward_fc(y)
        y = self.reward_relu(y)

        # x (32, 3136)  y (32, 20)
        # z (32, 3136 + 20)
        z = torch.cat((x, y), 1)
        z = self.merge_fc1(z)
        z = self.merge_relu(z)
        z = self.merge_fc2(z)
        return z
