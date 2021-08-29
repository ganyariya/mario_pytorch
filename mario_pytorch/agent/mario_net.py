from typing import Tuple
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
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x) -> nn.Module:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
