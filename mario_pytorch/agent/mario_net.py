import copy
from torch import nn

HW_SIZE = 84


class MarioNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != HW_SIZE:
            raise ValueError(f"Expecting input height: {HW_SIZE}, got: {h}")
        if w != HW_SIZE:
            raise ValueError(f"Expecting input width: {HW_SIZE}, got: {w}")

        # https://www.koi.mashykom.com/deep_learning.html
        # チャンネル数を増やして 画像サイズを小さくする
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model: str) -> nn.Module:
        if model == "online":
            return self.online(input)
        if model == "target":
            return self.target(input)
