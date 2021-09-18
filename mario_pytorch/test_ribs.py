import matplotlib.pyplot as plt
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

# 行動空間
archive = GridArchive(dims=([20, 20]), ranges=[(-1, 1), (-1, 1)])

# パラメータ空間
emitters = [ImprovementEmitter(archive, [0.0] * 8, 0.1, bounds=[(-0.5, 0.25)] * 8)]

optimizer = Optimizer(archive, emitters)


def simulate(solutions: np.ndarray):
    return -np.sum(np.square(solutions), axis=1)


for _ in range(1000):
    solutions = optimizer.ask()
    # (10, 8)
    # 10個数分のパラメータ空間の解を持ってくる
    # 各パラメータの長さは8
    print(solutions.shape)

    # 最大化させる（最小化したいので，マイナスをつけて最大化させている）
    # 複数解を与えて，それらの目的関数値を返す
    objectives = simulate(solutions)
    # (10, )
    print(objectives.shape)

    # bcs は特徴量
    bcs = solutions[:, :2]

    optimizer.tell(objectives, bcs)

grid_archive_heatmap(archive)
plt.show()
