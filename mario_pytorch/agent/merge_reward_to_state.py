import numpy as np
import torch


def merge_reward_to_state(
    state: torch.Tensor, reward_weights: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """stateの個数に合わせて報酬関数をつける.

    state = (sample, 4, 84, 84)
    r = (1, len(reward_weights))

    Notes
    -----
    state.shape[0] は 画像の枚数（サンプル数）になっている
    そのため，この関数はその枚数分の Tensor (shape[0], len(reward_weights)) を返す
    """
    r = torch.Tensor([reward_weights]).repeat((state.shape[0], 1))
    return state, r
