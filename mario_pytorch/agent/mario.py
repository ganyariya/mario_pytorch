import random
from collections import OrderedDict, deque
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from gym.wrappers.frame_stack import LazyFrames

from mario_pytorch.agent.mario_net import MarioNet
from mario_pytorch.agent.merge_reward_to_state import merge_reward_to_state as frts

logger = getLogger(__name__)


class BaseMario:
    def __init__(
        self, state_dim: Tuple[int, int, int], action_dim: int, reward_dim: int
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_cuda = torch.cuda.is_available()

        self.online_net: MarioNet = MarioNet(
            self.state_dim, self.action_dim, reward_dim
        ).float()
        if self.use_cuda:
            self.online_net = self.online_net.to(device="cuda")
        logger.info(self.online_net)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0  # Frame 回数記憶 (エピソードではなく フレーム)

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 10000  # 訓練前に経験させるFrame回数
        self.learn_every = 3  # learn_every Frame ごとに Q_online を学習させる
        self.sync_every = 1e4  # Q_target & Q_online の同期タイミング

    def act(self, state: LazyFrames, reward_weights: np.ndarray) -> int:
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame):
            A single observation of the current state, dimension is (state_dim)
        Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()  # LazyFrames -> ndarray
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)

            state = state.unsqueeze(0)  # (4, 84, 84) -> (1, 4, 84, 84)
            action_values = self.online_net(*frts(state, reward_weights))
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx


class Mario(BaseMario):
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        action_dim: int,
        reward_dim: int,
    ):
        super().__init__(state_dim, action_dim, reward_dim)
        self.target_net = deepcopy(self.online_net)
        self._sync_Q_target()

    def cache(
        self,
        state: LazyFrames,
        next_state: LazyFrames,
        action: int,
        reward: float,
        done: bool,
        reward_weights: np.ndarray,
    ) -> None:
        """
        Store the experience to self.memory (replay buffer)

        Notes
        -----
        memory に入る変数の shape
        - state (4, 84, 84)
        - action (1)  reward (1)  done (1)
        - weights (1, 2)
        """
        state = state.__array__()
        next_state = next_state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
            reward_weights = torch.tensor([reward_weights]).float().cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
            reward_weights = torch.tensor([reward_weights]).float()

        self.memory.append(
            (
                state,
                next_state,
                action,
                reward,
                done,
                reward_weights,
            )
        )

    def recall(self) -> Tuple[torch.Tensor]:
        """
        Retrieve a batch of experiences from memory

        Notes
        -----
        state: (32, 4, 84, 84) torch
        action: (32, 1) torch
        action.squeeze(): (32) torch
        weights: (32, 1, 2) torch
        weights.squeeze(): (32, 2) torch
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done, weights = map(torch.stack, zip(*batch))
        return (
            state,
            next_state,
            action.squeeze(),
            reward.squeeze(),
            done.squeeze(),
            weights.squeeze(),
        )

    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """Update online action value (Q) function with a batch of experiences.

        Target は固定する
        Online を学習する
        """

        if self.curr_step % self.sync_every == 0:
            self._sync_Q_target()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        # 過去のスタックから self.batch_size (32) だけ持ってくる
        # 1 つに 4 frame 入っているはず
        state, next_state, action, reward, done, weights = self.recall()

        # TD Estimate -- Online (学習)
        td_est = self._td_estimate(state, action, weights)

        # TD Target -- Target (固定)
        td_tgt = self._td_target(reward, next_state, done, weights)

        # https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/4_RL/4_2_mario_rl_tutorial_jp.ipynb#scrollTo=hjDCD1o3PKHX
        # Backpropagate loss through Q_online
        loss = self._update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def _td_estimate(
        self, state: torch.Tensor, action: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """推定する.

        Notes
        -----
        r: (batch_size, len(reward_weights))
        shape: (batch_size, 4, 84, 84)
        current_Q[np=s, action=a] = [32] (取り出している)
        """
        current_Q = self.online_net(state, weights)[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def _td_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        DQN
        https://qiita.com/ishizakiiii/items/5eff79b59bce74fdca0d#q-learning

        Q(s, a) <- Q(s, a) + \alpha ( R(s, a) +
            gamma * (argmax_{a'}Qt(s', a')) - Qo(s, a))
        Qt=target（固定）  Qo=online (学習ネットワーク)

        R(s, a) + rQt(s', a') = 実際に行動して得られた値 C
        Qo(s, a) = エージェントの現在の予測値であり，これがCに近づくように修正する

        ----------
        DDQN
        https://www.renom.jp/ja/notebooks/product/renom_rl/ddqn/notebook.html
        https://blog.syundo.org/post/20171208-reinforcement-learning-dqn-and-impl/
        https://gyazo.com/365ac89c3f956f4a7bbb35359e9e18a3

        Notes
        -----
        torch.no_grad で勾配計算を無効にしている
        """
        next_state_Q = self.online_net(next_state, weights)
        best_action: int = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target_net(next_state, weights)[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def _update_Q_online(
        self, td_estimate: torch.Tensor, td_target: torch.Tensor
    ) -> float:
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()
        self.optimizer.step()  # 重みの更新
        return loss.item()

    def _sync_Q_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False


class ReLearnMario(Mario):
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        action_dim: int,
        reward_dim: int,
        model: OrderedDict,
        exploration_rate: float,
        step: int,
    ):
        super().__init__(state_dim, action_dim, reward_dim)
        self.online_net.load_state_dict(model)
        self._sync_Q_target()
        self.exploration_rate = exploration_rate
        self.curr_step = step


class LearnedMario(BaseMario):
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        action_dim: int,
        reward_dim: int,
        model: OrderedDict,
        exploration_rate: float,
    ):
        super().__init__(state_dim, action_dim, reward_dim)
        self.online_net.load_state_dict(model)
        self.exploration_rate = exploration_rate
