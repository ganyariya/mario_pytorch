import random
from typing import *
from pathlib import Path
from collections import deque

import torch
import numpy as np

from gym.wrappers.frame_stack import LazyFrames
from mario_pytorch.agent.mario_net import MarioNet


class Mario:
    def __init__(
        self, state_dim: Tuple[int, int, int], action_dim: int, save_dir: Path
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net: MarioNet = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0  # episode 回数記憶
        self.save_every = 5e5

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # 訓練前に経験させる回数
        self.learn_every = 3  # learn_every ごとに Q_online を学習させる
        self.sync_every = 1e4  #  Q_target & Q_online の同期タイミング

    def act(self, state: LazyFrames) -> int:
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)

            # (4, 84, 84) -> (1, 4, 84, 84)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(
        self,
        state: LazyFrames,
        next_state: LazyFrames,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append(
            (
                state,
                next_state,
                action,
                reward,
                done,
            )
        )

    def recall(self) -> Tuple[torch.Tensor]:
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """Update online action value (Q) function with a batch of experiences.

        Target は固定する
        Online を学習する
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        # 過去のスタックから self.batch_size (32) だけ持ってくる
        # 1 つに 4 frame 入っているはず
        state, next_state, action, reward, done = self.recall()

        # TD Estimate -- Online (学習)
        td_est = self.td_estimate(state, action)

        # TD Target -- Target (固定)
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state: torch.Tensor, action: int) -> torch.Tensor:
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a) # shape torch.Size([32])
        return current_Q

    # target については backpropagete しない
    # 学習させるのは online
    @torch.no_grad()
    def td_target(
        self, reward: float, next_state: torch.Tensor, done: bool
    ) -> torch.Tensor:
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]  # Q_target(s', a') # shape torch.Size([32])
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target) -> float:
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()
        self.optimizer.step()  # 重みの更新
        return loss.item()

    def sync_Q_target(self) -> None:
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self) -> None:
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
