import time
import datetime

from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from mario_pytorch.agent.mario import BaseMario

writer = SummaryWriter()


class MetricLogger:
    def __init__(self, save_dir: Path, base_mario: BaseMario) -> None:
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # 1エピソードのみ
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0  # 1エピソードに含まれたロス計算回数

        # スタートからゴール = 1 エピソード
        # 1エピソードごとのログをリストで保存する（Nエピソード）
        self.ep_rewards: List[float] = []
        self.ep_lengths: List[float] = []
        self.ep_avg_losses: List[float] = []
        self.ep_avg_qs: List[float] = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards: List[float] = []
        self.moving_avg_ep_lengths: List[float] = []
        self.moving_avg_ep_avg_losses: List[float] = []
        self.moving_avg_ep_avg_qs: List[float] = []

        self.base_mario = base_mario

        # Current episode metric
        self._init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward: float, loss: float, q: float) -> None:
        """1フレーム(正確にはnum_tack Frame)ごとに reward loss q を加算する.

        Notes
        -----
        1 Episode = マリオがスタートからゴールするまで
        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self) -> None:
        """1エピソードごとのログを保存する."""
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)

        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self._init_episode()

    def _init_episode(self) -> None:
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode: float, epsilon: float, step: int) -> None:
        """Record というまとまった単位で保存する.

        Notes
        -----
        Record は エポックではない
        おそらく学習がうまく行っているかを見るために，
        マリオevery_record体（every_recordエピソード）ごとに平均を取る

        なぜか 100 の平均とっているけど every_record じゃないのかな
        """
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        writer.add_scalar("Episode/MeanReward", mean_ep_reward, episode)
        writer.add_scalar("Episode/MeanLoss", mean_ep_loss, episode)
        writer.add_scalar("Episode/MeanQ", mean_ep_q, episode)
        writer.add_scalar("Episode/MeanLength", mean_ep_length, episode)
        writer.add_scalar(
            "Episode/Exploration", self.base_mario.exploration_rate, episode
        )

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
