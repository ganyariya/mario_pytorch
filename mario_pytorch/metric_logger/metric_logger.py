import datetime
import time
from logging import (
    INFO,
    WARNING,
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    getLogger,
)
from pathlib import Path
from typing import Final, List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def _set_logger(save_dir: Path) -> None:
    logger = getLogger()
    getLogger("matplotlib").setLevel(WARNING)

    formatter = Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s\n%(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = FileHandler(save_dir / "logger_log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(INFO)


class MetricLogger:
    def __init__(self, save_path: Path) -> None:
        _set_logger(save_path)

        self._writer: Final[SummaryWriter] = SummaryWriter()
        self.save_path: Final[Path] = save_path
        self.image_path: Final[Path] = save_path / "image"
        self.save_metric_log: Final[Path] = save_path / "metric_log"
        self.logger: Final[Logger] = getLogger(__name__)
        self._init_metric_log_file(self.save_metric_log)

        # Images
        self.image_path.mkdir(parents=True, exist_ok=True)
        self.ep_rewards_plot = self.image_path / "reward_plot.jpg"
        self.ep_lengths_plot = self.image_path / "length_plot.jpg"
        self.ep_avg_losses_plot = self.image_path / "loss_plot.jpg"
        self.ep_avg_qs_plot = self.image_path / "q_plot.jpg"

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

        # TensorBoard
        self._writer.add_scalar("Episode/MeanReward", mean_ep_reward, episode)
        self._writer.add_scalar("Episode/MeanLoss", mean_ep_loss, episode)
        self._writer.add_scalar("Episode/MeanQ", mean_ep_q, episode)
        self._writer.add_scalar("Episode/MeanLength", mean_ep_length, episode)
        self._writer.add_scalar("Episode/Exploration", epsilon, episode)

        self.logger.info(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
        )

        self._append_write_metric_log_file(
            self.save_metric_log,
            episode,
            step,
            epsilon,
            mean_ep_reward,
            mean_ep_length,
            mean_ep_loss,
            mean_ep_q,
            time_since_last_record,
        )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

    def _init_episode(self) -> None:
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def _init_metric_log_file(self, log_path: Path) -> None:
        with open(log_path, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>10}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

    def _append_write_metric_log_file(
        self,
        log_path: Path,
        episode: int,
        step: int,
        epsilon: float,
        mean_ep_reward: float,
        mean_ep_length: float,
        mean_ep_loss: float,
        mean_ep_q: float,
        time_since_last_record: float,
    ) -> None:
        with open(log_path, "a") as f:
            f.write(
                f"{episode:8d}{step:10d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
