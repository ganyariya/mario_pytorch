import datetime
from pathlib import Path
from typing import Optional

import yaml

from mario_pytorch.util.config import EnvConfig, PlayLogScopeConfig, RewardScopeConfig


def get_env_config_path(env_config_name: str) -> Path:
    return Path(__file__).parents[2] / "config" / "env" / env_config_name


def get_reward_scope_config_path(reward_scope_config_name: str) -> Path:
    return Path(__file__).parents[2] / "config" / "reward" / reward_scope_config_name


def get_playlog_scope_config_path(playlog_scope_config_name: str) -> Path:
    return Path(__file__).parents[2] / "config" / "playlog" / playlog_scope_config_name


def get_results_path() -> Path:
    results_path = Path(__file__).parents[2] / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def get_save_path(results_path: Path) -> Path:
    save_path = results_path / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def get_checkpoint_path(save_path: Path) -> Path:
    checkpoint_path = save_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_reward_models_path(save_path: Path) -> Path:
    checkpoint_path = save_path / "models"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_pickles_path(save_path: Path) -> Path:
    checkpoint_path = save_path / "pickles"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_playlog_path(save_path: Path) -> Path:
    playlog_path = save_path / "playlogs"
    playlog_path.mkdir(parents=True, exist_ok=True)
    return playlog_path


def copy_and_backup_env_files(
    save_path: Path,
    env_config: EnvConfig,
    reward_scope_config: RewardScopeConfig,
    playlog_scope_config: PlayLogScopeConfig,
) -> None:
    with open(save_path / "env_config.yaml", "w") as f:
        yaml.safe_dump(env_config.dict(), f, encoding="utf-8", allow_unicode=True)
    with open(save_path / "reward_scope_config.yaml", "w") as f:
        yaml.safe_dump(
            reward_scope_config.dict(), f, encoding="utf-8", allow_unicode=True
        )
    with open(save_path / "playlog_scope_config.yaml", "w") as f:
        yaml.safe_dump(
            playlog_scope_config.dict(), f, encoding="utf-8", allow_unicode=True
        )


def generate_README_file(save_path: Path) -> None:
    with open(save_path / "README.md", "w") as f:
        f.write("# Why")


def get_date_path(results_path: Path, date_str: str) -> Path:
    return results_path / date_str


def get_model_path(date_path: Path, checkpoint_idx: int, prefix: Optional[str]) -> Path:
    if prefix is None:
        return date_path / f"mario_net_{checkpoint_idx}.chkpt"
    return date_path / f"{prefix}_mario_net_{checkpoint_idx}.chkpt"
