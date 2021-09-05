import datetime
from pathlib import Path

import yaml

from mario_pytorch.util.config import EnvConfig, RewardConfig


def get_env_config_path(env_config_name: str) -> Path:
    return Path(__file__).parents[2] / "config" / "env" / env_config_name


def get_reward_scope_config_path(reward_scope_config_name: str) -> Path:
    return Path(__file__).parents[2] / "config" / "reward" / reward_scope_config_name


def get_results_path() -> Path:
    results_path = Path(__file__).parents[2] / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def get_save_path(results_path: Path) -> Path:
    save_path = results_path / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def copy_and_save_env_files(
    save_path: Path, env_config: EnvConfig, reward_config: RewardConfig
) -> None:
    with open(save_path / "env_config.yaml", "w") as f:
        yaml.safe_dump(env_config.dict(), f, encoding="utf-8", allow_unicode=True)
    with open(save_path / "reward_config.yaml", "w") as f:
        yaml.safe_dump(reward_config.dict(), f, encoding="utf-8", allow_unicode=True)
