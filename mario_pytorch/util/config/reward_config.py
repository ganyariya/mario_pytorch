from __future__ import annotations

import numpy as np
import yaml
from pydantic import BaseModel


class RewardConfig(BaseModel):
    POSITION: float
    ENEMY: float
    COIN: float
    GOAL: float
    LIFE: float
    ITEM: float
    TIME: float
    SCORE: float

    @staticmethod
    def create(path: str) -> RewardConfig:
        with open(path, "r") as f:
            return RewardConfig(**yaml.safe_load(f))

    @staticmethod
    def init() -> RewardConfig:
        return RewardConfig(
            **{
                "POSITION": 0,
                "ENEMY": 0,
                "COIN": 0,
                "GOAL": 0,
                "LIFE": 0,
                "ITEM": 0,
                "TIME": 0,
                "SCORE": 0,
            }
        )

    @staticmethod
    def init_with_keys(parameter: np.ndarray, keys: list[str]) -> RewardConfig:
        ret = RewardConfig.init()
        for i in range(len(parameter)):
            setattr(ret, keys[i], parameter[i])
        return ret


class RewardScope(BaseModel):
    MIN: float = -100
    MAX: float = 100
    USE: bool = False


class RewardScopeConfig(BaseModel):
    POSITION: RewardScope
    ENEMY: RewardScope
    COIN: RewardScope
    GOAL: RewardScope
    LIFE: RewardScope
    ITEM: RewardScope
    TIME: RewardScope
    SCORE: RewardScope

    @staticmethod
    def create(path: str) -> RewardScopeConfig:
        with open(path, "r") as f:
            return RewardScopeConfig(**yaml.safe_load(f))

    @staticmethod
    def take_out_use(
        config: RewardScopeConfig,
    ) -> tuple[list[tuple[int, int]], list[str]]:
        bounds, keys = [], []
        for k, v in config:
            if v.USE:
                bounds.append((v.MIN, v.MAX))
                keys.append(k)
        return bounds, keys
