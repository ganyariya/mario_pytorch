from __future__ import annotations

import yaml
from pydantic import BaseModel


class RewardConfig(BaseModel):
    POSITION: int
    ENEMY: int
    COIN: int
    GOAL: int
    LIFE: int
    ITEM: int
    TIME: int
    SCORE: int

    @staticmethod
    def create(path: str) -> RewardConfig:
        with open(path, "r") as f:
            return RewardConfig(**yaml.safe_load(f))


class RewardScope(BaseModel):
    MIN: int = -100
    MAX: int = 100
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
