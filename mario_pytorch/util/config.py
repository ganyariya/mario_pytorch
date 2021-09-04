from __future__ import annotations

import yaml
from pydantic import BaseModel


class EnvConfig(BaseModel):
    WORLD: int
    STAGE: int
    VERSION: int

    SHAPE: int
    SKIP_FRAME: int
    NUM_STACK: int

    IS_RENDER: bool
    EPISODES: int
    EVERY_RECORD: int
    EVERY_RENDER: int

    INTENTION: str

    @staticmethod
    def create(path: str) -> EnvConfig:
        with open(path, "r") as f:
            return EnvConfig(**yaml.safe_load(f))


class Scope(BaseModel):
    MIN: int
    MAX: int


class RewardConfig(BaseModel):
    POSITION: int
    ENEMY: int
    COIN: int
    GOAL: int
    LIFE: int
    ITEM: int
    TIME: int
    SCORE: int


class RewardScopeConfig(BaseModel):
    POSITION: Scope
    ENEMY: Scope
    COIN: Scope
    GOAL: Scope
    LIFE: Scope
    ITEM: Scope
    TIME: Scope
    SCORE: Scope

    @staticmethod
    def create(path: str) -> EnvConfig:
        with open(path, "r") as f:
            return RewardScopeConfig(**yaml.safe_load(f))
