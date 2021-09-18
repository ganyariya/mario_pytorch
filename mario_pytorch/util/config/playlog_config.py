from __future__ import annotations

import yaml
from pydantic import BaseModel


class PlayLogScope(BaseModel):
    MIN: int = 0
    MAX: int = 0
    BIN: int = 20
    USE: bool = False


class PlayLogScopeConfig(BaseModel):
    X_POS: PlayLogScope
    X_ABS: PlayLogScope
    X_PLUS: PlayLogScope
    X_MINUS: PlayLogScope
    COINS: PlayLogScope
    LIFE: PlayLogScope
    LIFE_PLUS: PlayLogScope
    LIFE_MINUS: PlayLogScope
    GOAL: PlayLogScope
    ITEM: PlayLogScope
    ITEM_PLUS: PlayLogScope
    ITEM_MINUS: PlayLogScope
    ELAPSED: PlayLogScope
    SCORE: PlayLogScope
    KILLS: PlayLogScope

    @staticmethod
    def create(path: str) -> PlayLogScopeConfig:
        with open(path, "r") as f:
            return PlayLogScopeConfig(**yaml.safe_load(f))

    @staticmethod
    def take_out_use(
        config: PlayLogScopeConfig,
    ) -> tuple[list[tuple[int, int]], list[int], list[str]]:
        ranges, bins, keys = [], [], []
        for k, v in config:
            if v.USE:
                ranges.append((v.MIN, v.MAX))
                bins.append(v.BIN)
                keys.append(k)
        return ranges, bins, keys
