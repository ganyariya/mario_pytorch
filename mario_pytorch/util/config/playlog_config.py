from __future__ import annotations

from pydantic import BaseModel


class PlayLogScope(BaseModel):
    MIN: int
    MAX: int
    BIN: int
    USE: bool = False


class PlayLogConfig(BaseModel):
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
