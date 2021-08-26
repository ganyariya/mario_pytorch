from __future__ import annotations

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Config"""

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
    def create(path: str) -> Config:
        with open(path, "r") as f:
            return Config(**yaml.safe_load(f))
