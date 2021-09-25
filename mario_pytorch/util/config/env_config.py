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
    EVERY_EPISODE_SAVE: int
    EVERY_STEP_SAVE: int

    INTENTION: str

    @staticmethod
    def create(path: str) -> EnvConfig:
        with open(path, "r") as f:
            return EnvConfig(**yaml.safe_load(f))
