from __future__ import annotations
from typing import Dict

from pydantic import BaseModel


class InfoModel(BaseModel):
    x_pos: int
    coins: int
    life: int
    flag_get: bool
    status: str
    time: int
    score: int
    kills: int

    @staticmethod
    def create(info: Dict) -> InfoModel:
        info["x_pos"] = info["x_pos"].item()
        info["life"] = info["life"].item()
        return InfoModel(**info)
