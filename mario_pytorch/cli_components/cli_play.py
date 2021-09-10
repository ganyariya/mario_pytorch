from mario_pytorch.play import play
from mario_pytorch.util.decorators import (
    argument_print_decorator,
    time_measurement_decorator,
)


@time_measurement_decorator
@argument_print_decorator
def cli_play(
    env_config_name: str,
    reward_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
    exploration_rate: float,
) -> None:
    play(
        env_config_name,
        reward_scope_config_name,
        date_str,
        checkpoint_idx,
        exploration_rate,
    )
