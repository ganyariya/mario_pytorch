from mario_pytorch.play import play


def cli_play(
    env_config_name: str,
    reward_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
) -> None:
    play(env_config_name, reward_scope_config_name, date_str, checkpoint_idx)
