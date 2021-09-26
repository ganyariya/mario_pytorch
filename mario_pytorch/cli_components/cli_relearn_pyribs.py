from mario_pytorch.relearn_pyribs import relearn_pyribs


def cli_relearn_pyribs(
    env_config_name: str,
    reward_scope_config_name: str,
    playlog_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
) -> None:
    relearn_pyribs(
        env_config_name,
        reward_scope_config_name,
        playlog_scope_config_name,
        date_str,
        checkpoint_idx,
    )
