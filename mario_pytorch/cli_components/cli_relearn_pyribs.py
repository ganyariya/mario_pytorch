from mario_pytorch.learn_pyribs import learn_pyribs


def cli_relearn_pyribs(
    env_config_name: str,
    reward_scope_config_name: str,
    playlog_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
) -> None:
    # learn_pyribs(env_config_name, reward_scope_config_name, playlog_scope_config_name)
    print(20)
