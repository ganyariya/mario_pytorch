from mario_pytorch.learn import learn


def cli_learn(env_config_name: str, reward_scope_config_name: str) -> None:
    learn(env_config_name, reward_scope_config_name)
