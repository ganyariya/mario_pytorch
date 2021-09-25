from typer import Typer

from mario_pytorch.cli_components.cli_learn import cli_learn
from mario_pytorch.cli_components.cli_learn_pyribs import cli_learn_pyribs
from mario_pytorch.cli_components.cli_relearn_pyribs import cli_relearn_pyribs
from mario_pytorch.cli_components.cli_play import cli_play

app: Typer = Typer()


@app.command()
def learn(env_config_name: str, reward_scope_config_name: str) -> None:
    cli_learn(env_config_name, reward_scope_config_name)


@app.command()
def learn_pyribs(
    env_config_name: str, reward_scope_config_name: str, playlog_scope_config_name: str
) -> None:
    cli_learn_pyribs(
        env_config_name, reward_scope_config_name, playlog_scope_config_name
    )


@app.command()
def relearn_pyribs(
    env_config_name: str,
    reward_scope_config_name: str,
    playlog_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
) -> None:
    cli_relearn_pyribs(
        env_config_name,
        reward_scope_config_name,
        playlog_scope_config_name,
        date_str,
        checkpoint_idx,
    )


@app.command()
def play(
    env_config_name: str,
    reward_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
    exploration_rate: float,
) -> None:
    cli_play(
        env_config_name,
        reward_scope_config_name,
        date_str,
        checkpoint_idx,
        exploration_rate,
    )


if __name__ == "__main__":
    app()
