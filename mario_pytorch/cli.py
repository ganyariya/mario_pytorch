from typer import Typer

from mario_pytorch.cli_components.cli_learn import cli_learn
from mario_pytorch.cli_components.cli_play import cli_play

app: Typer = Typer()


@app.command()
def learn(env_config_name: str, reward_scope_config_name: str):
    cli_learn(env_config_name, reward_scope_config_name)


@app.command()
def play(
    env_config_name: str,
    reward_scope_config_name: str,
    date_str: str,
    checkpoint_idx: int,
    exploration_rate: float,
):
    cli_play(
        env_config_name,
        reward_scope_config_name,
        date_str,
        checkpoint_idx,
        exploration_rate,
    )


if __name__ == "__main__":
    app()
