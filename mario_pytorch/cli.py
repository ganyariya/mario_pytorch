from typer import Typer

from mario_pytorch.cli_components.cli_learn import cli_learn

app: Typer = Typer()


@app.command()
def learn(env_config_name: str, reward_scope_config_name: str):
    cli_learn(env_config_name, reward_scope_config_name)


@app.command()
def play():
    pass


if __name__ == "__main__":
    app()
