from mario_pytorch.util.get_env import _get_env_name


def test_single_env_name() -> None:
    world = 1
    stage = 1
    version = 0
    ret = _get_env_name(world, stage, version)
    assert ret == "SuperMarioBros-1-1-v0"


def test_continuous_env_name() -> None:
    world = -1
    stage = -1
    version = 0
    ret = _get_env_name(world, stage, version)
    assert ret == "SuperMarioBros-v0"
