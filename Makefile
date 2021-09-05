env_config_name = base.yaml
reward_scope_config_name = base.yaml

date_str=""
checkpoint_idx=-1

.PHONY: learn
learn:
	python -m mario_pytorch.cli learn ${env_config_name} ${reward_scope_config_name}

.PHONY: play
play:
	python -m mario_pytorch.cli play ${env_config_name} ${reward_scope_config_name} ${date_str} ${checkpoint_idx}