config_name = base.yaml

date_str=""
checkpoint_idx=-1

.PHONY: learn
learn:
	python -m mario_pytorch.cli learn ${config_name} ${config_name}

.PHONY: play
play:
	python -m mario_pytorch.cli play ${config_name} ${config_name} ${date_str} ${checkpoint_idx}