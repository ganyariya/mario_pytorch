config_name = base.yaml

date_str=""
checkpoint_idx=-1

.PHONY: learn
learn: ## config_name
	python -m mario_pytorch.cli learn ${config_name} ${config_name}

.PHONY: play
play: ## config_name date_str checkpoint_idx
	python -m mario_pytorch.cli play ${config_name} ${config_name} ${date_str} ${checkpoint_idx}

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'