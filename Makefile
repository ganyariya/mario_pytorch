config_name = base.yaml

date_str=""
checkpoint_idx=-1
exploration_rate=0.9

.PHONY: learn
learn: ## config_name
	python -m mario_pytorch.cli learn ${config_name} ${config_name}

.PHONY: learn_pyribs
learn_pyribs: ## config_name
	python -m mario_pytorch.cli learn-pyribs ${config_name} ${config_name} ${config_name}

.PHONY: relearn_pyribs
relearn_pyribs: ## config_name
	python -m mario_pytorch.cli relearn-pyribs ${config_name} ${config_name} ${config_name} ${date_str} ${checkpoint_idx}

.PHONY: play
play: ## config_name date_str checkpoint_idx
	python -m mario_pytorch.cli play ${config_name} ${config_name} ${date_str} ${checkpoint_idx} 

.PHONY: lint
lint: ##
	poetry run pflake8 mario_pytorch
	poetry run black --check --diff .

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'