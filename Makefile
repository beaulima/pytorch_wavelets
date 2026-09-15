# Makefile for pytorch_wavelets
#
# Everything runs inside a conda/mamba environment so that the pytorch build
# and the pure python package stay consistent. Override any of the variables
# below on the command line, e.g.
#
#     make dev ENV_NAME=pw-cuda PYTHON_VERSION=3.12
#
# Run `make help` for the list of targets.

ENV_NAME       ?= pytorch_wavelets
PYTHON_VERSION ?= 3.11
ENV_FILE       ?= environment.yml

# Prefer mamba (much faster solver) and fall back to conda.
SOLVER    ?= $(shell command -v mamba 2>/dev/null || command -v conda 2>/dev/null)
# `conda run` itself is only shipped by conda, even when mamba does the solving.
CONDA     ?= $(shell command -v conda 2>/dev/null || command -v mamba 2>/dev/null)

# Run a command inside the environment without needing `conda activate`,
# which does not work from a non-interactive make recipe.
RUN = $(CONDA) run --no-capture-output -n $(ENV_NAME)

CACHE_DIR      = .make
RENDERED_ENV   = $(CACHE_DIR)/environment.$(ENV_NAME).yml
# Stamp files record that the environment was solved / the package installed,
# so `make test` does not re-run a full conda solve on every invocation. They
# are invalidated when environment.yml or pyproject.toml actually changes.
ENV_STAMP      = $(CACHE_DIR)/env.$(ENV_NAME).stamp
INSTALL_STAMP  = $(CACHE_DIR)/install.$(ENV_NAME).stamp

.DEFAULT_GOAL := help
.PHONY: help env env-update install dev test test-cov test-slow test-all lint notebooks check-notebooks docs build clean clean-build clean-env

help: ## Show this help
	@echo "pytorch_wavelets - available targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: ENV_NAME=$(ENV_NAME) PYTHON_VERSION=$(PYTHON_VERSION)"

# Render environment.yml with the requested name and python version so that
# ENV_NAME / PYTHON_VERSION overrides actually take effect.
$(RENDERED_ENV): $(ENV_FILE) | $(CACHE_DIR)
	sed -e 's/^name: .*/name: $(ENV_NAME)/' \
	    -e 's/^\( *- *\)python=.*/\1python=$(PYTHON_VERSION)/' \
	    $(ENV_FILE) > $@

$(CACHE_DIR):
	mkdir -p $(CACHE_DIR)

env: $(ENV_STAMP) ## Create the conda environment (or update it if it exists)

$(ENV_STAMP): $(RENDERED_ENV)
	@if [ -z "$(SOLVER)" ]; then \
		echo "ERROR: neither mamba nor conda was found on PATH."; \
		echo "Install miniforge: https://github.com/conda-forge/miniforge"; \
		exit 1; \
	fi
	@echo "Using solver: $(SOLVER)"
	@if $(CONDA) env list | awk '{print $$1}' | grep -qx '$(ENV_NAME)'; then \
		echo "Environment '$(ENV_NAME)' exists - updating."; \
		$(SOLVER) env update -n $(ENV_NAME) -f $(RENDERED_ENV) --prune --yes; \
	else \
		echo "Creating environment '$(ENV_NAME)'."; \
		$(SOLVER) env create -n $(ENV_NAME) -f $(RENDERED_ENV) --yes; \
	fi
	@touch $@

env-update: ## Force a re-solve of the environment even if nothing changed
	@rm -f $(ENV_STAMP)
	@$(MAKE) env

# --no-deps: environment.yml owns the dependencies, so pip must not pull a
# second copy of torch from PyPI on top of the conda one.
install: $(INSTALL_STAMP) ## Install pytorch_wavelets into the environment (editable)

$(INSTALL_STAMP): pyproject.toml $(ENV_STAMP)
	$(RUN) python -m pip install --no-deps --no-build-isolation -e .
	@touch $@

dev: install ## Editable install + smoke check that the package imports
	$(RUN) python -c "import pytorch_wavelets as p; print('pytorch_wavelets', p.__version__, 'OK')"

test: install ## Run the test suite
	$(RUN) python -m pytest

test-cov: install ## Run the test suite with a coverage report
	$(RUN) python -m pytest --cov=pytorch_wavelets --cov-report=term-missing

test-slow: install ## Run only the slow gradchecks (minutes, not seconds)
	$(RUN) python -m pytest -m slow

test-all: install ## Run every test, slow ones included
	$(RUN) python -m pytest -m ''

lint: env ## Run flake8 over the package
	$(RUN) python -m flake8 pytorch_wavelets tests examples

notebooks: install ## Regenerate notebooks/ from the examples/ scripts
	$(RUN) python tools/make_notebooks.py

check-notebooks: install ## Fail if notebooks/ has drifted from examples/
	$(RUN) python tools/make_notebooks.py --check

docs: install ## Build the html documentation
	$(RUN) sphinx-build -b html docs docs/_build/html
	@echo "Docs written to docs/_build/html/index.html"

build: env ## Build the sdist and wheel into dist/
	# Clear dist/ first: leaving artefacts from an older version behind means a
	# later `twine upload dist/*` would try to publish both.
	rm -rf dist
	$(RUN) python -m build
	$(RUN) python -m twine check dist/*

clean: clean-build ## Remove build artefacts and caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache .coverage htmlcov docs/_build

clean-build:
	rm -rf build dist *.egg-info $(CACHE_DIR)

clean-env: ## Delete the conda environment entirely
	$(CONDA) env remove -n $(ENV_NAME) --yes
	@rm -f $(ENV_STAMP) $(INSTALL_STAMP)
