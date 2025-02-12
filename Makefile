SRC_DIR=src
SOURCES=$(shell find $(SRC_DIR) -name "*.py")

PYTHON_VERSION ?= python3.12
OS := $(shell uname -s)

install:
	curl -sSL https://install.python-poetry.org | $(PYTHON_VERSION) -
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry env use $(PYTHON_VERSION)
	poetry lock
	poetry install

develop: install
	poetry run pre-commit install

format: develop
	poetry run pre-commit run --all-files

run_etl:
	poetry run kedro run --pipeline=etl_app

run_ml:
	poetry run kedro run --pipeline=ml_app

viz:
	clear; poetry run kedro viz

mlflow:
	poetry run kedro mlflow ui

test_model:
	clear; poetry run pytest -v tests/test_whole_regression_inference_pipeline.py --disable-warnings
