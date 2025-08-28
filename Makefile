# Makefile for Deep Learning HPC DEMO
# This file contains common commands for development, testing, and deployment

# Variables
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
HELM := helm
PROJECT_NAME := deep-learning-hpc-demo
IMAGE_NAME := $(PROJECT_NAME):latest

# Default target
.PHONY: help
help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Development targets
.PHONY: install
install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install development dependencies
	$(PIP) install -r requirements-dev.txt

.PHONY: format
format: ## Format code with black
	black src tests

.PHONY: lint
lint: ## Lint code with flake8
	flake8 src tests

.PHONY: type-check
type-check: ## Type check with mypy
	mypy --package src

.PHONY: test
test: ## Run tests
	pytest tests/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html

# Docker targets
.PHONY: build
build: ## Build Docker image
	$(DOCKER) build -t $(IMAGE_NAME) .

.PHONY: build-no-cache
build-no-cache: ## Build Docker image without cache
	$(DOCKER) build --no-cache -t $(IMAGE_NAME) .

.PHONY: run
run: ## Run Docker container
	$(DOCKER) run --rm -it --gpus all -p 8000:8000 -p 8080:8080 $(IMAGE_NAME)

.PHONY: up
up: ## Start services with docker-compose
	$(DOCKER_COMPOSE) up -d

.PHONY: down
down: ## Stop services with docker-compose
	$(DOCKER_COMPOSE) down

.PHONY: logs
logs: ## View logs from docker-compose services
	$(DOCKER_COMPOSE) logs -f

.PHONY: ps
ps: ## List running services
	$(DOCKER_COMPOSE) ps

# Kubernetes targets
.PHONY: kube-deploy
kube-deploy: ## Deploy to Kubernetes
	$(HELM) dependency update ./helm
	$(HELM) upgrade --install $(PROJECT_NAME) ./helm \
		--namespace $(PROJECT_NAME) \
		--create-namespace \
		--set image.repository=$(IMAGE_NAME) \
		--timeout 10m \
		--wait

.PHONY: kube-delete
kube-delete: ## Delete deployment from Kubernetes
	$(HELM) uninstall $(PROJECT_NAME) --namespace $(PROJECT_NAME)

.PHONY: kube-status
kube-status: ## Check deployment status
	$(KUBECTL) get all -n $(PROJECT_NAME)

.PHONY: kube-logs
kube-logs: ## View pod logs
	$(KUBECTL) logs -n $(PROJECT_NAME) -l app=$(PROJECT_NAME) --tail=100 -f

# Data management targets
.PHONY: download-data
download-data: ## Download sample dataset
	$(PYTHON) src/utils/dataset.py --download-sample

.PHONY: preprocess-data
preprocess-data: ## Preprocess dataset
	$(PYTHON) src/utils/dataset.py --preprocess

# Model management targets
.PHONY: train-flax
train-flax: ## Train model with Flax
	$(PYTHON) src/training/train_flax.py

.PHONY: train-torch
train-torch: ## Train model with PyTorch
	$(PYTHON) src/training/train_torch.py

.PHONY: serve
serve: ## Start model serving
	$(PYTHON) src/deployment/serve_ray.py

# Documentation targets
.PHONY: docs
docs: ## Generate documentation
	cd docs && make html

.PHONY: docs-clean
docs-clean: ## Clean documentation build
	cd docs && make clean

# Security targets
.PHONY: security-check
security-check: ## Run security checks
	bandit -r src
	safety check -r requirements.txt

# Performance targets
.PHONY: perf-test
perf-test: ## Run performance tests
	locust -f tests/performance/locustfile.py

# Cleanup targets
.PHONY: clean
clean: ## Clean build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml

.PHONY: clean-all
clean-all: clean ## Clean all artifacts including docs and builds
	cd docs && make clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Release targets
.PHONY: release-patch
release-patch: ## Create a patch release
	bump2version patch

.PHONY: release-minor
release-minor: ## Create a minor release
	bump2version minor

.PHONY: release-major
release-major: ## Create a major release
	bump2version major

# Utility targets
.PHONY: shell
shell: ## Open a shell in the Docker container
	$(DOCKER) run --rm -it --gpus all -v $(PWD):/app $(IMAGE_NAME) /bin/bash

.PHONY: jupyter
jupyter: ## Start Jupyter notebook
	$(DOCKER) run --rm -it --gpus all -p 8888:8888 -v $(PWD):/app $(IMAGE_NAME) jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

.PHONY: ray-dashboard
ray-dashboard: ## Open Ray dashboard
	@echo "Open http://localhost:8265 in your browser"

.PHONY: grafana
grafana: ## Open Grafana dashboard
	@echo "Open http://localhost:3000 in your browser (default user/pass: admin/admin)"

.PHONY: mlflow
mlflow: ## Open MLflow dashboard
	@echo "Open http://localhost:5000 in your browser"