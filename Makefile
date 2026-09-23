# =============================================================================
# Qwen3-VL-4B Benchmark Makefile
# =============================================================================

.PHONY: help env build dataset inspect smoke benchmark accuracy report validate clean

PYTHON ?= python3
PROJECT_DIR ?= .
SERVER_URL ?= http://127.0.0.1:8080

help: ## Show this help
	@echo "Qwen3-VL-4B Benchmark Suite"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

env: ## Collect environment information
	bash scripts/collect_environment.sh

build: ## Build llama.cpp with CUDA
	bash scripts/build_llamacpp.sh

inspect: ## Inspect model artifacts and build compatibility matrix
	$(PYTHON) scripts/inspect_model.py --project-dir $(PROJECT_DIR)

dataset: ## Prepare benchmark dataset
	$(PYTHON) scripts/prepare_dataset.py --project-dir $(PROJECT_DIR)

smoke: ## Run smoke test (5 requests CCU1 + 2 CCU2)
	bash run_benchmark.sh --smoke

benchmark: ## Run full benchmark (all phases)
	bash run_benchmark.sh

accuracy: ## Run accuracy benchmark only
	bash run_benchmark.sh --phase 9

validate: ## Validate benchmark results
	$(PYTHON) scripts/validate_results.py --project-dir $(PROJECT_DIR)

report: ## Generate summary report and charts
	$(PYTHON) scripts/summarize.py --project-dir $(PROJECT_DIR)

clean: ## Clean generated results (keep models and data)
	rm -rf results/raw results/gpu results/server_metrics results/report results/charts
	rm -f results/config_snapshot.yaml results/validation_report.json

clean-all: ## Clean everything including dataset
	rm -rf results benchmark_data/performance benchmark_data/manifest.*

status: ## Show benchmark status
	@echo "=== Model Files ==="
	@find models -name "*.gguf" 2>/dev/null || echo "  No models found"
	@echo ""
	@echo "=== Dataset ==="
	@if [ -f benchmark_data/manifest.jsonl ]; then \
		echo "  Manifest: benchmark_data/manifest.jsonl"; \
		wc -l benchmark_data/manifest.jsonl; \
	else \
		echo "  Not prepared. Run: make dataset"; \
	fi
	@echo ""
	@echo "=== Results ==="
	@if [ -f results/raw/requests.csv ]; then \
		echo "  Raw results:"; \
		wc -l results/raw/requests.csv; \
	else \
		echo "  No results yet. Run: make benchmark"; \
	fi
