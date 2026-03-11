SHELL := /usr/bin/env bash
PYTHON ?= python3
VENV_ACTIVATE := if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

.PHONY: help setup check check-runtime preflight-web preflight-node test test-quick \
	public-testnet-start public-testnet-start-strict public-testnet-check \
	public-testnet-status public-testnet-stop

help:
	@echo "UNFED AI developer tasks"
	@echo ""
	@echo "Setup:"
	@echo "  make setup                       Create venv and install deps"
	@echo ""
	@echo "Validation:"
	@echo "  make test                        Run full pytest suite"
	@echo "  make test-quick                  Run quick pytest smoke checks"
	@echo "  make preflight-web               Run web preflight checks"
	@echo "  make preflight-node              Run node preflight checks"
	@echo "  make check                       test-quick + preflight checks"
	@echo "  make check-runtime               Runtime readiness check (requires running web)"
	@echo ""
	@echo "Public testnet helpers:"
	@echo "  make public-testnet-start        Launch default profile + checklist"
	@echo "  make public-testnet-start-strict Launch strict profile + checklist"
	@echo "  make public-testnet-check        Run checklist only"
	@echo "  make public-testnet-status       Show service status"
	@echo "  make public-testnet-stop         Stop launcher-managed services"

setup:
	$(PYTHON) -m venv .venv
	source .venv/bin/activate && pip install -r requirements.txt

test:
	$(VENV_ACTIVATE) && $(PYTHON) -m pytest -q

test-quick:
	$(VENV_ACTIVATE) && $(PYTHON) -m pytest tests/test_registry_model_health.py tests/test_registry_auth_controls.py -q

preflight-web:
	$(VENV_ACTIVATE) && $(PYTHON) -m scripts.testnet_preflight web --host 127.0.0.1

preflight-node:
	$(VENV_ACTIVATE) && $(PYTHON) -m scripts.testnet_preflight node --advertise 127.0.0.1:50051

check: test-quick preflight-web preflight-node

check-runtime:
	$(VENV_ACTIVATE) && $(PYTHON) -m scripts.testnet_preflight runtime --web-url http://127.0.0.1:8080

public-testnet-start:
	./scripts/start_public_testnet.sh

public-testnet-start-strict:
	./scripts/start_public_testnet.sh --env-file .env.testnet.strict

public-testnet-check:
	./scripts/run_public_testnet_checklist.sh

public-testnet-status:
	./scripts/status_public_testnet.sh

public-testnet-stop:
	./scripts/stop_public_testnet.sh
