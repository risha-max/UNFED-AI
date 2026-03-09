# Contributing to UNFED AI

Thanks for contributing.

## Quick Setup

```bash
git clone https://github.com/risha-max/UNFED-AI.git
cd UNFED-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Validation

Run focused checks before opening a PR:

```bash
make check
```

If you have a running local stack, include runtime readiness:

```bash
make check-runtime
```

## Public Testnet Helpers

The repository includes env-driven operational scripts:

```bash
# Launch services + run Go/No-Go checks
make public-testnet-start

# Re-run checks only
make public-testnet-check

# Inspect running services
make public-testnet-status

# Stop services started by the launcher
make public-testnet-stop
```

Strict profile:

```bash
cp .env.testnet.strict.example .env.testnet.strict
make public-testnet-start-strict
```

## Commit Guidelines

- Keep changes scoped to one problem.
- Update docs for any workflow/config changes.
- Avoid committing runtime artifacts from `.runtime/` or local secrets.
