#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found. Install Python 3.12 or set PYTHON_BIN." >&2
  exit 1
fi

if [ ! -d "venv" ]; then
  "$PYTHON_BIN" -m venv venv
fi

./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -e ".[dev]"

cat <<'EOF'
Development environment bootstrapped.

Next:
  source venv/bin/activate
  make check
EOF
