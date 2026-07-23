#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-serl}"
PYTHON_VERSION="3.10"

# Pick a faster solver if available
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi

echo "[1/4] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
$SOLVER create -y -n "${ENV_NAME}" -c conda-forge \
  python="${PYTHON_VERSION}" \
  pip \
  git

echo "[2/4] Upgrading pip tooling + installing uv..."
conda run -n "${ENV_NAME}" python -m pip install -U pip setuptools wheel uv

echo "[3/4] Installing project + dependencies into conda env (editable), honoring tool.uv.sources..."
# IMPORTANT: run this script from repo root so -e . works
conda run -n "${ENV_NAME}" uv pip install -e .

echo "[4/4] Done."
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Sanity check:   python -c \"import sys; print(sys.executable); import jax; print(jax.devices())\""
