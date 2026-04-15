#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${1:-$HOME/e222-tax-rag-v52}"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_cloud_portable.txt
mkdir -p outputs input
printf "SageMaker Studio bootstrap complete in %s\n" "$BASE_DIR"
