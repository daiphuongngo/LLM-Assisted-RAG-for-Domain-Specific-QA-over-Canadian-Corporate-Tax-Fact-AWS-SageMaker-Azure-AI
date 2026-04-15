#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${1:-$HOME/e222-tax-rag-v52}"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip unzip
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_cloud_portable.txt
mkdir -p outputs input
printf "EC2 bootstrap complete in %s
" "$BASE_DIR"
