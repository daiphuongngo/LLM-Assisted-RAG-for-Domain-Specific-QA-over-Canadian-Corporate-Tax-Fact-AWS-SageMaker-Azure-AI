#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${BASE_DIR:-$PWD}"
ARTIFACT_ZIP="${ARTIFACT_ZIP:-}"
QUESTION="${QUESTION:-What is the 2025 combined tax rate for a CCPC in Quebec on small business income?}"

if [[ -z "$ARTIFACT_ZIP" ]]; then
  echo "Set ARTIFACT_ZIP to the original v5.2 artifact zip path." >&2
  exit 1
fi

python kpmg_tax_rag_v52_aws.py --base-dir "$BASE_DIR" --bundle-zip "$ARTIFACT_ZIP" preflight
python kpmg_tax_rag_v52_aws.py --base-dir "$BASE_DIR" --bundle-zip "$ARTIFACT_ZIP" ask "$QUESTION"
