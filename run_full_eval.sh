#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="${BASE_DIR:-$PWD}"
ARTIFACT_ZIP="${ARTIFACT_ZIP:-}"
OUT_CSV="${OUT_CSV:-outputs/re_eval.csv}"
OUT_JSON="${OUT_JSON:-outputs/re_eval_summary.json}"

if [[ -z "$ARTIFACT_ZIP" ]]; then
  echo "Set ARTIFACT_ZIP to the original v5.2 artifact zip path." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"
python kpmg_tax_rag_v52_aws.py --base-dir "$BASE_DIR" --bundle-zip "$ARTIFACT_ZIP" evaluate reconstructed_eval_50q.csv --output-csv "$OUT_CSV" --summary-json "$OUT_JSON"
