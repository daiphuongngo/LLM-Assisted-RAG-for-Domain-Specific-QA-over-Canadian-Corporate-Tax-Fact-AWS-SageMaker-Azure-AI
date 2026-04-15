# Portable cloud run package for v5.2

This package keeps **one core script** for all providers:
- `kpmg_tax_rag_v52_aws.py`

It adds only thin wrappers for:
- EC2
- SageMaker AI Studio
- Lambda Cloud / Lambda Labs

That design matches the TA guidance: treat cloud providers mostly as alternative places to run the same scripts, and avoid splitting the pipeline unless there is a strong reason.

## Included files
- `kpmg_tax_rag_v52_aws.py` — core portable script
- `kpmg_tax_rag_v52_aws.ipynb` — notebook version
- `reconstructed_eval_50q.csv` — evaluation CSV reconstructed from the observed v5.2 outputs
- `requirements_cloud_portable.txt` — common package list
- `sample.env` — common environment template
- `bootstrap_ec2.sh` — EC2 setup
- `bootstrap_sagemaker_studio.sh` — SageMaker AI Studio setup
- `bootstrap_lambda_cloud.sh` — Lambda Cloud / Lambda Labs setup
- `run_smoketest.sh` — common first-run check (preflight + one ask)
- `run_full_eval.sh` — common 50-question evaluation command
- `CLOUD_RUNBOOK.md` — step-by-step instructions for EC2, SageMaker AI Studio, and Lambda Cloud
- `QUESTIONS_FOR_SESSION.md` — focused questions to ask in the AWS teaching session

## Philosophy
Use the same project files and commands everywhere:
1. unpack the portable bundle
2. install dependencies
3. point `ARTIFACT_ZIP` at your original v5.2 artifact zip
4. run `run_smoketest.sh`
5. only then run `run_full_eval.sh`

## Core commands
These are the same on every provider once the environment is ready.

```bash
python kpmg_tax_rag_v52_aws.py --base-dir "$PWD" --bundle-zip "$ARTIFACT_ZIP" preflight
python kpmg_tax_rag_v52_aws.py --base-dir "$PWD" --bundle-zip "$ARTIFACT_ZIP" ask "What is the 2025 combined tax rate for a CCPC in Quebec on small business income?"
python kpmg_tax_rag_v52_aws.py --base-dir "$PWD" --bundle-zip "$ARTIFACT_ZIP" evaluate reconstructed_eval_50q.csv --output-csv outputs/re_eval.csv --summary-json outputs/re_eval_summary.json
```
