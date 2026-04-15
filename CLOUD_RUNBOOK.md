# Cloud runbook for v5.2

## Core rule
Use the same script on every provider:
- `kpmg_tax_rag_v52_aws.py`

Only the environment setup changes.

## Fastest order of work before the AWS teaching session
1. Prepare SageMaker AI Studio as the primary AWS path.
2. Use EC2 only if you want more control or if Studio setup is blocked.
3. Treat Lambda Cloud as optional extra comparison, not required for the class.
4. Run only `run_smoketest.sh` before the session.
5. Save the full 50-question evaluation for after the session.

## Path 1: SageMaker AI Studio (primary)
1. Create the SageMaker domain with Quick setup.
2. Launch Studio.
3. Create a private JupyterLab space with GPU-backed compute for the real run.
4. Upload two zip files:
   - this portable bundle
   - the original `kpmg_tax_rag_outputs_v52_corporate_50q...zip`
5. Open a terminal in JupyterLab and run:

```bash
cd ~
mkdir -p e222-tax-rag-v52
cd e222-tax-rag-v52
unzip -o ~/kpmg_tax_rag_v52_cloud_portable_bundle.zip
cd kpmg_tax_rag_v52_cloud_portable
chmod +x bootstrap_sagemaker_studio.sh run_smoketest.sh run_full_eval.sh
./bootstrap_sagemaker_studio.sh "$PWD"
source .venv/bin/activate 2>/dev/null || true
export ARTIFACT_ZIP=~/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip
./run_smoketest.sh
```

6. If smoke test works, later run:

```bash
./run_full_eval.sh
```

## Path 2: EC2 (fallback or more control)
1. Launch a GPU-backed Ubuntu instance.
2. SSH into the instance.
3. Copy the same two zip files onto the box.
4. Run:

```bash
mkdir -p ~/e222-tax-rag-v52
cd ~/e222-tax-rag-v52
unzip -o ~/kpmg_tax_rag_v52_cloud_portable_bundle.zip
cd kpmg_tax_rag_v52_cloud_portable
chmod +x bootstrap_ec2.sh run_smoketest.sh run_full_eval.sh
./bootstrap_ec2.sh "$PWD"
source .venv/bin/activate
export ARTIFACT_ZIP=~/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip
./run_smoketest.sh
```

5. For the full evaluation:

```bash
./run_full_eval.sh
```

## Path 3: Lambda Cloud / Lambda Labs (optional)
1. Add an SSH key to your Lambda Cloud account.
2. Launch an instance.
3. Open JupyterLab from the Lambda console, or SSH into the instance.
4. Upload the same two zip files.
5. In a terminal, run:

```bash
mkdir -p ~/e222-tax-rag-v52
cd ~/e222-tax-rag-v52
unzip -o ~/kpmg_tax_rag_v52_cloud_portable_bundle.zip
cd kpmg_tax_rag_v52_cloud_portable
chmod +x bootstrap_lambda_cloud.sh run_smoketest.sh run_full_eval.sh
./bootstrap_lambda_cloud.sh "$PWD"
source .venv/bin/activate
export ARTIFACT_ZIP=~/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip
./run_smoketest.sh
```

6. Only run the full evaluation if the smoke test is clean.

## Notebook path
After the smoke test works, you can open:
- `kpmg_tax_rag_v52_aws.ipynb`

and run notebook cells in the same environment.

## Suggested provider choice
- Primary: SageMaker AI Studio
- Fallback: EC2
- Optional learning comparison: Lambda Cloud
