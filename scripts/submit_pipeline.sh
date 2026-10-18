#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_CODE:?Set authorized NTU project code}"
: "${JEPA_CONDA_SH:?Set Conda shell path}"
: "${JEPA_CONDA_ENV:?Set Conda environment}"
: "${JEPA_MODEL_PATH:?Set the local LLaDA model snapshot path}"
: "${JEPA_ARTIFACT_ROOT:?Set writable storage for generated artifacts}"

if [[ "$PROJECT_CODE" == "YOUR_AUTHORIZED_PROJECT_CODE" ]]; then
    echo "ERROR: Replace PROJECT_CODE placeholder first"
    exit 1
fi

JEPA_REPO_DIR="${JEPA_REPO_DIR:-$(pwd)}"
JEPA_HF_HOME="${JEPA_HF_HOME:-$JEPA_ARTIFACT_ROOT/huggingface}"
mkdir -p "$JEPA_ARTIFACT_ROOT/logs"

export_vars="JEPA_CONDA_SH=${JEPA_CONDA_SH},JEPA_CONDA_ENV=${JEPA_CONDA_ENV},JEPA_MODEL_PATH=${JEPA_MODEL_PATH},JEPA_ARTIFACT_ROOT=${JEPA_ARTIFACT_ROOT},JEPA_REPO_DIR=${JEPA_REPO_DIR},JEPA_HF_HOME=${JEPA_HF_HOME}"

job1=$(qsub -P "$PROJECT_CODE" -v "$export_vars" pbs/01_generate_trajectories.pbs)
echo "Trajectory job: $job1"

job2=$(qsub -P "$PROJECT_CODE" -v "$export_vars" -W "depend=afterok:${job1}" pbs/02_train_jepa.pbs)
echo "Training job: $job2"

job3=$(qsub -P "$PROJECT_CODE" -v "$export_vars" -W "depend=afterok:${job2}" pbs/03_evaluate.pbs)
echo "Evaluation job: $job3"

echo "Pipeline submitted successfully."
