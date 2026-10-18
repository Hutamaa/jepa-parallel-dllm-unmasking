#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_CODE:?Set authorized PBS project code}"
: "${JEPA_CONDA_SH:?Set readable path to conda.sh}"
: "${JEPA_CONDA_ENV:?Set compatible pre-existing Conda env}"
: "${JEPA_MODEL_PATH:?Set the local LLaDA model snapshot path}"
: "${JEPA_ARTIFACT_ROOT:?Set writable storage for generated artifacts}"
if [[ "$PROJECT_CODE" == *YOUR* || "$PROJECT_CODE" == *PLACEHOLDER* ]]; then
    echo "Refusing placeholder project code" >&2
    exit 1
fi
JEPA_REPO_DIR="${JEPA_REPO_DIR:-$(pwd)}"
JEPA_HF_HOME="${JEPA_HF_HOME:-$JEPA_ARTIFACT_ROOT/huggingface}"
cd "$JEPA_REPO_DIR"
VARS="JEPA_CONDA_SH=${JEPA_CONDA_SH},JEPA_CONDA_ENV=${JEPA_CONDA_ENV},JEPA_MODEL_PATH=${JEPA_MODEL_PATH},JEPA_ARTIFACT_ROOT=${JEPA_ARTIFACT_ROOT},JEPA_REPO_DIR=${JEPA_REPO_DIR},JEPA_HF_HOME=${JEPA_HF_HOME}"
job1=$(qsub -P "$PROJECT_CODE" -v "$VARS" pbs/08_phase3_generate.pbs)
echo "Phase3 generation: $job1"
job2=$(qsub -P "$PROJECT_CODE" -v "$VARS" -W "depend=afterok:${job1}" pbs/09_phase3_train.pbs)
echo "Phase3 training: $job2"
job3=$(qsub -P "$PROJECT_CODE" -v "$VARS" -W "depend=afterok:${job2}" pbs/10_phase3_evaluate.pbs)
echo "Phase3 evaluation: $job3"
