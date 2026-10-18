# Reproducibility guide

## What is and is not archived

This repository contains source code, phase configurations, scheduler scripts, and compact result summaries. It does not contain LLaDA weights, prepared GSM8K rows, trajectories, checkpoints, prediction-level JSONL, Hugging Face caches, or raw scheduler logs. A full rerun therefore requires recreating data splits and all learned checkpoints.

The code loads a local LLaDA snapshot with `trust_remote_code: true`. The recorded runs pointed to the Hugging Face snapshot revision `852152c60bdfed4632635bfcf3e83408ad3f65bf` of `GSAI-ML/LLaDA-8B-Instruct`. Obtain it under its upstream terms, review its remote code, and set `JEPA_MODEL_PATH` to that pinned local snapshot. Model files are not included here.

## Dependencies

- Python >= 3.10
- PyTorch >= 2.1 with a CUDA build suitable for the target GPU
- Transformers 4.38.2
- Datasets >= 2.18
- PyYAML >= 6.0
- NumPy >= 1.24
- tqdm >= 4.66
- A PBS Pro/OpenPBS environment for the supplied cluster scripts

`pyproject.toml` and `requirements.txt` contain the available constraints. No exact lockfile, Python patch version, CUDA version, driver version, GPU model manifest, or package freeze was retained, so bitwise reproduction is not promised.

## Path configuration

YAML strings support `${NAME}` and `${NAME:-default}` expansion. Set:

```bash
export JEPA_MODEL_PATH=/absolute/path/to/pinned/LLaDA-8B-Instruct/snapshot
export JEPA_ARTIFACT_ROOT=/absolute/path/to/writable/jepa-artifacts
```

The artifact root will contain `data/`, phase-specific `trajectories*`, `checkpoints*`, `results*`, `logs/`, and optional caches. Defaults under `artifacts/` are useful for inspecting configuration but model runs should use explicit absolute paths.

For PBS submission also set:

```bash
export PROJECT_CODE=your_authorized_project
export JEPA_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export JEPA_CONDA_ENV=/path/to/or/name/of/environment
export JEPA_REPO_DIR="$PWD"
# Optional; defaults to $JEPA_ARTIFACT_ROOT/huggingface:
export JEPA_HF_HOME=/path/to/huggingface/cache
```

The submission wrappers pass repository, model, artifact, and Conda variables to each job. Queue/module/resource directives remain cluster-specific and may need local edits.

## Seeds and deterministic choices

- Global experiment/data seed: `17` in every saved config.
- Test shuffle seed: `18` (`seed + 1`) in `jepa_pu.data`.
- Frozen Gaussian projection seed: `314159`.
- Temperature: `0.0`; rollouts are single deterministic greedy continuations.
- Ranking order: deterministic shuffle with `seed + epoch`.

`seed_everything` initializes Python, NumPy, PyTorch, and all visible CUDA generators. Exact GPU reproducibility may still vary because deterministic CUDA algorithms are not globally enforced and the complete historical software/hardware environment is unavailable.

## Phase execution order and outputs

All commands below load the 8B model and are GPU workloads unless noted otherwise.

### Phase 1

```bash
python -m jepa_pu.data --config configs/phase1.yaml
python -m jepa_pu.trajectories --config configs/phase1.yaml --split train
python -m jepa_pu.trajectories --config configs/phase1.yaml --split validation
python -m jepa_pu.train --config configs/phase1.yaml
python -m jepa_pu.evaluate --config configs/phase1.yaml
```

Expected paths: `data/gsm8k_{train,validation,test}.jsonl`, `trajectories/`, `checkpoints/phase1.pt`, and `results/{summary.json,summary.md,predictions-*.jsonl}` under `$JEPA_ARTIFACT_ROOT`. `scripts/submit_pipeline.sh` submits the equivalent three-job PBS chain.

### Phase 2a

Reuse Phase 1 data, trajectories, and checkpoint:

```bash
python -m jepa_pu.ranking_audit --config configs/phase2a.yaml
python -m jepa_pu.evaluate --config configs/phase2a.yaml
```

Expected path: `results_phase2a/`. PBS entry point: `pbs/04_phase2a_evaluate.pbs`.

### Phase 2b

```bash
python -m jepa_pu.trajectories --config configs/phase2b.yaml --split train
python -m jepa_pu.trajectories --config configs/phase2b.yaml --split validation
python -m jepa_pu.train --config configs/phase2b.yaml
python -m jepa_pu.ranking_audit --config configs/phase2b.yaml
python -m jepa_pu.evaluate --config configs/phase2b.yaml
```

Expected paths: `trajectories_phase2/`, `checkpoints_phase2/`, and `results_phase2/`. Submission wrapper: `scripts/submit_phase2b.sh`.

### Phase 3

```bash
python -m jepa_pu.phase3_trajectories --config configs/phase3.yaml --split train
python -m jepa_pu.phase3_trajectories --config configs/phase3.yaml --split validation
python -m jepa_pu.train --config configs/phase3.yaml
python -m jepa_pu.phase3_rank_train --config configs/phase3.yaml
python -m jepa_pu.phase3_outcome_audit --config configs/phase3.yaml
python -m jepa_pu.evaluate --config configs/phase3.yaml
```

Expected paths: `trajectories_phase3/`, `checkpoints_phase3/phase1-before-ranking.pt`, `checkpoints_phase3/phase1.pt`, and `results_phase3/`. Submission wrapper: `scripts/submit_phase3.sh`.

### Phase 3.1

Reuse both Phase 3 checkpoints and trajectories; do not retrain:

```bash
python -m jepa_pu.phase31_audit --config configs/phase3.yaml
python -m jepa_pu.phase31_evaluate --config configs/phase3.yaml --budget 32 --examples 250
```

Expected path: `results_phase31/`. PBS entry point: `pbs/11_phase31_ablation.pbs`.

### Phase 4

Reuse the Phase 3 pre-ranking checkpoint and saved Phase 3.1 baseline:

```bash
python -m jepa_pu.phase4_selective_evaluate --config configs/phase3.yaml
```

Expected path: `results_phase4/`. PBS entry point: `pbs/12_phase4_selective.pbs`.

## Safe validation without model weights

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
bash -n scripts/*.sh pbs/*.pbs
```

These checks validate syntax, configuration expansion, answer extraction, and small CPU tensor utilities. They do not validate remote model code, CUDA compatibility, data availability, checkpoint compatibility, numerical equivalence, or end-to-end results.

## Operational cautions

- Data and trajectory writers refuse to overwrite existing artifacts unless explicitly forced. Inspect interrupted shards before recovery.
- Do not publish generated GSM8K rows or prediction JSONL without reviewing dataset terms and potential content/privacy concerns.
- Do not commit model snapshots, `.pt`/`.bin`/`.safetensors` files, caches, credentials, or scheduler logs.
- The stored result summaries are the reference values for this archive; reruns on a different stack may not reproduce latency or exact outputs.
