# Repository audit

Audit date: 2026-10-18.

## Scope inspected

The audit covered all Python modules under `src/jepa_pu`, four YAML configurations, twelve PBS scripts, three active submission wrappers, historical phase notes, raw scheduler output present in the working directory, and every compact JSON/Markdown result under `research_artifacts`.

## Findings and disposition

| Finding | Disposition |
|---|---|
| Root README described only Phase 1 and implied future work | Replaced with an archived-project overview and honest final conclusions |
| Results were split across terse phase summaries and operator notes | Consolidated in `docs/experiments.md`; original compact summaries retained |
| Active YAML/PBS/scripts embedded one user's home/scratch paths | Replaced with `JEPA_*` environment variables and documented defaults |
| Raw PBS stdout files were at repository root | Preserved locally but ignored by Git via `*.o[0-9]*`/log rules |
| `jepa-results-for-docs.tar.gz` duplicates `research_artifacts` | Preserved locally but ignored as an archive |
| `scripts/submit_pipeline.sh.bak` duplicates the active wrapper | Preserved locally but ignored as a backup |
| `src/jepa_pu/decode.py.before_phase31` is a historical implementation snapshot | Deliberately retained and documented; it preserves pre-Phase-3.1 behavior |
| Phase-specific root READMEs contain obsolete deployment details and personal cluster identifiers | Preserved locally but excluded from publication; superseded by `docs/` |
| No test suite existed | Added lightweight CPU-only unit tests |
| No exact environment lockfile, checkpoints, raw trajectories, or prediction files were present | Documented as reproducibility limitations; not reconstructed |
| No repository license or citation metadata was present | Left unresolved for owner review rather than inventing ownership/reuse terms |

## Source organization

- `data.py`: deterministic GSM8K preparation and numeric extraction.
- `llada.py`, `features.py`, `models.py`: frozen backbone wrapper, action/target representations, and learned heads.
- `trajectories.py`, `train.py`, `decode.py`, `evaluate.py`: Phase 1/2 generation, training, decoding, and evaluation.
- `ranking_audit.py`: Phase 2 proxy-ranking analysis.
- `phase3_candidates.py`, `phase3_rollouts.py`, `phase3_trajectories.py`, `phase3_rank_train.py`, `phase3_outcome_audit.py`: Phase 3 implementation.
- `phase31_audit.py`, `phase31_evaluate.py`: Phase 3.1 ablations.
- `phase4_selective_evaluate.py`: Phase 4 gate calibration and selective evaluation.

There is some intentional duplication among evaluation/audit entry points because they encode historical phases. Consolidating them would make the archive smaller but risk changing research behavior, so they were left intact.

## Publication boundary

Tracked publication artifacts should include source, configs, PBS/scripts, documentation, tests, and compact summaries/audits. Large or sensitive artifacts are ignored: model weights, checkpoints, trajectories, prepared datasets, prediction-level outputs, Hugging Face caches, raw logs, archives, credentials, and editor/build temporaries.
