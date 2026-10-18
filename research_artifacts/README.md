# Research artifacts

This directory contains the small result files used to write the public experiment record. They were copied from completed runs; raw predictions, trajectories, checkpoints, datasets, and logs are intentionally excluded.

| Directory | Contents |
|---|---|
| `results/` | Phase 1 summaries |
| `results_phase2a/` | Phase 2a summaries and original-proxy ranking audit |
| `results_phase2/` | Phase 2b summaries and corrected-proxy ranking audit |
| `results_phase3/` | Phase 3 summaries and rollout-outcome audit |
| `results_phase31/` | Phase 3.1 summary, paired comparisons, and mixed-outcome audit |
| `results_phase4/` | Phase 4 calibration and final selective-policy summary |

`results_32step.csv` is a publication-oriented index derived directly from those JSON files. The phase JSON files remain the source of truth. Latencies are hardware/environment dependent; configured budgets are maximum steps rather than matched realized compute.

![Phase 3 and Phase 4 accuracy versus realized LLaDA passes](phase3_phase4_accuracy_vs_passes.svg)

The plot shows recorded operating points only. It does not represent a compute-matched frontier.
