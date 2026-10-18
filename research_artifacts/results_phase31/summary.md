# Phase 3.1 — no retraining

| Method | Budget | Accuracy | Steps | LLaDA passes | Latency (s) | Policy overhead (s) | Nonprefix choices | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_confidence | 32 | 0.5240 | 27.44 | 27.44 | 0.703 | 0.007 | 0.00 | 15.41 |
| direct_post_all | 32 | 0.4400 | 24.52 | 24.52 | 0.698 | 0.075 | 11.14 | 15.41 |
| direct_post_prefix | 32 | 0.4520 | 23.16 | 23.16 | 0.640 | 0.051 | 0.00 | 15.41 |
| direct_pre_all | 32 | 0.4240 | 22.95 | 22.95 | 0.655 | 0.071 | 10.24 | 15.41 |
| direct_pre_prefix | 32 | 0.4240 | 22.10 | 22.10 | 0.611 | 0.049 | 0.00 | 15.41 |
| jepa_post_all | 32 | 0.4720 | 24.59 | 24.59 | 0.705 | 0.078 | 11.65 | 15.41 |
| jepa_post_prefix | 32 | 0.4520 | 23.35 | 23.35 | 0.648 | 0.054 | 0.00 | 15.41 |
| jepa_pre_all | 32 | 0.4440 | 23.69 | 23.69 | 0.678 | 0.076 | 10.59 | 15.41 |
| jepa_pre_prefix | 32 | 0.4800 | 22.69 | 22.69 | 0.630 | 0.053 | 0.00 | 15.41 |

## Paired correctness versus adaptive confidence

| Method | Rescued | Lost | Net |
|---|---:|---:|---:|
| jepa_pre_prefix | 14 | 25 | -11 |
| direct_pre_prefix | 10 | 35 | -25 |
| jepa_pre_all | 10 | 30 | -20 |
| direct_pre_all | 11 | 36 | -25 |
| jepa_post_prefix | 7 | 25 | -18 |
| direct_post_prefix | 10 | 28 | -18 |
| jepa_post_all | 23 | 36 | -13 |
| direct_post_all | 14 | 35 | -21 |
