# Experiment record

This document is a retrospective account based on the compact outputs under `research_artifacts/`. Percentages below are exact conversions of recorded proportions; no missing runs or uncertainty estimates have been reconstructed.

## Common design

All phases use a frozen LLaDA-8B-Instruct backbone and GSM8K numeric exact match. The saved test summaries contain 250 examples per condition. Candidate actions commit LLaDA's own proposed tokens; gold answers are not inputs to online action selection. JEPA predicts projected post-action hidden representations and a head maps those predictions plus action features to a scalar score. The direct control scores the same action features without JEPA's predicted representation.

The configured step budget is a maximum. Policies can commit larger groups and finish early, so realized LLaDA passes and latency must accompany accuracy.

## Phase 1: representation prediction and initial decoding

**Question.** Can predicted post-action representations help choose the size of a confidence-prefix token group?

**Method.** Candidate groups were confidence-ranked prefixes. Four current/action representations were projected to 256 dimensions. The JEPA predictor targeted projected means for the future full response, remaining masked positions, and committed positions after an immediate counterfactual action. Its scalar target was an instability proxy on positions left masked:

`0.75 × retained-proposal probability loss + 0.25 × argmax flip rate`.

The direct MLP used the same action features but no predicted future representations. Both adaptive policies combined predicted risk with a confidence reward. The terminal action's proxy could become zero, later recognized as a target artifact.

**Recorded 32-step results.** Vanilla achieved 52.0% accuracy at 32.000 passes; JEPA and direct each achieved 50.8%, at 26.704 and 25.556 passes respectively. At 16 steps, all three methods achieved 39.2%. The initial JEPA policy therefore did not improve answer accuracy over vanilla and did not separate from the direct control.

Source: `research_artifacts/results/summary.json`.

## Phase 2: confidence baselines and risk-target corrections

### Phase 2a: stronger baseline against the Phase 1 checkpoint

**Change.** Adaptive confidence was added using the same prefix candidate sizes. It selected the largest prefix whose minimum token confidence was at least 0.90, otherwise the smallest feasible prefix. An offline ranking audit was added, but it still evaluated the Phase 1 instability proxy rather than downstream correctness.

**Recorded 32-step results.** Adaptive confidence achieved 52.8% at 29.520 passes, versus 52.0% at 32.000 for vanilla and 50.8% for both JEPA and direct. The Phase 1 JEPA controller did not beat the stronger heuristic.

The Phase 2a audit reported JEPA proxy top-1 accuracy 76.5%, direct 69.75%, and adaptive confidence 46.0% over 400 validation states. These values describe agreement with the stored proxy only; they are not GSM8K accuracy and did not predict an online advantage.

Sources: `research_artifacts/results_phase2a/summary.json` and `ranking-audit-validation.json`.

### Phase 2b: corrected non-oracle proxy

**Change.** Trajectories and heads were regenerated with `commitment_shift_v2`:

`0.75 × mean selected-token uncertainty + 0.25 × normalized Jensen–Shannon shift on remaining masks`.

Unlike Phase 1, a terminal action retained commitment risk. This remained a teacher-free proxy—not a correctness label.

**Recorded 32-step results.** Adaptive confidence achieved 52.8% at 29.520 passes. JEPA, direct, and vanilla each achieved 52.0%, at 31.796, 31.764, and 32.000 passes respectively. The audit showed strong proxy ranking (JEPA top-1 89.0%; direct 87.0%; adaptive 71.0% over 400 states), but the better proxy fit again did not yield higher downstream accuracy.

Sources: `research_artifacts/results_phase2/summary.json` and `ranking-audit-validation.json`.

## Phase 3: downstream counterfactual rollout supervision

**Question.** Were weak proxy labels and a prefix-only action space hiding a downstream benefit?

**Changes.** Each state considered at most six candidates drawn from confidence prefixes and deterministic alternating-rank non-prefix groups. For training/validation states sampled at generation steps 0, 12, and 24, each candidate was committed and followed by one deterministic vanilla continuation. The label was:

`-correct + 0.10 × continuation passes / remaining available slots`.

Correctness dominated a small compute tie-breaker. After ordinary training, scalar heads received within-state pairwise ranking fine-tuning. At online selection, no additional confidence reward was added.

**Offline audit.** Among 192 validation states, only 38 had different rollout correctness across actions. JEPA and direct each selected the minimum recorded rollout cost in 92.71% of states, while adaptive confidence reached 93.23% on the same action set. JEPA's pairwise outcome concordance was higher than direct's (0.664 versus 0.589), but the label was one off-policy deterministic continuation and did not guarantee online accuracy.

**Recorded online results.** The complete table is:

| Maximum steps | Adaptive confidence | JEPA | Direct MLP | Vanilla |
|---:|---:|---:|---:|---:|
| 16 | **39.2%** / 15.844 passes | 37.6% / 14.976 | 36.4% / 14.964 | **39.2%** / 16.000 |
| 24 | 44.0% / 22.428 | 39.6% / 20.060 | 41.6% / 20.132 | **45.2%** / 24.000 |
| 32 | **52.4%** / 27.440 | 47.2% / 24.588 | 44.0% / 24.516 | 52.0% / 32.000 |

JEPA exceeded the direct control at 16 and 32 maximum steps, but it did not exceed confidence-based decoding. Non-prefix choice and rollout supervision therefore did not validate the main hypothesis.

Sources: `research_artifacts/results_phase3/summary.json` and `phase3-outcome-audit.json`.

## Phase 3.1: prefix/non-prefix and ranking ablations

**Question.** Did non-prefix actions or ranking fine-tuning cause the Phase 3 degradation?

**Changes.** No retraining occurred. The same capped candidate list was generated before filtering, then evaluated with either all candidates or prefix-only candidates. Both the pre-ranking and post-ranking checkpoints were tested for JEPA and direct heads at a 32-step maximum budget.

| Condition | Accuracy | Avg. passes |
|---|---:|---:|
| Adaptive confidence | **52.4%** | 27.440 |
| JEPA pre-ranking, prefix-only | 48.0% | 22.692 |
| JEPA post-ranking, all | 47.2% | 24.588 |
| JEPA post-ranking, prefix-only | 45.2% | 23.348 |
| JEPA pre-ranking, all | 44.4% | 23.688 |
| Direct post-ranking, prefix-only | 45.2% | 23.156 |
| Direct post-ranking, all | 44.0% | 24.516 |
| Direct pre-ranking, prefix-only | 42.4% | 22.096 |
| Direct pre-ranking, all | 42.4% | 22.948 |

The best JEPA ablation was pre-ranking and prefix-only, suggesting that restricting the action space helped and that ranking refinement did not improve this online setting. It still trailed adaptive confidence by 4.4 percentage points. Against adaptive confidence it rescued 14 questions and lost 25.

The accompanying offline audit is explicitly off-policy. Its confidence-index correction uses the minimum-confidence feature rather than mean confidence; that correction affects the audit, not the online evaluation. Phase 3/3.1 used group sizes `[4, 8, 16]`, whereas Phase 2 also included size 2, so the saved 52.8% Phase 2 adaptive run and the 52.4% Phase 3/3.1 adaptive run are not identical configurations.

Sources: `research_artifacts/results_phase31/summary.json` and `mixed_outcome_audit.json`.

## Phase 4: selective JEPA intervention

**Question.** Can the best Phase 3.1 controller help only when its predicted advantage over adaptive confidence is large enough?

**Changes.** The pre-ranking JEPA checkpoint was used with prefix candidates only. Adaptive confidence was the default; JEPA overrode it when predicted rollout-cost improvement exceeded a margin. Margins `[0, 0.02, 0.05, 0.10]` were compared on GSM8K validation rows 64–99 (36 questions not used for Phase 3 validation labels). Margin 0 was selected after scoring 21/36 versus the baseline's 18/36.

**Recorded test result.** On the repeatedly inspected 250-example test subset, the hybrid achieved 121/250 (**48.4%**) at **16.248 average LLaDA passes**. The saved Phase 3.1 adaptive baseline achieved 131/250 (**52.4%**) at **27.44 passes**. The hybrid rescued 15 adaptive errors, lost 25 adaptive successes, and had net −10 correct answers.

The lower realized pass count is a speed advantage for this measured operating point. It does **not** establish JEPA superiority because no confidence-only baseline was tuned or evaluated at comparable realized compute (approximately 16.248 passes). Gate scoring also requires hidden states and adds policy computation. The 36-question selection set is small, and prior inspection of the test subset prevents treating this as a clean confirmatory test.

Source: `research_artifacts/results_phase4/summary.json`.

## Overall conclusion

The experiments show that predicting future representations and fitting action-risk or rollout-ranking targets is not sufficient evidence of a better decoding policy. Strong offline cosine, correlation, or ranking metrics repeatedly failed to produce an online GSM8K advantage. Within the tested design, adaptive confidence was simpler and more accurate. The project is archived as a negative result rather than extended with further tuning.
