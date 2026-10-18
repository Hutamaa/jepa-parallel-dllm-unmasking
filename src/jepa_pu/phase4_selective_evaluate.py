"""Last, inexpensive diagnostic: JEPA only overrides adaptive confidence if its
predicted rollout-cost advantage clears a validation-chosen margin.

No retraining. Runs on PBS GPU; calibration uses gsm8k_validation examples
64:100, which were not used to generate Phase 3 validation trajectories.
Never tune thresholds using the test set.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .config import load_config
from .data import extract_answer
from .decode import AdaptivePolicy, decode
from .evaluate import summarize, markdown_table
from .llada import LLaDA
from .utils import read_jsonl, seed_everything, write_jsonl


MARGINS = (0.0, 0.02, 0.05, 0.10)


def _cfg(base: dict, margin: float) -> dict:
    result = copy.deepcopy(base)
    result["generation"]["phase4_intervention_margin"] = margin
    return result


def _evaluate(rows: list[dict], backbone: LLaDA, cfg: dict,
              method: str, policy: AdaptivePolicy | None,
              label: str, warmup: int = 1) -> list[dict]:
    for example in rows[:warmup]:
        decode(backbone, example["question"], cfg, method, 32, policy)
    records = []
    print(f"START {label} 0/{len(rows)}", flush=True)
    for idx, row in enumerate(rows, 1):
        output, stats = decode(backbone, row["question"], cfg, method, 32, policy)
        predicted = extract_answer(output)
        records.append({
            "id": row["id"], "method": label, "step_budget": 32,
            "prediction": output, "extracted_answer": predicted,
            "gold_answer": row["gold"],
            "correct": predicted is not None and predicted == row["gold"],
            "stats": stats,
        })
        if idx % 12 == 0 or idx == len(rows):
            print(f"PROGRESS {label} {idx}/{len(rows)}", flush=True)
    print(f"DONE {label} correct={sum(int(r['correct']) for r in records)}/{len(rows)}", flush=True)
    return records


def _score(records: list[dict]) -> dict:
    stats = summarize(records)[0]
    return {
        "correct_count": sum(int(r["correct"]) for r in records),
        "total": len(records),
        "accuracy": stats["accuracy"],
        "latency_seconds": stats["avg_latency_seconds"],
        "llada_passes": stats["avg_llada_forward_passes"],
        "interventions_per_example": sum(r["stats"]["interventions"] for r in records) / len(records),
        "intervention_opportunities_per_example": sum(r["stats"]["intervention_opportunities"] for r in records) / len(records),
    }


def run(config_file: str) -> None:
    cfg = load_config(config_file)
    seed_everything(int(cfg["seed"]))
    data_dir = Path(cfg["paths"]["data_dir"])
    validation = read_jsonl(data_dir / "gsm8k_validation.jsonl")
    if len(validation) < 100:
        raise ValueError("Expected >=100 validation questions; need the held-out indices [64:100]")
    val_rows = validation[64:100]  # not used in the Phase 3 rollout validation labels
    test_rows = read_jsonl(data_dir / "gsm8k_test.jsonl")[:250]
    if len(test_rows) != 250:
        raise ValueError("Expected 250 test questions")
    if len({r["id"] for r in val_rows + test_rows}) != len(val_rows) + len(test_rows):
        raise ValueError("Duplicate IDs across evaluation splits")

    outdir = Path(cfg["paths"]["output_dir"]).parent / "results_phase4"
    if (outdir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite old results: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg["paths"]["checkpoint_dir"]) / "phase1-before-ranking.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing pre-ranking JEPA checkpoint: {checkpoint}")
    print("PHASE4: No training, no counterfactual generation; loading frozen LLaDA", flush=True)
    backbone = LLaDA(cfg)
    policy = AdaptivePolicy(checkpoint, backbone.device)
    policy.attach_backbone(backbone)

    baseline = _evaluate(val_rows, backbone, cfg, "adaptive_confidence", None, "adaptive_val")
    write_jsonl(outdir / "validation-adaptive.jsonl", baseline)
    base_score = _score(baseline)
    calibrations = []
    val_records = {}
    for margin in MARGINS:
        label = f"hybrid_margin_{margin:g}"
        local = _cfg(cfg, margin)
        records = _evaluate(val_rows, backbone, local, "selective_jepa", policy, f"{label}_val")
        write_jsonl(outdir / f"validation-{label}.jsonl", records)
        val_records[margin] = records
        measurements = _score(records)
        calibrations.append({"margin": margin, **measurements})
        print(f"CALIBRATION margin={margin:g}: {measurements}", flush=True)

    # Pre-registered: require at least two additional correct out of 36 AND
    # no >10% measured latency increase. Otherwise conclude no promising gate.
    qualifying = [
        r for r in calibrations
        if r["correct_count"] >= base_score["correct_count"] + 2
        and r["latency_seconds"] <= 1.10 * base_score["latency_seconds"]
    ]
    qualified = sorted(qualifying, key=lambda r: (-r["correct_count"], r["latency_seconds"], -r["margin"]))
    selected = qualified[0] if qualified else None
    result = {
        "phase": "4_final_selective_intervention",
        "status": "selected_for_test" if selected else "stop_no_validation_gain",
        "heldout_calibration": "gsm8k_validation rows 64:100 (36 questions), unused in Phase 3 labels",
        "notes": [
            "JEPA checkpoint is pre-ranking; actions are strictly confidence-prefix candidates.",
            "Gate is based on JEPA predicted rollout cost improvement over adaptive confidence.",
            "Selected margin uses validation only. The 250-question test set has been inspected in earlier experiments; treat follow-up comparisons as exploratory.",
            "Gate scoring always requires hidden states and adds compute overhead; compare actual latency and passes.",
        ],
        "margins": list(MARGINS),
        "validation_baseline": base_score,
        "validation_hybrids": calibrations,
        "selected_margin": selected["margin"] if selected else None,
        "test": None,
    }
    if not selected:
        (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (outdir / "summary.md").write_text(
            "# Phase 4: stop (no validation gain)\n\n"
            "No margin improved held-out 36-question validation by >=2 correct "
            "while remaining within 10% of baseline latency. No test re-run.\n\n"
            "See summary.json.\n")
        print("PHASE4_STOP: no validation evidence for selective intervention. Pivot recommended.", flush=True)
        return

    margin = selected["margin"]
    print(f"SELECTED margin={margin:g} (validation only); test 250 questions", flush=True)
    test = _evaluate(test_rows, backbone, _cfg(cfg, margin), "selective_jepa", policy, "hybrid_test")
    write_jsonl(outdir / "test-hybrid.jsonl", test)
    test_score = _score(test)

    # Reuse exact paired baseline predictions from Phase 3.1, avoiding another 250 GPU inference calls.
    baseline_path = outdir.parent / "results_phase31" / "predictions-adaptive_confidence-steps32.jsonl"
    paired = {r["id"]: r for r in read_jsonl(baseline_path)}
    if set(paired) != {r["id"] for r in test}:
        raise ValueError("Saved Phase 3.1 adaptive baseline IDs differ from this test set")
    rescued = sum(int(r["correct"] and not paired[r["id"]]["correct"]) for r in test)
    lost = sum(int(not r["correct"] and paired[r["id"]]["correct"]) for r in test)
    result["test"] = {
        "hybrid": test_score,
        "baseline_saved_phase31_accuracy": sum(int(r["correct"]) for r in paired.values()) / len(paired),
        "baseline_saved_phase31_avg_latency_seconds": sum(r["stats"]["total_seconds"] for r in paired.values()) / len(paired),
        "baseline_saved_phase31_avg_llada_passes": sum(r["stats"]["llada_forward_passes"] for r in paired.values()) / len(paired),
        "rescued": rescued, "lost": lost, "net": rescued - lost,
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (outdir / "summary.md").write_text(
        "# Phase 4: Final selective JEPA intervention\n\n"
        f"Held-out validation baseline: {base_score['correct_count']}/36. "
        f"Selected margin: {margin:g}.\n\n"
        f"Test hybrid: {test_score['correct_count']}/250 "
        f"({test_score['accuracy']:.1%}); saved adaptive baseline: "
        f"{result['test']['baseline_saved_phase31_accuracy']:.1%}.\n\n"
        f"Paired rescued={rescued}, lost={lost}, net={rescued-lost}.\n\n"
        "Latency comparison across separate GPU runs is approximate. This previously analyzed "
        "test subset cannot count as independent confirmation.\n"
    )
    print("PHASE4_DONE " + json.dumps(result["test"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
