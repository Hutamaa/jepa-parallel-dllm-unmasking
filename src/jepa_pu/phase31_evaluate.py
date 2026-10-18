"""Phase 3.1: 2x2 action-space/ranking-checkpoint ablation, NO retraining.

Run only on a PBS-allocated GPU compute node. Reuses frozen Phase 3 LLaDA and
pre-/post-ranking checkpoints, and evaluates a contemporaneous confidence
baseline on the exact same held-out GSM8K test examples.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .config import load_config
from .data import extract_answer
from .decode import AdaptivePolicy, decode
from .evaluate import markdown_table, summarize
from .llada import LLaDA
from .utils import read_jsonl, seed_everything, write_jsonl


def _variant_cfg(cfg: dict, candidate_filter: str) -> dict:
    local = copy.deepcopy(cfg)
    local["generation"]["candidate_policy"] = "nonprefix_v3"
    local["generation"]["phase31_candidate_filter"] = candidate_filter
    local["generation"]["confidence_weight"] = 0.0
    return local


def run(cfg: dict, budget: int, max_examples: int, warmup: int) -> None:
    seed_everything(int(cfg["seed"]))
    base = Path(cfg["paths"]["checkpoint_dir"])
    checkpoints = {
        "pre": base / "phase1-before-ranking.pt",
        "post": base / "phase1.pt",
    }
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint missing: {path}")
    rows = read_jsonl(Path(cfg["paths"]["data_dir"]) / "gsm8k_test.jsonl")[:max_examples]
    if len(rows) != max_examples:
        raise ValueError(f"Need {max_examples} test examples but found {len(rows)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test IDs")
    outdir = Path(cfg["paths"]["output_dir"]).parent / "results_phase31"
    if (outdir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 3.1 results: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading ONE frozen LLaDA backbone (no downloads; no training)", flush=True)
    backbone = LLaDA(cfg)
    policies = {name: AdaptivePolicy(path, backbone.device) for name, path in checkpoints.items()}
    for policy in policies.values():
        policy.attach_backbone(backbone)

    conditions = [("adaptive_confidence", None, "all", "adaptive_confidence")]
    for phase in ("pre", "post"):
        for mode in ("prefix", "all"):
            for kind in ("jepa", "direct"):
                conditions.append((kind, phase, mode, f"{kind}_{phase}_{mode}"))

    all_rows = []
    by_condition: dict[str, dict[str, dict]] = {}
    for kind, checkpoint_name, mode, label in conditions:
        local = _variant_cfg(cfg, mode)
        policy = policies[checkpoint_name] if checkpoint_name else None
        for item in rows[:warmup]:
            decode(backbone, item["question"], local, kind, budget, policy)
        records = []
        print(f"START condition={label} n={len(rows)} budget={budget}", flush=True)
        for index, item in enumerate(rows, 1):
            text, stats = decode(backbone, item["question"], local, kind, budget, policy)
            pred = extract_answer(text)
            records.append({
                "id": item["id"], "method": label, "step_budget": budget,
                "prediction": text, "extracted_answer": pred,
                "gold_answer": item["gold"],
                "correct": pred is not None and pred == item["gold"],
                "stats": stats,
            })
            if index % 25 == 0 or index == len(rows):
                print(f"PROGRESS condition={label} {index}/{len(rows)}", flush=True)
        write_jsonl(outdir / f"predictions-{label}-steps{budget}.jsonl", records)
        by_condition[label] = {r["id"]: r for r in records}
        all_rows.extend(records)
        print(f"DONE condition={label} correct={sum(r['correct'] for r in records)}/{len(records)}", flush=True)

    summary = summarize(all_rows)
    adaptive = by_condition["adaptive_confidence"]
    comparisons = []
    for label, outcomes in by_condition.items():
        if label == "adaptive_confidence":
            continue
        improves = sum(outcomes[k]["correct"] and not adaptive[k]["correct"] for k in adaptive)
        regressions = sum(not outcomes[k]["correct"] and adaptive[k]["correct"] for k in adaptive)
        comparisons.append({
            "method": label, "baseline": "adaptive_confidence",
            "rescued_questions": improves, "lost_questions": regressions,
            "net_rescued": improves - regressions,
        })
    payload = {
        "phase": "3.1", "goal": "isolate action space and ranking fine-tuning",
        "dataset": "same 250 GSM8K test examples as Phase 3" if max_examples == 250 else "subset of Phase 3 GSM8K test",
        "step_budget": budget, "no_retraining": True,
        "same_full_candidate_list_before_filtering": True,
        "ranked_checkpoint": str(checkpoints["post"]),
        "pre_rank_checkpoint": str(checkpoints["pre"]),
        "summary": summary,
        "paired_comparisons_vs_adaptive": comparisons,
        "caveat": "Budget is a maximum, not a fixed number of forward passes; inspect realized compute and latency.",
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (outdir / "summary.md").write_text(
        "# Phase 3.1 — no retraining\n\n" + markdown_table(summary)
        + "\n## Paired correctness versus adaptive confidence\n\n"
        + "| Method | Rescued | Lost | Net |\n|---|---:|---:|---:|\n"
        + "".join(f"| {r['method']} | {r['rescued_questions']} | {r['lost_questions']} | {r['net_rescued']} |\n"
                  for r in comparisons)
    )
    print("PHASE31_EVAL_DONE " + json.dumps({"summary": summary, "comparisons": comparisons}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase3.yaml")
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--examples", type=int, default=250)
    p.add_argument("--warmup", type=int, default=2)
    a = p.parse_args()
    run(load_config(a.config), a.budget, a.examples, a.warmup)


if __name__ == "__main__":
    main()
