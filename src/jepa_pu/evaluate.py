from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ensure_output_dirs, load_config
from .data import extract_answer
from .decode import AdaptivePolicy, decode
from .llada import LLaDA
from .utils import read_jsonl, seed_everything, write_jsonl


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["step_budget"]), []).append(row)
    summary = []
    for (method, budget), items in sorted(groups.items()):
        summary.append({
            "method": method,
            "step_budget": budget,
            "examples": len(items),
            "accuracy": float(np.mean([item["correct"] for item in items])),
            "avg_steps": float(np.mean([item["stats"]["steps"] for item in items])),
            "avg_nonprefix_selections": float(np.mean([item["stats"].get("nonprefix_selected", 0) for item in items])),
            "avg_candidates_scored": float(np.mean([item["stats"].get("candidate_sets_evaluated", 0) for item in items])),
            "avg_llada_forward_passes": float(np.mean([item["stats"]["llada_forward_passes"] for item in items])),
            "avg_latency_seconds": float(np.mean([item["stats"]["total_seconds"] for item in items])),
            "avg_llada_seconds": float(np.mean([item["stats"]["llada_seconds"] for item in items])),
            "avg_policy_overhead_seconds": float(np.mean([item["stats"]["policy_seconds"] for item in items])),
            "peak_gpu_memory_bytes": int(max(item["stats"]["peak_gpu_memory_bytes"] for item in items)),
        })
    return summary


def markdown_table(summary: list[dict]) -> str:
    lines = [
        "| Method | Budget | Accuracy | Steps | LLaDA passes | Latency (s) | Policy overhead (s) | Nonprefix choices | Peak GPU GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['step_budget']} | {row['accuracy']:.4f} | "
            f"{row['avg_steps']:.2f} | {row['avg_llada_forward_passes']:.2f} | "
            f"{row['avg_latency_seconds']:.3f} | {row['avg_policy_overhead_seconds']:.3f} | "
            f"{row['avg_nonprefix_selections']:.2f} | "
            f"{row['peak_gpu_memory_bytes'] / 2**30:.2f} |"
        )
    return "\n".join(lines) + "\n"


def run(cfg: dict) -> None:
    seed_everything(int(cfg["seed"]))
    ensure_output_dirs(cfg)
    rows = read_jsonl(Path(cfg["paths"]["data_dir"]) / "gsm8k_test.jsonl")
    backbone = LLaDA(cfg)
    checkpoint = Path(cfg["paths"]["checkpoint_dir"]) / str(cfg["evaluation"].get("checkpoint_name", "phase1.pt"))
    methods = list(cfg["evaluation"]["methods"])
    policy = AdaptivePolicy(checkpoint, backbone.device) if any(m in ("jepa", "direct") for m in methods) else None
    if policy:
        policy.attach_backbone(backbone)
    output_dir = Path(cfg["paths"]["output_dir"])
    all_results = []
    warmup_count = int(cfg["evaluation"].get("warmup_examples", 0))

    for budget in cfg["evaluation"]["step_budgets"]:
        for method in methods:
            for row in rows[:warmup_count]:
                decode(backbone, row["question"], cfg, method, int(budget), policy)
            method_rows = []
            for index, row in enumerate(rows):
                text, stats = decode(backbone, row["question"], cfg, method, int(budget), policy)
                predicted = extract_answer(text)
                result = {
                    "id": row["id"], "method": method, "step_budget": int(budget),
                    "prediction": text, "extracted_answer": predicted, "gold_answer": row["gold"],
                    "correct": predicted is not None and predicted == row["gold"], "stats": stats,
                }
                method_rows.append(result)
                print(json.dumps({"index": index, **result}, ensure_ascii=False), flush=True)
            write_jsonl(output_dir / f"predictions-{method}-steps{budget}.jsonl", method_rows)
            all_results.extend(method_rows)

    summary = summarize(all_results)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(markdown_table(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
