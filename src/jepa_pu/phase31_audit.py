"""Phase 3.1 held-out counterfactual ranking audit (NO training).

Focus on within-state outcome-disagreement, not just top-1 cost accuracy
inflated by homogeneous/tied states. Read existing validation shards; all
computations run inside a PBS-allocated compute-node job.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .config import load_config
from .decode import AdaptivePolicy


@torch.inference_mode()
def run(cfg: dict) -> None:
    cp = Path(cfg["paths"]["checkpoint_dir"])
    paths = {
        "pre": cp / "phase1-before-ranking.pt",
        "post": cp / "phase1.pt",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    shard_paths = sorted(Path(cfg["paths"]["trajectory_dir"]).glob("validation-*.pt"))
    if not shard_paths:
        raise FileNotFoundError("Missing Phase 3 validation trajectory shards")
    policies = {name: AdaptivePolicy(path, torch.device("cpu")) for name, path in paths.items()}
    states: dict[str, list[dict]] = defaultdict(list)
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("metadata", {}).get("risk_objective") != "rollout_answer_correctness_plus_compute_v3":
            raise ValueError(f"Not Phase 3 rollout data: {path}")
        samples = shard["samples"]
        if not samples:
            continue
        features = torch.stack([r["features"].float() for r in samples])
        outputs = {}
        for phase, policy in policies.items():
            z = policy.predictor(features)
            outputs[phase] = (
                policy.jepa_head(torch.cat([features, z], dim=-1)).view(-1).tolist(),
                policy.direct_head(features).view(-1).tolist(),
            )
        for idx, sample in enumerate(samples):
            feat = sample["features"].float()
            states[sample["state_id"]].append({
                "cost": float(sample["damage"]),
                "correct": bool(sample["correct"]),
                "is_prefix": bool(sample["is_prefix"]),
                "group_size": int(sample["group_size"]),
                # Last ten scalar features: the 4th/5th are mean/min confidence.
                # The previous Phase 3 audit incorrectly used mean confidence
                # for a minimum-confidence threshold; this fixes it.
                "mean_confidence": float(feat[-7]),
                "min_confidence": float(feat[-6]),
                "jepa_pre": outputs["pre"][0][idx],
                "direct_pre": outputs["pre"][1][idx],
                "jepa_post": outputs["post"][0][idx],
                "direct_post": outputs["post"][1][idx],
            })

    def select(options: list[dict], method: str) -> int:
        if method == "adaptive_confidence":
            prefix = [i for i, o in enumerate(options) if o["is_prefix"]]
            if not prefix:
                raise RuntimeError("No prefix action in validation candidate set")
            cutoff = float(cfg["generation"].get("adaptive_confidence_threshold", 0.90))
            eligible = [i for i in prefix if options[i]["min_confidence"] >= cutoff]
            return (max(eligible, key=lambda i: options[i]["group_size"])
                    if eligible else min(prefix, key=lambda i: options[i]["group_size"]))
        return min(range(len(options)), key=lambda i: options[i][method])

    summary = []
    for candidate_filter in ("prefix", "all"):
        dataset = []
        for options in states.values():
            sub = [o for o in options if candidate_filter == "all" or o["is_prefix"]]
            if len(sub) >= 2:
                dataset.append(sub)
        mixed = [opts for opts in dataset if len({o["correct"] for o in opts}) > 1]
        for method in ("adaptive_confidence", "jepa_pre", "jepa_post", "direct_pre", "direct_post"):
            chosen_correct_all, chosen_correct_mixed, top1, regret, nonprefix = 0, 0, 0, 0., 0
            valid_pairs, concordant_pairs = 0, 0.
            for options in dataset:
                selected = select(options, method)
                best_cost = min(o["cost"] for o in options)
                o = options[selected]
                chosen_correct_all += int(o["correct"])
                top1 += int(o["cost"] <= best_cost + 1e-6)
                regret += o["cost"] - best_cost
                nonprefix += int(not o["is_prefix"])
                if len({v["correct"] for v in options}) > 1:
                    chosen_correct_mixed += int(o["correct"])
                    if method != "adaptive_confidence":
                        for i in range(len(options)):
                            for j in range(i):
                                if options[i]["correct"] == options[j]["correct"]:
                                    continue
                                # Correct candidate should have a lower predicted cost.
                                better, worse = ((i, j) if options[i]["correct"] else (j, i))
                                diff = options[better][method] - options[worse][method]
                                valid_pairs += 1
                                concordant_pairs += (1.0 if diff < -1e-8 else 0.5 if abs(diff) <= 1e-8 else 0.0)
            n = len(dataset)
            summary.append({
                "candidate_filter": candidate_filter,
                "method": method,
                "states": n,
                "states_with_mixed_correctness": len(mixed),
                "mixed_state_fraction": len(mixed) / n if n else None,
                "chosen_correct_rate_all": chosen_correct_all / n if n else None,
                "chosen_correct_rate_mixed": chosen_correct_mixed / len(mixed) if mixed else None,
                "oracle_correct_rate_all": sum(any(o["correct"] for o in ops) for ops in dataset) / n if n else None,
                "top1_rollout_cost_accuracy": top1 / n if n else None,
                "mean_rollout_cost_regret": regret / n if n else None,
                "nonprefix_selection_rate": nonprefix / n if n else None,
                "cross_correctness_pairwise_concordance": concordant_pairs / valid_pairs if valid_pairs else None,
                "cross_correctness_pair_count": valid_pairs,
            })

    out = Path(cfg["paths"]["output_dir"]).parent / "results_phase31"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "held-out Phase 3 vanilla-trajectory validation rollouts",
        "limitation": "Off-policy single deterministic continuation; this audit is NOT online GSM8K accuracy.",
        "confidence_index_fix": "minimum-confidence threshold uses features[-6], not features[-7] mean confidence",
        "summary": summary,
    }
    (out / "mixed_outcome_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PHASE31_AUDIT_DONE " + json.dumps(summary), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase3.yaml")
    args = p.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
