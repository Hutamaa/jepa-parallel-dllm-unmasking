"""Validation-state ranking against actual *stored rollout outcomes*, not JS proxy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .config import load_config
from .decode import AdaptivePolicy


@torch.inference_mode()
def run(cfg: dict) -> dict:
    path = Path(cfg["paths"]["trajectory_dir"])
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "phase1.pt"
    policy = AdaptivePolicy(ckpt, torch.device("cpu"))
    states = defaultdict(list)
    for fn in sorted(path.glob("validation-*.pt")):
        shard = torch.load(fn, map_location="cpu", weights_only=False)
        if shard["metadata"].get("risk_objective") != "rollout_answer_correctness_plus_compute_v3":
            raise ValueError(f"Unexpected dataset objective: {fn}")
        samples = shard["samples"]
        if not samples:
            continue
        feat = torch.stack([r["features"].float() for r in samples])
        latent = policy.predictor(feat)
        jepa = policy.jepa_head(torch.cat([feat, latent], dim=-1)).tolist()
        direct = policy.direct_head(feat).tolist()
        for r, j, d in zip(samples, jepa, direct):
            states[r["state_id"]].append({
                "outcome_cost": float(r["damage"]),
                "correct": bool(r["correct"]),
                "jepa": float(j), "direct": float(d),
                "confidence": float(r["features"].float()[-7]),
                "group_size": int(r["group_size"]),
                "is_prefix": bool(r["is_prefix"]),
            })

    def aggregate(method: str) -> dict:
        n, top, advantage, chosen_correct, chosen_nonprefix, pair_total, pair_correct = 0, 0, 0.0, 0, 0, 0, 0
        for options in states.values():
            if len(options) < 2:
                continue
            n += 1
            costs = torch.tensor([o["outcome_cost"] for o in options])
            if method == "max_group":
                winner = max(range(len(options)), key=lambda i: (options[i]["group_size"], options[i]["confidence"]))
            elif method == "confidence":
                threshold = float(cfg["generation"].get("adaptive_confidence_threshold", 0.90))
                eligible = [i for i, o in enumerate(options) if o["confidence"] >= threshold and o["is_prefix"]]
                if eligible:
                    winner = max(eligible, key=lambda i: options[i]["group_size"])
                else:
                    winner = min(range(len(options)), key=lambda i: options[i]["group_size"])
            else:
                winner = min(range(len(options)), key=lambda i: options[i][method])
                for i in range(len(options)):
                    for j in range(i):
                        if abs(options[i]["outcome_cost"] - options[j]["outcome_cost"]) < 1e-5:
                            continue
                        pair_total += 1
                        pair_correct += int(
                            (options[i]["outcome_cost"] - options[j]["outcome_cost"]) *
                            (options[i][method] - options[j][method]) > 0
                        )
            regret = float(costs[winner] - costs.min())
            top += int(regret < 1e-6)
            advantage += regret
            chosen_correct += int(options[winner]["correct"])
            chosen_nonprefix += int(not options[winner]["is_prefix"])
        return {"states": n, "top1_rollout_cost_accuracy": top / n if n else None,
                "mean_rollout_cost_regret": advantage / n if n else None,
                "selected_action_rollout_correct_rate": chosen_correct / n if n else None,
                "selected_nonprefix_rate": chosen_nonprefix / n if n else None,
                "pairwise_outcome_concordance": pair_correct / pair_total if pair_total else None,
                "non_tied_pairs": pair_total}

    variable_correctness = sum(len({v["correct"] for v in options}) > 1 for options in states.values())
    report = {"source": "held-out GSM8K train-derived validation trajectory states",
              "important": "Labels evaluate one deterministic vanilla continuation per action; they do NOT guarantee final online policy accuracy.",
              "states": len(states),
              "states_with_different_correctness_across_actions": variable_correctness,
              "jepa": aggregate("jepa"), "direct": aggregate("direct"),
              "adaptive_confidence_on_same_action_set": aggregate("confidence"),
              "largest_group_on_same_action_set": aggregate("max_group")}
    output = Path(cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase3-outcome-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
