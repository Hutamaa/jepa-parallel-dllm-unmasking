"""Offline action-ranking audit on stored counterfactual trajectory shards.

Never loads the LLaDA backbone. Uses existing representations and the trained
heads, so it is possible to separate proxy-rank performance from GSM8K quality.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_config
from .decode import AdaptivePolicy


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() < 2 or float(a.std()) <= 1e-10 or float(b.std()) <= 1e-10:
        return None
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _rank_diagnostics(states: dict, kind: str, threshold: float) -> dict:
    n_states, top1, regret_sum, n_pairs, concordant = 0, 0, 0.0, 0, 0
    for items in states.values():
        if len(items) < 2:
            continue
        n_states += 1
        y = torch.tensor([v["damage"] for v in items])
        if kind == "confidence":
            eligible = [i for i, v in enumerate(items) if v["min_confidence"] >= threshold]
            chosen = max(eligible, key=lambda i: items[i]["group_size"]) if eligible else min(
                range(len(items)), key=lambda i: items[i]["group_size"]
            )
        else:
            pred = torch.tensor([v[kind] for v in items])
            chosen = int(pred.argmin())
            for i in range(len(items)):
                for j in range(i):
                    y_delta = y[i] - y[j]
                    p_delta = pred[i] - pred[j]
                    if abs(float(y_delta)) < 1e-8 or abs(float(p_delta)) < 1e-8:
                        continue
                    n_pairs += 1
                    concordant += int(float(y_delta * p_delta) > 0)
        regret = max(0.0, float(y[chosen] - y.min()))
        regret_sum += regret
        top1 += int(regret < 1e-7)
    return {
        "states_with_multiple_candidates": n_states,
        "proxy_top1_accuracy": top1 / n_states if n_states else None,
        "mean_proxy_regret": regret_sum / n_states if n_states else None,
        "within_state_pairwise_concordance": concordant / n_pairs if n_pairs else None,
        "compared_pairs": n_pairs,
    }


@torch.inference_mode()
def audit(cfg: dict, split: str = "validation") -> dict:
    files = sorted(Path(cfg["paths"]["trajectory_dir"]).glob(f"{split}-*.pt"))
    if not files:
        raise FileNotFoundError(f"No {split} shards under {cfg['paths']['trajectory_dir']}")
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / str(cfg["evaluation"].get("checkpoint_name", "phase1.pt"))
    policy = AdaptivePolicy(ckpt, torch.device("cpu"))
    states = defaultdict(list)
    all_damage, all_jepa, all_direct = [], [], []
    baseline_cos, learned_cos = [], []
    repr_dim = int(policy.metadata["projection_dim"])
    total = 0
    for filename in files:
        shard = torch.load(filename, map_location="cpu", weights_only=False)
        if any(shard["metadata"].get(key) != policy.metadata.get(key) for key in ("hidden_size", "projection_dim", "projection_seed", "feature_dim", "target_dim")):
            raise ValueError(f"Checkpoint / trajectory metadata mismatch in {filename}")
        samples = shard["samples"]
        if not samples:
            continue
        feat = torch.stack([s["features"].float() for s in samples])
        target = torch.stack([s["target"].float() for s in samples])
        pred = policy.predictor(feat)
        jr = policy.jepa_head(torch.cat([feat, pred], dim=-1))
        dr = policy.direct_head(feat)
        base_repr = feat[:, :3 * repr_dim].reshape(-1, 3, repr_dim)
        tgt_repr = target.reshape(-1, 3, repr_dim)
        pred_repr = pred.reshape(-1, 3, repr_dim)
        baseline_cos.extend(F.cosine_similarity(base_repr, tgt_repr, dim=-1).mean(-1).tolist())
        learned_cos.extend(F.cosine_similarity(pred_repr, tgt_repr, dim=-1).mean(-1).tolist())
        for i, sample in enumerate(samples):
            key = (sample["example_id"], int(sample["step"]))
            datum = {
                "group_size": int(sample["group_size"]),
                "damage": float(sample["damage"]),
                "jepa": float(jr[i]),
                "direct": float(dr[i]),
                # feature scalar tail index 4 is min selected-token confidence
                "min_confidence": float(feat[i, -6]),
            }
            states[key].append(datum)
            all_damage.append(datum["damage"])
            all_jepa.append(datum["jepa"])
            all_direct.append(datum["direct"])
        total += len(samples)
    y = torch.tensor(all_damage)
    result = {
        "split": split,
        "risk_target_description": policy.metadata.get("damage"),
        "important": "Ranking and correlation assess only the stored proxy risk, NOT downstream correctness.",
        "samples": total,
        "states": len(states),
        "future_cosine_identity_current_repr": sum(baseline_cos) / len(baseline_cos),
        "future_cosine_trained_jepa": sum(learned_cos) / len(learned_cos),
        "global_proxy_pearson_jepa": _pearson(torch.tensor(all_jepa), y),
        "global_proxy_pearson_direct": _pearson(torch.tensor(all_direct), y),
        "ranking_jepa": _rank_diagnostics(states, "jepa", 0.0),
        "ranking_direct": _rank_diagnostics(states, "direct", 0.0),
        "ranking_adaptive_confidence": _rank_diagnostics(
            states, "confidence", float(cfg["generation"].get("adaptive_confidence_threshold", 0.90))
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2a.yaml")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = audit(cfg, args.split)
    dest = Path(cfg["paths"]["output_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"ranking-audit-{args.split}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
