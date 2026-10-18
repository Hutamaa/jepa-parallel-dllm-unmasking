"""Fine-tune heads to select better *within-state* counterfactual outcomes.

Run after normal JEPA + scalar-head training. Frozen predictor; ranking loss
uses actual greedy rollout labels, NEVER GSM8K test references.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_config
from .decode import AdaptivePolicy
from .utils import seed_everything


def read_groups(path: Path, split: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    grouped = defaultdict(list)
    shards = sorted(path.glob(f"{split}-*.pt"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards under {path}")
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if shard["metadata"].get("risk_objective") != "rollout_answer_correctness_plus_compute_v3":
            raise ValueError(f"Not Phase 3 rollout supervision: {shard_path}")
        for sample in shard["samples"]:
            grouped[sample["state_id"]].append(sample)
    return [
        (torch.stack([v["features"].float() for v in samples]),
         torch.stack([v["damage"].float() for v in samples]))
        for samples in grouped.values() if len(samples) >= 2
    ]


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float,
                 margin: float) -> tuple[torch.Tensor, int]:
    i, j = torch.triu_indices(len(target), len(target), 1, device=target.device)
    preference_gap = target[i] - target[j]
    valid = preference_gap.abs() > margin
    if not bool(valid.any()):
        return pred.sum() * 0, 0
    i, j, preference_gap = i[valid], j[valid], preference_gap[valid]
    # If label_i < label_j, want pred_i < pred_j (both are costs).
    preferred_i = (preference_gap < 0).float()
    logits = (pred[j] - pred[i]) / temperature
    return F.binary_cross_entropy_with_logits(logits, preferred_i), len(i)


def loss_for_group(p: AdaptivePolicy, features: torch.Tensor, y: torch.Tensor,
                   temper: float, gap: float) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        latent = p.predictor(features)
    pj = p.jepa_head(torch.cat([features, latent], dim=-1))
    pd = p.direct_head(features)
    rj, nj = ranking_loss(pj, y, temper, gap)
    rd, nd = ranking_loss(pd, y, temper, gap)
    scalar = F.smooth_l1_loss(pj, y) + F.smooth_l1_loss(pd, y)
    loss = scalar + 0.25 * (rj + rd)
    return loss, {"jepa_pairwise": nj, "direct_pairwise": nd}


@torch.no_grad()
def validate(p: AdaptivePolicy, groups, device, temper, gap) -> float:
    p.jepa_head.eval(); p.direct_head.eval()
    score = []
    for feats, labels in groups:
        loss, _ = loss_for_group(p, feats.to(device), labels.to(device), temper, gap)
        score.append(float(loss.item()))
    return float(sum(score) / max(len(score), 1))


def run(cfg: dict) -> None:
    seed_everything(int(cfg["seed"]))
    tcfg = cfg["training"]
    device = torch.device(cfg["model"].get("device", "cuda"))
    cp = Path(cfg["paths"]["checkpoint_dir"]) / "phase1.pt"
    source_cp = cp.with_name("phase1-before-ranking.pt")
    if source_cp.exists():
        raise FileExistsError(f"Will not overwrite existing pre-ranking checkpoint: {source_cp}")
    shutil.copy2(cp, source_cp)
    policy = AdaptivePolicy(cp, device)
    policy.predictor.eval().requires_grad_(False)
    policy.jepa_head.train().requires_grad_(True)
    policy.direct_head.train().requires_grad_(True)
    train_groups = read_groups(Path(cfg["paths"]["trajectory_dir"]), "train")
    val_groups = read_groups(Path(cfg["paths"]["trajectory_dir"]), "validation")
    if not train_groups or not val_groups:
        raise RuntimeError("No rankable groups found in training/validation trajectories")
    opt = torch.optim.AdamW(
        list(policy.jepa_head.parameters()) + list(policy.direct_head.parameters()),
        lr=float(tcfg.get("ranking_learning_rate", 0.0001)),
        weight_decay=float(tcfg["weight_decay"]),
    )
    epochs = int(tcfg.get("ranking_epochs", 8))
    temper = float(tcfg.get("ranking_temperature", 0.20))
    gap = float(tcfg.get("ranking_min_gap", 0.005))
    patience = int(tcfg.get("ranking_patience", 3))
    best_val, stagnant, best_heads = math.inf, 0, None
    history = []
    for epoch in range(epochs):
        policy.jepa_head.train(); policy.direct_head.train()
        order = list(range(len(train_groups)))
        random.Random(int(cfg["seed"]) + epoch).shuffle(order)
        train_losses = []
        for i in order:
            feat, y = train_groups[i]
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_for_group(policy, feat.to(device), y.to(device), temper, gap)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().item()))
        val = validate(policy, val_groups, device, temper, gap)
        metric = {"epoch": epoch + 1, "train_group_loss": sum(train_losses) / len(train_losses),
                  "validation_group_loss": val}
        history.append(metric)
        print(json.dumps(metric), flush=True)
        if val < best_val:
            best_val, stagnant = val, 0
            best_heads = {
                "jepa_head": {k: v.detach().cpu().clone() for k, v in policy.jepa_head.state_dict().items()},
                "direct_head": {k: v.detach().cpu().clone() for k, v in policy.direct_head.state_dict().items()},
            }
        else:
            stagnant += 1
            if stagnant >= patience:
                break
    if best_heads is None:
        raise RuntimeError("No ranked checkpoint selected")
    ckpt = torch.load(source_cp, map_location="cpu", weights_only=False)
    ckpt.update(best_heads)
    ckpt["phase3_ranking_history"] = history
    tmp = cp.with_suffix(".ranking-tmp")
    torch.save(ckpt, tmp)
    tmp.replace(cp)
    (Path(cfg["paths"]["checkpoint_dir"]) / "phase3_ranking_metrics.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    print(f"Saved Phase 3 rank-refined heads to {cp}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
