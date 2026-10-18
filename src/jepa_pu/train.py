from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ensure_output_dirs, load_config
from .models import JEPAPredictor, RiskHead, representation_loss
from .utils import atomic_torch_save, seed_everything


def load_split(directory: Path, split: str) -> tuple[TensorDataset, dict]:
    files = sorted(directory.glob(f"{split}-*.pt"))
    if not files:
        raise FileNotFoundError(f"No {split} trajectory shards in {directory}")
    features, targets, damage = [], [], []
    metadata = None
    for path in files:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if metadata is None:
            metadata = shard["metadata"]
        elif metadata != shard["metadata"]:
            raise ValueError(f"Inconsistent metadata in {path}")
        features.extend(item["features"] for item in shard["samples"])
        targets.extend(item["target"] for item in shard["samples"])
        damage.extend(item["damage"] for item in shard["samples"])
    dataset = TensorDataset(
        torch.stack(features).float(), torch.stack(targets).float(), torch.stack(damage).float()
    )
    return dataset, metadata


def correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    p = prediction.double().flatten()
    t = target.double().flatten()
    if p.numel() < 2 or p.std() == 0 or t.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([p, t]))[0, 1])


@torch.no_grad()
def evaluate_repr(model, loader, repr_dim, variance_weight, device) -> dict:
    model.eval()
    losses, cosines = [], []
    for features, target, _ in loader:
        prediction = model(features.to(device))
        loss, parts = representation_loss(
            prediction, target.to(device), repr_dim, variance_weight
        )
        losses.append(loss.item())
        cosines.append(1.0 - parts["cosine_loss"].item())
    return {"loss": float(np.mean(losses)), "cosine_similarity": float(np.mean(cosines))}


@torch.no_grad()
def evaluate_risk(predictor, jepa_head, direct_head, loader, device) -> dict:
    predictor.eval(); jepa_head.eval(); direct_head.eval()
    jepa_predictions, direct_predictions, targets = [], [], []
    for features, _, damage in loader:
        features = features.to(device)
        predicted_repr = predictor(features)
        jepa_predictions.append(jepa_head(torch.cat([features, predicted_repr], dim=-1)).cpu())
        direct_predictions.append(direct_head(features).cpu())
        targets.append(damage)
    target = torch.cat(targets)
    jepa = torch.cat(jepa_predictions)
    direct = torch.cat(direct_predictions)
    return {
        "jepa_mae": (jepa - target).abs().mean().item(),
        "jepa_pearson": correlation(jepa, target),
        "direct_mae": (direct - target).abs().mean().item(),
        "direct_pearson": correlation(direct, target),
    }


def train(cfg: dict) -> None:
    seed_everything(int(cfg["seed"]))
    ensure_output_dirs(cfg)
    device = torch.device(cfg["model"].get("device", "cuda"))
    tcfg = cfg["training"]
    train_data, metadata = load_split(Path(cfg["paths"]["trajectory_dir"]), "train")
    val_data, val_metadata = load_split(Path(cfg["paths"]["trajectory_dir"]), "validation")
    if metadata != val_metadata:
        raise ValueError("Train and validation trajectory metadata differ")
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_data, batch_size=int(tcfg["batch_size"]), shuffle=True,
        num_workers=int(tcfg["num_workers"]), generator=generator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=int(tcfg["num_workers"]), pin_memory=True,
    )
    input_dim = int(metadata["feature_dim"])
    repr_dim = int(metadata["projection_dim"])
    hidden_dim = int(tcfg["hidden_dim"])
    dropout = float(tcfg["dropout"])
    predictor = JEPAPredictor(input_dim, hidden_dim, repr_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=float(tcfg["weight_decay"])
    )
    history = []
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(int(tcfg["epochs"])):
        predictor.train()
        epoch_losses = []
        for features, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = predictor(features.to(device))
            loss, _ = representation_loss(
                prediction, target.to(device), repr_dim, float(tcfg["variance_weight"])
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        metrics = evaluate_repr(
            predictor, val_loader, repr_dim, float(tcfg["variance_weight"]), device
        )
        record = {"stage": "jepa", "epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["loss"] < best_loss:
            best_loss, stale = metrics["loss"], 0
            best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}
        else:
            stale += 1
            if stale >= int(tcfg["patience"]):
                break
    predictor.load_state_dict(best_state)
    predictor.eval().requires_grad_(False)

    jepa_head = RiskHead(input_dim + 3 * repr_dim, hidden_dim // 2, dropout).to(device)
    direct_head = RiskHead(input_dim, hidden_dim // 2, dropout).to(device)
    params = list(jepa_head.parameters()) + list(direct_head.parameters())
    head_optimizer = torch.optim.AdamW(
        params, lr=float(tcfg["learning_rate"]), weight_decay=float(tcfg["weight_decay"])
    )
    loss_fn = nn.SmoothL1Loss()
    best_heads, best_head_loss, stale = None, float("inf"), 0
    for epoch in range(int(tcfg["epochs"])):
        jepa_head.train(); direct_head.train()
        losses = []
        for features, _, damage in train_loader:
            features, damage = features.to(device), damage.to(device)
            with torch.no_grad():
                predicted_repr = predictor(features)
            head_optimizer.zero_grad(set_to_none=True)
            jepa_loss = loss_fn(jepa_head(torch.cat([features, predicted_repr], -1)), damage)
            direct_loss = loss_fn(direct_head(features), damage)
            loss = jepa_loss + direct_loss
            loss.backward()
            head_optimizer.step()
            losses.append(loss.item())
        metrics = evaluate_risk(predictor, jepa_head, direct_head, val_loader, device)
        combined = metrics["jepa_mae"] + metrics["direct_mae"]
        record = {"stage": "risk", "epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if combined < best_head_loss:
            best_head_loss, stale = combined, 0
            best_heads = {
                "jepa_head": {k: v.detach().cpu().clone() for k, v in jepa_head.state_dict().items()},
                "direct_head": {k: v.detach().cpu().clone() for k, v in direct_head.state_dict().items()},
            }
        else:
            stale += 1
            if stale >= int(tcfg["patience"]):
                break
    jepa_head.load_state_dict(best_heads["jepa_head"])
    direct_head.load_state_dict(best_heads["direct_head"])

    checkpoint = {
        "predictor": predictor.cpu().state_dict(),
        "jepa_head": jepa_head.cpu().state_dict(),
        "direct_head": direct_head.cpu().state_dict(),
        "metadata": metadata,
        "architecture": {"hidden_dim": hidden_dim, "dropout": dropout},
        "config": cfg,
        "history": history,
    }
    path = Path(cfg["paths"]["checkpoint_dir"]) / "phase1.pt"
    atomic_torch_save(checkpoint, path)
    with open(path.with_suffix(".metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
