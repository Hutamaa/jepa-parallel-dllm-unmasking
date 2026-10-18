"""Generate outcome-labeled, non-prefix counterfactuals for Phase 3.

Expensive *offline* supervision: one frozen-LLaDA greedy rollout per proposed
candidate, using GSM8K gold answers only from prepared train/validation splits.
Generation must execute under PBS on an allocated GPU node, never login node.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ensure_output_dirs, load_config
from .data import extract_answer
from .features import FrozenProjector, action_features, target_representation
from .llada import LLaDA, commit, extend_attention_mask, make_candidates
from .phase3_candidates import build_candidates
from .phase3_rollouts import _propose, complete_after_action, outcome_cost
from .utils import atomic_torch_save, read_jsonl, seed_everything


def save_shard(rows: list[dict], path: Path, metadata: dict) -> None:
    if rows:
        atomic_torch_save({"metadata": metadata, "samples": rows}, path)


@torch.inference_mode()
def generate(cfg: dict, split: str) -> None:
    if split not in ("train", "validation"):
        raise ValueError("Training trajectories must use train/validation, never test")
    seed_everything(int(cfg["seed"]))
    ensure_output_dirs(cfg)
    out = Path(cfg["paths"]["trajectory_dir"])
    if list(out.glob(f"{split}-*.pt")):
        raise FileExistsError(f"Existing Phase 3 {split} shards found; will not overwrite: {out}")

    gen = cfg["generation"]
    tc = cfg["trajectories"]
    length = int(gen["gen_length"])
    if length != int(gen["block_length"]):
        raise ValueError("Phase 3 supports a single generation block only")
    steps = int(gen["steps"])
    states = sorted(set(int(s) for s in tc["sample_step_indices"]))
    if not states or states[0] < 0 or states[-1] >= steps:
        raise ValueError("sample_step_indices must fall within generation steps")
    n = int(tc["train_examples"] if split == "train" else tc["validation_examples"])
    rows = read_jsonl(Path(cfg["paths"]["data_dir"]) / f"gsm8k_{split}.jsonl")[:n]
    if len(rows) != n:
        raise ValueError(f"Not enough prepared {split} rows for Phase 3: {len(rows)} vs {n}")
    if any(not row.get("gold") for row in rows):
        raise ValueError("Prepared training data must include numeric gold answers")

    backbone = LLaDA(cfg)
    hidden_size = int(getattr(backbone.model.config, "d_model", 0) or backbone.model.config.hidden_size)
    proj = FrozenProjector(hidden_size, int(tc["projection_dim"]),
                           int(tc["projection_seed"]), backbone.device)
    metadata = {
        "hidden_size": hidden_size,
        "projection_dim": proj.output_size,
        "projection_seed": int(tc["projection_seed"]),
        "feature_dim": 4 * proj.output_size + 26,
        "target_dim": 3 * proj.output_size,
        "risk_objective": "rollout_answer_correctness_plus_compute_v3",
        "damage": "-GSM8K_numeric_correct + lambda*(vanilla_continuation_passes/remaining_slots)",
        "candidate_policy": "nonprefix_v3",
        "sample_step_indices": states,
        "latency_weight": float(tc["latency_weight"]),
        "continuation_policy": "vanilla confidence with fixed remaining step slots",
    }
    examples_per_shard = int(tc["examples_per_shard"])
    samples = []
    shard = 0
    examples_in_shard = 0
    observations = {"states": 0, "nonprefix_candidates": 0, "candidates": 0,
                    "correct_rollouts": 0}

    for row in tqdm(rows, desc=f"phase3-outcome-{split}"):
        prompt, pmask = backbone.encode_prompt(row["question"])
        plen = int(prompt.shape[1])
        response = slice(plen, plen + length)
        attention = extend_attention_mask(pmask, length)
        x = backbone.initial_state(prompt, length)
        for step in range(steps):
            need_hidden = step in states
            current = backbone.forward(x, attention, extract_hidden=need_hidden)
            mask = x.eq(backbone.mask_id)
            active = mask.clone()
            active[:, :plen] = False
            active[:, plen + length:] = False
            remaining = int(active.sum().item())
            if remaining == 0:
                break
            sampled, confidence, margin = _propose(backbone, cfg, current.logits, mask)
            min_group = math.ceil(remaining / (steps - step))
            if need_hidden:
                candidates = build_candidates(
                    sampled, confidence, margin, active,
                    list(gen["group_sizes"]), min_group,
                    int(gen.get("max_candidates", 6)),
                )
                if len(candidates) < 2:
                    print(f"WARN no rankable candidate choices for {row['id']} step {step}", flush=True)
                ranked = torch.nonzero(active[0], as_tuple=False).flatten()
                ranked = ranked[torch.argsort(confidence[0, ranked], descending=True, stable=True)]
                for idx, c in enumerate(candidates):
                    candidate_state = commit(x, c)
                    future = backbone.forward(candidate_state, attention, extract_hidden=True)
                    feats = action_features(current, x, c, response, active,
                                            backbone.token_embeddings(c.tokens), proj)
                    target = target_representation(future, candidate_state, c,
                                                   response, backbone.mask_id, proj)
                    text, forward_calls = complete_after_action(
                        backbone, cfg, x, attention, plen, c, step, steps,
                        first_continuation_output=future,
                    )
                    predicted = extract_answer(text)
                    correct = predicted is not None and predicted == row["gold"]
                    risk = outcome_cost(correct, forward_calls, steps - step - 1,
                                        float(tc["latency_weight"]))
                    is_prefix = (set(int(v) for v in c.positions.tolist()) ==
                                 set(int(v) for v in ranked[:len(c.positions)].tolist()))
                    samples.append({
                        "features": feats.detach().cpu().to(torch.float16),
                        "target": target.detach().cpu().to(torch.float16),
                        "damage": torch.tensor(risk, dtype=torch.float32),
                        "correct": bool(correct),
                        "continuation_passes": int(forward_calls),
                        "example_id": row["id"], "step": int(step),
                        "state_id": f"{row['id']}/step{step}",
                        "candidate_index": idx, "is_prefix": bool(is_prefix),
                        "group_size": len(c.positions),
                    })
                    observations["candidates"] += 1
                    observations["correct_rollouts"] += int(correct)
                    observations["nonprefix_candidates"] += int(not is_prefix)
                observations["states"] += 1
            vanilla = make_candidates(sampled, confidence, margin, active,
                                      [min_group], minimum_size=min_group)[0]
            x = commit(x, vanilla)
        examples_in_shard += 1
        if examples_in_shard >= examples_per_shard:
            save_shard(samples, out / f"{split}-{shard:05d}.pt", metadata)
            print(f"SAVED split={split} shard={shard} examples={examples_in_shard} "
                  f"total_candidates={observations['candidates']}", flush=True)
            samples = []
            examples_in_shard = 0
            shard += 1
    save_shard(samples, out / f"{split}-{shard:05d}.pt", metadata)
    print(f"PHASE3 {split} observations={observations}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3.yaml")
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    args = parser.parse_args()
    generate(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
