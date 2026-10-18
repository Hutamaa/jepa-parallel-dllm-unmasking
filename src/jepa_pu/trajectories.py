from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ensure_output_dirs, load_config
from .data import read_jsonl
from .features import (
    FrozenProjector,
    action_features,
    counterfactual_damage,
    commitment_shift_risk,
    target_representation,
)
from .llada import LLaDA, commit, extend_attention_mask, make_candidates, proposals
from .utils import atomic_torch_save, seed_everything


def _sampled_steps(total_steps: int, count: int) -> set[int]:
    if count >= total_steps:
        return set(range(total_steps))
    return {round(i * (total_steps - 1) / max(count - 1, 1)) for i in range(count)}


def _save_shard(samples: list[dict], output_dir: Path, split: str, shard: int, metadata: dict) -> None:
    if not samples:
        return
    atomic_torch_save(
        {"metadata": metadata, "samples": samples},
        output_dir / f"{split}-{shard:05d}.pt",
    )


@torch.inference_mode()
def generate(cfg: dict, split: str, force: bool = False) -> None:
    if split not in ("train", "validation"):
        raise ValueError("Counterfactual trajectories may only use train or validation data")
    seed_everything(int(cfg["seed"]))
    ensure_output_dirs(cfg)
    output_dir = Path(cfg["paths"]["trajectory_dir"])
    existing = sorted(output_dir.glob(f"{split}-*.pt"))
    if existing and not force:
        raise FileExistsError(f"Found existing {split} shards; pass --force to replace")
    if force:
        for path in existing:
            path.unlink()

    rows = read_jsonl(Path(cfg["paths"]["data_dir"]) / f"gsm8k_{split}.jsonl")
    backbone = LLaDA(cfg)
    gcfg = cfg["generation"]
    tcfg = cfg["trajectories"]
    gen_length = int(gcfg["gen_length"])
    block_length = int(gcfg["block_length"])
    total_steps = int(gcfg["steps"])
    if gen_length % block_length or total_steps % (gen_length // block_length):
        raise ValueError("gen_length must divide into block_length and steps into blocks")
    num_blocks = gen_length // block_length
    steps_per_block = total_steps // num_blocks
    chosen_steps = _sampled_steps(total_steps, int(tcfg["states_per_example"]))

    hidden_size = int(getattr(backbone.model.config, "d_model", 0) or backbone.model.config.hidden_size)
    projector = FrozenProjector(
        hidden_size, int(tcfg["projection_dim"]), int(tcfg["projection_seed"]), backbone.device
    )
    risk_objective = str(tcfg.get("risk_objective", "instability_v1"))
    if risk_objective not in {"instability_v1", "commitment_shift_v2"}:
        raise ValueError(f"Unsupported trajectories.risk_objective: {risk_objective}")
    metadata = {
        "hidden_size": hidden_size,
        "projection_dim": projector.output_size,
        "projection_seed": int(tcfg["projection_seed"]),
        "feature_dim": 4 * projector.output_size + 26,
        "target_dim": 3 * projector.output_size,
        "damage": ("mean selected-token uncertainty plus residual-token JS distribution shift"
                   if risk_objective == "commitment_shift_v2"
                   else "retained-proposal probability loss plus argmax flip rate on remaining masks"),
        "risk_objective": risk_objective,
    }
    samples: list[dict] = []
    shard = 0
    examples_in_shard = 0
    global_step = 0

    for row in tqdm(rows, desc=f"counterfactual-{split}"):
        prompt, prompt_mask = backbone.encode_prompt(row["question"])
        prompt_len = prompt.shape[1]
        response_slice = slice(prompt_len, prompt_len + gen_length)
        attention_mask = extend_attention_mask(prompt_mask, gen_length)
        x = backbone.initial_state(prompt, gen_length)
        global_step = 0

        for block in range(num_blocks):
            block_start = prompt_len + block * block_length
            block_end = block_start + block_length
            for local_step in range(steps_per_block):
                current = backbone.forward(x, attention_mask)
                mask = x.eq(backbone.mask_id)
                active_mask = mask.clone()
                active_mask[:, :block_start] = False
                active_mask[:, block_end:] = False
                forbidden = []
                if gcfg.get("logits_eos_inf", False):
                    forbidden.append(int(gcfg["eos_id"]))
                sampled, confidence, margin = proposals(
                    current.logits, mask, float(gcfg["temperature"]), forbidden
                )
                if gcfg.get("confidence_eos_eot_inf", False):
                    for token_id in (int(gcfg["eos_id"]), int(gcfg["eot_id"])):
                        confidence[sampled.eq(token_id)] = -torch.inf

                remaining = int(active_mask.sum().item())
                steps_left = steps_per_block - local_step
                required = math.ceil(remaining / steps_left)
                if global_step in chosen_steps:
                    candidates = make_candidates(
                        sampled,
                        confidence,
                        margin,
                        active_mask,
                        list(gcfg["group_sizes"]),
                        minimum_size=1,
                        max_candidates=int(tcfg["max_candidates_per_state"]),
                    )
                    for candidate in candidates:
                        next_x = commit(x, candidate)
                        future = backbone.forward(next_x, attention_mask)
                        embeddings = backbone.token_embeddings(candidate.tokens)
                        feature = action_features(
                            current,
                            x,
                            candidate,
                            response_slice,
                            active_mask,
                            embeddings,
                            projector,
                        )
                        target = target_representation(
                            future,
                            next_x,
                            candidate,
                            response_slice,
                            backbone.mask_id,
                            projector,
                        )
                        remaining_after = next_x.eq(backbone.mask_id)
                        if risk_objective == "commitment_shift_v2":
                            damage = commitment_shift_risk(
                                current,
                                future,
                                candidate.positions,
                                candidate.confidences,
                                remaining_after,
                                float(tcfg["damage_retention_weight"]),
                                float(tcfg["damage_flip_weight"]),
                            )
                        else:
                            damage = counterfactual_damage(
                                current,
                                future,
                                sampled,
                                remaining_after,
                                float(tcfg["damage_retention_weight"]),
                                float(tcfg["damage_flip_weight"]),
                            )
                        samples.append(
                            {
                                "features": feature.cpu().to(torch.float16),
                                "target": target.cpu().to(torch.float16),
                                "damage": damage.cpu().float(),
                                "example_id": row["id"],
                                "step": global_step,
                                "group_size": len(candidate.positions),
                            }
                        )

                vanilla = make_candidates(
                    sampled, confidence, margin, active_mask, [required], minimum_size=required
                )[0]
                x = commit(x, vanilla)
                global_step += 1

        examples_in_shard += 1
        if examples_in_shard >= int(tcfg["examples_per_shard"]):
            _save_shard(samples, output_dir, split, shard, metadata)
            samples, examples_in_shard, shard = [], 0, shard + 1
    _save_shard(samples, output_dir, split, shard, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(load_config(args.config), args.split, args.force)


if __name__ == "__main__":
    main()
