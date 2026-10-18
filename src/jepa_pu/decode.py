from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .features import FrozenProjector, action_features
from .llada import LLaDA, commit, extend_attention_mask, make_candidates, proposals
from .models import JEPAPredictor, RiskHead
from .phase3_candidates import build_candidates


@dataclass
class DecodeStats:
    steps: int = 0
    llada_forward_passes: int = 0
    llada_seconds: float = 0.0
    policy_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_gpu_memory_bytes: int = 0
    nonprefix_selected: int = 0
    candidate_sets_evaluated: int = 0
    interventions: int = 0
    intervention_opportunities: int = 0
    predicted_advantage_sum: float = 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class AdaptivePolicy:
    def __init__(self, checkpoint_path: str | Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        architecture = checkpoint["architecture"]
        self.metadata = metadata
        input_dim = int(metadata["feature_dim"])
        repr_dim = int(metadata["projection_dim"])
        hidden_dim = int(architecture["hidden_dim"])
        dropout = float(architecture["dropout"])
        self.predictor = JEPAPredictor(input_dim, hidden_dim, repr_dim, dropout).to(device)
        self.jepa_head = RiskHead(input_dim + 3 * repr_dim, hidden_dim // 2, dropout).to(device)
        self.direct_head = RiskHead(input_dim, hidden_dim // 2, dropout).to(device)
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.jepa_head.load_state_dict(checkpoint["jepa_head"])
        self.direct_head.load_state_dict(checkpoint["direct_head"])
        self.predictor.eval().requires_grad_(False)
        self.jepa_head.eval().requires_grad_(False)
        self.direct_head.eval().requires_grad_(False)
        self.projector: FrozenProjector | None = None

    def attach_backbone(self, backbone: LLaDA) -> None:
        hidden_size = int(getattr(backbone.model.config, "d_model", 0) or backbone.model.config.hidden_size)
        if hidden_size != int(self.metadata["hidden_size"]):
            raise ValueError("Checkpoint and LLaDA hidden sizes do not match")
        self.projector = FrozenProjector(
            hidden_size,
            int(self.metadata["projection_dim"]),
            int(self.metadata["projection_seed"]),
            backbone.device,
        )


@torch.inference_mode()
def decode(
    backbone: LLaDA,
    question: str,
    cfg: dict,
    method: str,
    step_budget: int,
    policy: AdaptivePolicy | None = None,
) -> tuple[str, dict]:
    if method not in ("vanilla", "adaptive_confidence", "jepa", "direct", "selective_jepa"):
        raise ValueError(f"Unknown decoding method: {method}")
    if method in ("jepa", "direct", "selective_jepa") and policy is None:
        raise ValueError(f"{method} decoding requires a trained policy")
    if method in ("jepa", "direct", "selective_jepa") and policy.projector is None:
        policy.attach_backbone(backbone)
    device = backbone.device
    gcfg = cfg["generation"]
    gen_length = int(gcfg["gen_length"])
    block_length = int(gcfg["block_length"])
    if gen_length % block_length:
        raise ValueError("block_length must divide gen_length")
    num_blocks = gen_length // block_length
    if step_budget % num_blocks:
        raise ValueError("step budget must be divisible by the number of blocks")
    steps_per_block = step_budget // num_blocks
    prompt, prompt_mask = backbone.encode_prompt(question)
    prompt_len = prompt.shape[1]
    response_slice = slice(prompt_len, prompt_len + gen_length)
    attention_mask = extend_attention_mask(prompt_mask, gen_length)
    x = backbone.initial_state(prompt, gen_length)
    stats = DecodeStats()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    total_start = time.perf_counter()

    for block in range(num_blocks):
        block_start = prompt_len + block * block_length
        block_end = block_start + block_length
        for local_step in range(steps_per_block):
            mask = x.eq(backbone.mask_id)
            active_mask = mask.clone()
            active_mask[:, :block_start] = False
            active_mask[:, block_end:] = False
            remaining = int(active_mask.sum().item())
            if remaining == 0:
                break
            _sync(device)
            start = time.perf_counter()
            current = backbone.forward(x, attention_mask, extract_hidden=method in ("jepa", "direct", "selective_jepa"))
            _sync(device)
            stats.llada_seconds += time.perf_counter() - start
            stats.llada_forward_passes += 1
            stats.steps += 1
            forbidden = [int(gcfg["eos_id"])] if gcfg.get("logits_eos_inf", False) else []
            sampled, confidence, margin = proposals(
                current.logits, mask, float(gcfg["temperature"]), forbidden
            )
            if gcfg.get("confidence_eos_eot_inf", False):
                for token_id in (int(gcfg["eos_id"]), int(gcfg["eot_id"])):
                    confidence[sampled.eq(token_id)] = -torch.inf
            steps_left = steps_per_block - local_step
            minimum = math.ceil(remaining / steps_left)
            if method == "vanilla":
                selected = make_candidates(
                    sampled, confidence, margin, active_mask, [minimum], minimum_size=minimum
                )[0]
            else:
                _sync(device)
                policy_start = time.perf_counter()
                is_phase3 = (method in ("jepa", "direct") and
                             gcfg.get("candidate_policy") == "nonprefix_v3")
                if is_phase3:
                    candidates = build_candidates(
                        sampled, confidence, margin, active_mask,
                        list(gcfg["group_sizes"]), minimum,
                        int(gcfg.get("max_candidates", 6)),
                    )
                    # Phase 3.1: ablate the action space WITHOUT regenerating
                    # candidates. Both arms start from the identical capped
                    # candidate list; the prefix arm only removes non-prefixes.
                    candidate_filter = gcfg.get("phase31_candidate_filter", "all")
                    if candidate_filter not in ("all", "prefix"):
                        raise ValueError(f"Unknown phase31_candidate_filter: {candidate_filter}")
                    if candidate_filter == "prefix":
                        available = torch.nonzero(active_mask[0], as_tuple=False).flatten()
                        ranked = available[torch.argsort(
                            confidence[0, available], descending=True, stable=True
                        )]
                        candidates = [
                            c for c in candidates
                            if set(c.positions.tolist()) ==
                               set(ranked[:len(c.positions)].tolist())
                        ]
                        if not candidates:
                            raise RuntimeError("Prefix-only filter removed every candidate")
                else:
                    candidates = make_candidates(
                        sampled, confidence, margin, active_mask,
                        list(gcfg["group_sizes"]), minimum_size=minimum,
                    )
                if method in ("adaptive_confidence", "selective_jepa"):
                    # Both policies start with the same confidence-prefix baseline.
                    threshold = float(gcfg.get("adaptive_confidence_threshold", 0.90))
                    eligible = [c for c in candidates if float(c.confidences.min()) >= threshold]
                    selected = max(eligible, key=lambda c: len(c.positions)) if eligible else candidates[0]
                    if method == "selective_jepa" and len(candidates) > 1:
                        # Selectively override the default, based on the PRE-rank
                        # JEPA's predicted rollout cost, using the EXACT same
                        # prefix candidates as adaptive confidence.
                        stats.intervention_opportunities += 1
                        features = torch.stack([
                            action_features(
                                current, x, c, response_slice, active_mask,
                                backbone.token_embeddings(c.tokens), policy.projector,
                            ) for c in candidates
                        ])
                        predicted = policy.predictor(features)
                        costs = policy.jepa_head(torch.cat([features, predicted], dim=-1))
                        baseline_index = next(i for i, c in enumerate(candidates) if c is selected)
                        best_index = int(costs.argmin().item())
                        advantage = float((costs[baseline_index] - costs[best_index]).item())
                        gate = float(gcfg.get("phase4_intervention_margin", 0.05))
                        stats.candidate_sets_evaluated += len(candidates)
                        if best_index != baseline_index and advantage > gate:
                            selected = candidates[best_index]
                            stats.interventions += 1
                            stats.predicted_advantage_sum += advantage
                else:
                    features = []
                    for candidate in candidates:
                        embeddings = backbone.token_embeddings(candidate.tokens)
                        features.append(
                            action_features(
                                current, x, candidate, response_slice, active_mask,
                                embeddings, policy.projector,
                            )
                        )
                    feature_batch = torch.stack(features)
                    if method == "jepa":
                        predicted = policy.predictor(feature_batch)
                        risk = policy.jepa_head(torch.cat([feature_batch, predicted], dim=-1))
                    else:
                        risk = policy.direct_head(feature_batch)
                    current_confidence = torch.stack([item.confidences.mean() for item in candidates])
                    # Phase 3 heads predict final rollout outcome + compute cost.
                    # Adding confidence here would double-count a separate
                    # heuristic and break alignment with training labels.
                    score = risk if is_phase3 else (
                        risk - float(gcfg["confidence_weight"]) * current_confidence
                    )
                    if is_phase3:
                        stats.candidate_sets_evaluated += len(candidates)
                    selected = candidates[int(score.argmin().item())]
                _sync(device)
                stats.policy_seconds += time.perf_counter() - policy_start
                if method in ("jepa", "direct") and gcfg.get("candidate_policy") == "nonprefix_v3":
                    top = torch.nonzero(active_mask[0], as_tuple=False).flatten()
                    ranked = top[torch.argsort(confidence[0, top], descending=True, stable=True)]
                    prefix = set(int(v) for v in ranked[:len(selected.positions)].tolist())
                    chosen = set(int(v) for v in selected.positions.tolist())
                    stats.nonprefix_selected += int(chosen != prefix)
            x = commit(x, selected)

    _sync(device)
    stats.total_seconds = time.perf_counter() - total_start
    if device.type == "cuda":
        stats.peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
    return backbone.decode_response(x, prompt_len), asdict(stats)
