"""Counterfactual full-completion outcome labels, computed *offline* only.

Uses training/validation gold answers solely for the offline reward.  The
inference-time controller never receives a reference answer or rollout output.
"""
from __future__ import annotations

import math

import torch

from .data import extract_answer
from .llada import LLaDA, Candidate, BackboneOutput, commit, make_candidates, proposals


def _propose(backbone: LLaDA, cfg: dict, logits: torch.Tensor, mask: torch.Tensor):
    gcfg = cfg["generation"]
    forbidden = [int(gcfg["eos_id"])] if gcfg.get("logits_eos_inf", False) else []
    sampled, confidence, margin = proposals(logits, mask, float(gcfg["temperature"]), forbidden)
    if gcfg.get("confidence_eos_eot_inf", False):
        for token_id in (int(gcfg["eos_id"]), int(gcfg["eot_id"])):
            confidence[sampled.eq(token_id)] = -torch.inf
    return sampled, confidence, margin


@torch.inference_mode()
def complete_after_action(backbone: LLaDA, cfg: dict, x: torch.Tensor,
                          attention_mask: torch.Tensor, prompt_len: int,
                          candidate: Candidate, step_index: int,
                          total_steps: int,
                          first_continuation_output: BackboneOutput | None = None) -> tuple[str, int]:
    """Commit action, then finish with the same vanilla policy as Phase 1.

    Supports one generation block (the Phase 3 configuration uses 128/128).
    Exactly one deterministic continuation rollout per action: this estimates
    that trajectory's outcome, not the expectation under stochastic decoding.
    """
    gcfg = cfg["generation"]
    gen_length = int(gcfg["gen_length"])
    if int(gcfg["block_length"]) != gen_length:
        raise ValueError("Phase 3 outcome rollout currently requires a single generation block")
    state = commit(x, candidate)
    passes = 0
    for step in range(step_index + 1, total_steps):
        active = state.eq(backbone.mask_id)
        active[:, :prompt_len] = False
        active[:, prompt_len + gen_length:] = False
        remaining = int(active.sum().item())
        if remaining == 0:
            break
        if first_continuation_output is not None:
            current = first_continuation_output
            first_continuation_output = None
        else:
            current = backbone.forward(state, attention_mask, extract_hidden=False)
        passes += 1
        mask = state.eq(backbone.mask_id)
        sampled, confidence, margin = _propose(backbone, cfg, current.logits, mask)
        required = math.ceil(remaining / (total_steps - step))
        selected = make_candidates(sampled, confidence, margin, active,
                                   [required], minimum_size=required)[0]
        state = commit(state, selected)
    active_final = state[:, prompt_len:prompt_len + gen_length].eq(backbone.mask_id)
    if bool(active_final.any()):
        raise RuntimeError("Vanilla counterfactual continuation did not finish its generation budget")
    return backbone.decode_response(state, prompt_len), passes


def outcome_cost(correct: bool, continuation_passes: int,
                 remaining_budget: int, latency_weight: float) -> float:
    """Lower is better. The current forward pass is shared among actions.

    Negative correctness dominates.  Small normalized rollout-pass penalty
    breaks ties by preferring decisions which complete with fewer forwards.
    """
    return -float(correct) + latency_weight * (continuation_passes / max(remaining_budget, 1))
