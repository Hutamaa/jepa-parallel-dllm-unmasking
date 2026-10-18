from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .llada import BackboneOutput, Candidate


class FrozenProjector:
    """Deterministic Gaussian projection shared by data generation and inference."""

    def __init__(self, hidden_size: int, output_size: int, seed: int, device: torch.device):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn(hidden_size, output_size, generator=generator, dtype=torch.float32)
        self.matrix = (matrix / math.sqrt(output_size)).to(device)

    @property
    def output_size(self) -> int:
        return self.matrix.shape[1]

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value.float() @ self.matrix, (self.output_size,))


def _mean_or_zero(values: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    return values.mean(dim=0) if values.numel() else torch.zeros(width, device=device)


def _position_features(relative: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant sinusoidal encoding of the selected position set."""
    frequencies = (2.0 ** torch.arange(8, device=relative.device)) * math.pi
    angles = relative.float().unsqueeze(-1) * frequencies
    return torch.cat([angles.sin().mean(dim=0), angles.cos().mean(dim=0)], dim=0)


def action_features(
    output: BackboneOutput,
    x: torch.Tensor,
    candidate: Candidate,
    response_slice: slice,
    active_mask: torch.Tensor,
    proposed_embeddings: torch.Tensor,
    projector: FrozenProjector,
) -> torch.Tensor:
    if output.hidden is None:
        raise ValueError("Action features require extracted backbone hidden states")
    hidden = output.hidden[0]
    response_hidden = hidden[response_slice]
    masked_hidden = hidden[active_mask[0]]
    selected_hidden = hidden[candidate.positions]
    response_len = response_hidden.shape[0]
    remaining = int(active_mask.sum().item())
    relative = (candidate.positions - int(response_slice.start)) / max(response_len - 1, 1)
    scalars = torch.tensor(
        [
            len(candidate.positions) / max(remaining, 1),
            remaining / max(response_len, 1),
            1.0 - remaining / max(response_len, 1),
            candidate.confidences.mean().item(),
            candidate.confidences.min().item(),
            candidate.confidences.std(unbiased=False).item(),
            candidate.margins.mean().item(),
            relative.mean().item(),
            relative.std(unbiased=False).item(),
            math.log2(len(candidate.positions) + 1) / math.log2(response_len + 1),
        ],
        device=hidden.device,
        dtype=torch.float32,
    )
    pooled = [
        _mean_or_zero(response_hidden, hidden.shape[-1], hidden.device),
        _mean_or_zero(masked_hidden, hidden.shape[-1], hidden.device),
        _mean_or_zero(selected_hidden, hidden.shape[-1], hidden.device),
        _mean_or_zero(proposed_embeddings, hidden.shape[-1], hidden.device),
    ]
    return torch.cat(
        [*(projector(item) for item in pooled), _position_features(relative), scalars], dim=0
    )


def target_representation(
    next_output: BackboneOutput,
    next_x: torch.Tensor,
    candidate: Candidate,
    response_slice: slice,
    mask_id: int,
    projector: FrozenProjector,
) -> torch.Tensor:
    if next_output.hidden is None:
        raise ValueError("JEPA targets require extracted backbone hidden states")
    hidden = next_output.hidden[0]
    response_hidden = hidden[response_slice]
    remaining_mask = next_x[0].eq(mask_id)
    values = [
        response_hidden.mean(dim=0),
        _mean_or_zero(hidden[remaining_mask], hidden.shape[-1], hidden.device),
        hidden[candidate.positions].mean(dim=0),
    ]
    return torch.cat([projector(item) for item in values], dim=0)


def counterfactual_damage(
    current: BackboneOutput,
    future: BackboneOutput,
    current_proposals: torch.Tensor,
    remaining_mask: torch.Tensor,
    retention_weight: float,
    flip_weight: float,
) -> torch.Tensor:
    """Prediction instability, not entropy: loss of proposal mass plus argmax flips."""
    if not remaining_mask.any():
        return torch.zeros((), device=current.logits.device)
    old_tokens = current_proposals[remaining_mask]
    future_logits = future.logits[remaining_mask].float()
    retained_prob = F.softmax(future_logits, dim=-1).gather(
        -1, old_tokens.unsqueeze(-1)
    ).squeeze(-1)
    flip = future_logits.argmax(dim=-1).ne(old_tokens).float()
    return retention_weight * (1.0 - retained_prob.mean()) + flip_weight * flip.mean()


def commitment_shift_risk(
    current: BackboneOutput,
    future: BackboneOutput,
    selected_positions: torch.Tensor,
    selected_confidence: torch.Tensor,
    remaining_mask: torch.Tensor,
    commitment_weight: float = 0.75,
    shift_weight: float = 0.25,
) -> torch.Tensor:
    """Teacher-free, non-oracle risk *proxy* for a candidate action.

    A group's probability of being wrong is approximated by the mean current
    token error probability (1 - p_selected). The second term is bounded
    Jensen-Shannon divergence on still-masked positions, using before/after
    token distributions. Distribution shift is *not* necessarily an error.

    Crucially, the terminal action retains the commitment component rather
    than receiving a free zero-risk label. This is NOT final GSM8K correctness.
    """
    del selected_positions  # passed for clear action association in callers
    p_selected = selected_confidence.float().clamp(min=0.0, max=1.0)
    if not p_selected.numel():
        raise ValueError("Cannot score an empty candidate action")
    commitment_risk = (1.0 - p_selected).mean()
    if not bool(remaining_mask.any()):
        return commitment_weight * commitment_risk

    p_log = F.log_softmax(current.logits[remaining_mask].float(), dim=-1)
    q_log = F.log_softmax(future.logits[remaining_mask].float(), dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    m_log = torch.logaddexp(p_log, q_log) - math.log(2.0)
    js = 0.5 * (p * (p_log - m_log)).sum(dim=-1)
    js += 0.5 * (q * (q_log - m_log)).sum(dim=-1)
    shift = (js.mean() / math.log(2.0)).clamp(min=0.0, max=1.0)
    return commitment_weight * commitment_risk + shift_weight * shift
