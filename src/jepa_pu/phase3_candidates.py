"""Small deterministic action space with genuine non-prefix unmasking groups.

All actions use frozen LLaDA's proposed tokens.  No gold labels or extra
LLaDA forward passes are used to construct inference-time candidates.
"""
from __future__ import annotations

import torch

from .llada import Candidate


def build_candidates(sampled: torch.Tensor, confidence: torch.Tensor,
                     margin: torch.Tensor, active_mask: torch.Tensor,
                     group_sizes: list[int], minimum_size: int,
                     max_candidates: int = 6) -> list[Candidate]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    indices = torch.nonzero(active_mask[0], as_tuple=False).flatten()
    n = indices.numel()
    if n == 0:
        return []
    ranked = indices[torch.argsort(confidence[0, indices], descending=True, stable=True)]
    sizes = sorted({min(n, max(minimum_size, int(k))) for k in [minimum_size, *group_sizes]})
    unique: dict[tuple[int, ...], Candidate] = {}

    def add(pos: torch.Tensor) -> None:
        if len(unique) >= max_candidates:
            return
        key = tuple(sorted(int(z) for z in pos.tolist()))
        if key in unique:
            return
        unique[key] = Candidate(
            positions=pos,
            tokens=sampled[0, pos],
            confidences=confidence[0, pos], margins=margin[0, pos],
        )

    # Interleave both kinds of groups at each size to ensure that truncation
    # at max_candidates does not drop all non-prefix alternatives.
    for size in sizes:
        if len(unique) >= max_candidates:
            break
        add(ranked[:size])
        if size >= n:
            continue
        # Spread among the 2k most confident proposals, often skipping ranks
        # 2,4,... . This is not a confidence-ranked prefix.
        pool = ranked[:min(n, 2 * size)]
        alternating = torch.cat((pool[::2], pool[1::2]))[:size]
        add(alternating)

    return list(unique.values())
