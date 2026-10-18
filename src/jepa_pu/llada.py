from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    hidden: torch.Tensor | None


@dataclass
class Candidate:
    positions: torch.Tensor
    tokens: torch.Tensor
    confidences: torch.Tensor
    margins: torch.Tensor


class LLaDA:
    """Frozen LLaDA wrapper with logits and hidden-state extraction."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        model_path = cfg["paths"]["model"]
        mcfg = cfg["model"]
        dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
        self.device = torch.device(mcfg.get("device", "cuda"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=mcfg.get("trust_remote_code", True),
            local_files_only=mcfg.get("local_files_only", True),
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=mcfg.get("trust_remote_code", True),
            local_files_only=mcfg.get("local_files_only", True),
            torch_dtype=dtype,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        self.mask_id = int(mcfg["mask_id"])
        if self.tokenizer.pad_token_id == self.mask_id:
            raise ValueError("LLaDA mask and tokenizer padding IDs must differ")

    def format_prompt(self, question: str) -> str:
        content = (
            "Solve the following math problem. Show your reasoning, then write the final "
            "numeric answer after '####'.\n\n" + question
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def encode_prompt(self, question: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            self.format_prompt(question), add_special_tokens=False, return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def initial_state(self, prompt: torch.Tensor, gen_length: int) -> torch.Tensor:
        suffix = torch.full(
            (prompt.shape[0], gen_length), self.mask_id, dtype=torch.long, device=self.device
        )
        return torch.cat([prompt, suffix], dim=1)

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, extract_hidden: bool = True
    ) -> BackboneOutput:
        out = self.model(
            x,
            attention_mask=attention_mask,
            output_hidden_states=extract_hidden,
        )
        hidden_states = getattr(out, "hidden_states", None)
        if extract_hidden and hidden_states is None:
            raise RuntimeError("LLaDA did not return hidden states with output_hidden_states=True")
        hidden = None if hidden_states is None else hidden_states[int(self.cfg["model"].get("hidden_layer", -1))]
        return BackboneOutput(logits=out.logits, hidden=hidden)

    def token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedding = getattr(getattr(self.model, "transformer", None), "wte", None)
        if embedding is None:
            embedding = getattr(getattr(self.model, "model", None), "embed_tokens", None)
        if embedding is None and hasattr(self.model, "get_input_embeddings"):
            embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise RuntimeError("Could not locate LLaDA input embeddings")
        return embedding(token_ids)

    def decode_response(self, x: torch.Tensor, prompt_length: int) -> str:
        return self.tokenizer.decode(x[0, prompt_length:], skip_special_tokens=True)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    uniform = torch.rand_like(logits64)
    return logits64.exp() / (-torch.log(uniform)).pow(temperature)


def proposals(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.0,
    forbidden_ids: Iterable[int] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score_logits = logits.clone()
    for token_id in forbidden_ids:
        score_logits[..., int(token_id)] = -torch.inf
    sampled = add_gumbel_noise(score_logits, temperature).argmax(dim=-1)
    log_probs = F.log_softmax(score_logits.float(), dim=-1)
    confidence = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).exp()
    top2 = torch.topk(log_probs, k=2, dim=-1).values.exp()
    margin = top2[..., 0] - top2[..., 1]
    confidence = torch.where(mask, confidence, torch.full_like(confidence, -torch.inf))
    return sampled, confidence, margin


def make_candidates(
    sampled: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
    active_mask: torch.Tensor,
    group_sizes: list[int],
    minimum_size: int = 1,
    max_candidates: int | None = None,
) -> list[Candidate]:
    """Build a linear number of confidence-prefix candidates, never subset enumeration."""
    positions = torch.nonzero(active_mask[0], as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    ranked = positions[torch.argsort(confidence[0, positions], descending=True)]
    sizes = sorted({min(ranked.numel(), max(minimum_size, int(k))) for k in group_sizes})
    if ranked.numel() not in sizes and minimum_size >= ranked.numel():
        sizes.append(ranked.numel())
    candidates = [
        Candidate(
            positions=ranked[:k],
            tokens=sampled[0, ranked[:k]],
            confidences=confidence[0, ranked[:k]],
            margins=margin[0, ranked[:k]],
        )
        for k in sizes
    ]
    return candidates[:max_candidates] if max_candidates else candidates


def commit(x: torch.Tensor, candidate: Candidate) -> torch.Tensor:
    next_x = x.clone()
    next_x[0, candidate.positions] = candidate.tokens
    return next_x


def extend_attention_mask(prompt_mask: torch.Tensor, gen_length: int) -> torch.Tensor:
    suffix = torch.ones(
        (prompt_mask.shape[0], gen_length), dtype=prompt_mask.dtype, device=prompt_mask.device
    )
    return torch.cat([prompt_mask, suffix], dim=-1)
