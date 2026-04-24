# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hidden-layer FFT auxiliary loss for [`GRPOTrainer`].

This is the regularizer referred to in the paper ``Anxiety Treatment for Large Language Models: Fine Tuning with
Hidden Layer FFT'' as ``\\fftaux{}``. It pulls wrong-completion hidden-state spectra toward the in-group
correct-completion prototype. It plugs into ``GRPOTrainer`` via the ``fft_trace_aux_*`` fields on
[`GRPOConfig`].

The module exposes a single entrypoint, [`compute_fft_aux_loss`], designed to be called from
``GRPOTrainer._compute_loss`` when ``args.fft_trace_aux_lambda > 0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pivot_aux_gate_decision(
    *,
    bad_corr_idx: bool,
    n_valid: int,
    n_correct: int,
    n_wrong: int,
    min_valid: int,
    min_ratio: float,
) -> tuple[bool, str]:
    """Return ``(allow_aux, reason)`` for the pivot-aware gating policy."""
    if bad_corr_idx:
        return False, "bad_corr_idx"
    if n_valid < max(int(min_valid), 1):
        return False, "too_few_valid"
    if n_correct == 0:
        return False, "all_wrong"
    if n_wrong == 0:
        return False, "all_correct"
    p_c = n_correct / n_valid
    p_w = n_wrong / n_valid
    if p_c < float(min_ratio):
        return False, "sparse_correct"
    if p_w < float(min_ratio):
        return False, "sparse_wrong"
    return True, "ok"


def _completion_correctness_masks(
    rewards_per_func: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    corr_idx: int,
    corr_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lens = completion_mask.float().sum(dim=1)
    is_valid = lens > 0
    if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
        is_correct = torch.zeros_like(is_valid, dtype=torch.bool)
        wrong = is_valid & ~is_correct
        return is_valid, is_correct, wrong
    correctness = rewards_per_func[:, corr_idx]
    is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid
    wrong = ~is_correct & is_valid
    return is_valid, is_correct, wrong


def _fft_trace_features(last_hidden: torch.Tensor, token_mask: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Per-sequence FFT trace features from completion hidden states, pooled to ``num_bins`` and L2-normalized."""
    # Spectral feature extraction in fp32 for numerical stability.
    x = last_hidden.float() * token_mask.unsqueeze(-1).float()
    z = torch.fft.rfft(x, dim=1)
    power = z.real.square() + z.imag.square()  # [B, F, H]
    feat = torch.log1p(power.mean(dim=2))  # [B, F]
    # Sequence lengths vary, so FFT frequency bins vary too. Pool to fixed width for stable prototypes.
    feat = F.adaptive_avg_pool1d(feat.unsqueeze(1), output_size=max(int(num_bins), 1)).squeeze(1)
    return F.normalize(feat, p=2, dim=1, eps=1e-8)


def _fft_aux_pair_loss(wrong_feat: torch.Tensor, proto: torch.Tensor, *, distance: str) -> torch.Tensor:
    if distance == "cosine":
        proto_n = F.normalize(proto.unsqueeze(0), p=2, dim=1, eps=1e-8)
        wrong_n = F.normalize(wrong_feat, p=2, dim=1, eps=1e-8)
        return (1.0 - (wrong_n * proto_n).sum(dim=1)).mean()
    if distance == "mse":
        return F.mse_loss(wrong_feat, proto.unsqueeze(0).expand_as(wrong_feat))
    raise ValueError(f"Unknown fft_trace_aux_distance={distance!r}; expected 'cosine' or 'mse'.")


def _completion_last_hidden(
    trainer,
    model,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Forward pass returning (last_layer_completion_hidden, token_mask, rewards_per_func)."""
    rewards_per_func = inputs.get("rewards_per_func")
    if rewards_per_func is None or rewards_per_func.numel() == 0:
        return None

    completion_ids = inputs["completion_ids"]
    completion_mask = inputs["completion_mask"]
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    token_mask = completion_mask if "tool_mask" not in inputs else (completion_mask * inputs["tool_mask"])
    fn = int(getattr(trainer.args, "fft_trace_aux_first_n_completion_tokens", 0) or 0)
    if fn > 0:
        cum = token_mask.long().cumsum(dim=1)
        token_mask = token_mask * (cum <= fn).to(dtype=token_mask.dtype)

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    model_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_attention_mask",
        "image_sizes",
        "token_type_ids",
        "mm_token_type_ids",
        "pixel_position_ids",
    ):
        if key in inputs:
            model_kwargs[key] = inputs[key]

    outputs = model(**model_kwargs)
    last_hidden = outputs.hidden_states[-1][:, -completion_ids.size(1) :, :]
    return last_hidden, token_mask, rewards_per_func


def _pivot_aux_evaluate(trainer, inputs: dict[str, torch.Tensor]) -> tuple[bool, str, dict[str, float]]:
    """Decide whether the FFT aux may run this step; also emit stats for logging."""
    rewards = inputs.get("rewards_per_func")
    cm = inputs.get("completion_mask")
    if rewards is None or rewards.numel() == 0 or cm is None:
        return False, "no_rewards", {}
    corr_idx = int(getattr(trainer.args, "correctness_reward_func_index", 0))
    corr_thr = float(getattr(trainer.args, "correctness_threshold", 0.5))
    is_valid, is_correct, wrong = _completion_correctness_masks(rewards, cm, corr_idx=corr_idx, corr_thr=corr_thr)
    n_valid = int(is_valid.sum().item())
    n_correct = int(is_correct.sum().item())
    n_wrong = int(wrong.sum().item())
    p_w = float(n_wrong) / float(n_valid) if n_valid > 0 else 0.0
    p_c = float(n_correct) / float(n_valid) if n_valid > 0 else 0.0
    stats = {
        "n_valid": float(n_valid),
        "n_correct": float(n_correct),
        "n_wrong": float(n_wrong),
        "p_wrong": p_w,
        "p_correct": p_c,
    }
    if not bool(getattr(trainer.args, "pivot_aux_enable_hard_gate", True)):
        return True, "disabled", stats
    bad = corr_idx < 0 or corr_idx >= rewards.shape[1]
    ok, reason = pivot_aux_gate_decision(
        bad_corr_idx=bad,
        n_valid=n_valid,
        n_correct=n_correct,
        n_wrong=n_wrong,
        min_valid=int(getattr(trainer.args, "pivot_aux_min_valid_samples", 4)),
        min_ratio=float(getattr(trainer.args, "pivot_aux_min_class_ratio", 0.1)),
    )
    return ok, reason, stats


def _log_pivot_metrics(trainer, mode: str, allowed: bool, reason: str, stats: dict[str, float]) -> None:
    m = trainer._metrics[mode]
    m.setdefault("grpo/pivot_valid_count", []).append(stats.get("n_valid", 0.0))
    m.setdefault("grpo/pivot_correct_count", []).append(stats.get("n_correct", 0.0))
    m.setdefault("grpo/pivot_wrong_count", []).append(stats.get("n_wrong", 0.0))
    m.setdefault("grpo/pivot_wrong_ratio", []).append(stats.get("p_wrong", 0.0))
    m.setdefault("grpo/pivot_has_pivot", []).append(1.0 if allowed else 0.0)


def compute_fft_aux_loss(trainer, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the hidden-layer FFT auxiliary loss (unscaled).

    Pulls wrong-completion FFT features toward the correct-completion prototype. The prototype is drawn in order
    of preference: (1) within the current GRPO group, (2) batch-level correct mean, (3) EMA memory across steps.
    When no correct/wrong pivot exists in the batch, returns a zero loss.

    This function mutates ``trainer._fft_correct_proto`` / ``trainer._fft_correct_proto_updates`` to maintain the
    EMA memory. Call it from ``GRPOTrainer._compute_loss`` when ``trainer.args.fft_trace_aux_lambda > 0``.
    """
    device = inputs["completion_ids"].device
    zero = torch.zeros((), device=device)

    mode = "train" if trainer.model.training else "eval"

    # Lazy state init so we don't touch GRPOTrainer.__init__.
    if not hasattr(trainer, "_fft_correct_proto"):
        trainer._fft_correct_proto = None
        trainer._fft_correct_proto_updates = 0

    pivot_ok, pivot_reason, pivot_stats = _pivot_aux_evaluate(trainer, inputs)
    _log_pivot_metrics(trainer, mode, pivot_ok, pivot_reason, pivot_stats)
    if not pivot_ok:
        return zero

    fwd = _completion_last_hidden(trainer, model, inputs)
    if fwd is None:
        return zero
    last_hidden, token_mask, rewards_per_func = fwd

    num_bins = int(getattr(trainer.args, "fft_trace_aux_num_bins", 32))
    feat = _fft_trace_features(last_hidden, token_mask, num_bins=num_bins)

    corr_idx = int(getattr(trainer.args, "correctness_reward_func_index", 0))
    corr_thr = float(getattr(trainer.args, "correctness_threshold", 0.5))
    if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
        return zero

    correctness = rewards_per_func[:, corr_idx]
    is_valid = token_mask.sum(dim=1) > 0
    is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

    n = feat.shape[0]
    num_generations = trainer.num_generations if trainer.model.training else trainer.num_generations_eval
    detach_correct = bool(getattr(trainer.args, "fft_trace_aux_detach_correct", True))
    use_batch_fallback = bool(getattr(trainer.args, "fft_trace_aux_batch_fallback", True))
    distance = str(getattr(trainer.args, "fft_trace_aux_distance", "cosine"))
    proto_momentum = float(getattr(trainer.args, "fft_trace_aux_proto_momentum", 0.95))
    proto_warmup = int(getattr(trainer.args, "fft_trace_aux_proto_warmup_correct", 1))

    losses = []
    can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
    if can_use_contiguous_grouping:
        n_groups = n // num_generations
        feat_g = feat.view(n_groups, num_generations, -1)
        correct_g = is_correct.view(n_groups, num_generations)
        for i in range(n_groups):
            c = correct_g[i]
            w = ~c
            if bool(c.any()) and bool(w.any()):
                proto = feat_g[i][c].mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                losses.append(_fft_aux_pair_loss(feat_g[i][w], proto, distance=distance))

    if not losses and use_batch_fallback:
        all_correct = feat[is_correct]
        all_wrong = feat[~is_correct]
        if all_correct.numel() > 0 and all_wrong.numel() > 0:
            proto = all_correct.mean(dim=0)
            if detach_correct:
                proto = proto.detach()
            losses.append(_fft_aux_pair_loss(all_wrong, proto, distance=distance))

    # Cross-step prototype memory: update with any current-step correct features, even if we already have a loss
    # to return. The memory is used as a last-resort fallback on the next step when in-batch pairing fails.
    current_correct = feat[is_correct]
    if current_correct.numel() > 0:
        cur_proto = current_correct.mean(dim=0).detach()
        if trainer._fft_correct_proto is None:
            trainer._fft_correct_proto = cur_proto
        else:
            m = min(max(proto_momentum, 0.0), 0.9999)
            trainer._fft_correct_proto = F.normalize(
                m * trainer._fft_correct_proto + (1.0 - m) * cur_proto,
                p=2,
                dim=0,
                eps=1e-8,
            )
        trainer._fft_correct_proto_updates += 1

    if (
        not losses
        and trainer._fft_correct_proto is not None
        and trainer._fft_correct_proto_updates >= max(proto_warmup, 1)
    ):
        all_wrong = feat[~is_correct]
        if all_wrong.numel() > 0:
            losses.append(_fft_aux_pair_loss(all_wrong, trainer._fft_correct_proto, distance=distance))

    trainer._metrics[mode].setdefault("grpo/fft_aux_proto_updates", []).append(float(trainer._fft_correct_proto_updates))
    if not losses:
        return zero
    return torch.stack(losses).mean()
