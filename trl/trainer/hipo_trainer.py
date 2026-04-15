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

"""HiPO trainer — primary implementation entry for HiPO.

Subclasses [`GRPOTrainer`]: generation and GRPO loss are unchanged; HiPO only overrides
`_compute_grpo_group_advantages` to add optional verifier-only wrong-set length shaping (see [`HiPOConfig`]).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .grpo_trainer import GRPOTrainer
from .utils import entropy_from_logits, nanstd

_EXP39_SPECIAL_ARMS = frozenset({"a2", "a3", "a4", "a5", "a6", "a7", "a8"})


class HiPOTrainer(GRPOTrainer):
    """[`GRPOTrainer`] + representation auxiliaries (FFT / hidden-direction / exp arms)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fft_aux_lambda = float(getattr(self.args, "fft_trace_aux_lambda", 0.0) or 0.0)
        dir_aux_lambda = float(getattr(self.args, "hidden_direction_aux_lambda", 0.0) or 0.0)
        exp39_arm = str(getattr(self.args, "exp39_arm", "") or "").strip().lower()
        if fft_aux_lambda <= 0.0 and dir_aux_lambda <= 0.0 and exp39_arm != "a0":
            raise ValueError(
                "HiPOTrainer requires at least one active auxiliary: "
                "fft_trace_aux_lambda > 0 or hidden_direction_aux_lambda > 0 "
                "(unless exp39_arm=a0 for a no-aux control)."
            )
        self._fft_correct_proto: torch.Tensor | None = None
        self._fft_correct_proto_updates: int = 0
        self._dir_correct_proto: torch.Tensor | None = None
        self._dir_correct_proto_updates: int = 0
        self._exp39_spec_proto: torch.Tensor | None = None
        self._exp39_spec_proto_updates: int = 0
        self._exp39_len_proto: list[torch.Tensor | None] = [None, None, None]
        self._exp40_last_freq_lowbin_mean: torch.Tensor | None = None

    @staticmethod
    def pivot_aux_gate_decision(
        *,
        bad_corr_idx: bool,
        n_valid: int,
        n_correct: int,
        n_wrong: int,
        min_valid: int,
        min_ratio: float,
    ) -> tuple[bool, str]:
        """Return (allow_aux, reason). Used for tests and HiPO pivot-aware aux gating."""
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

    @staticmethod
    def _completion_correctness_masks(
        rewards_per_func: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        corr_idx: int,
        corr_thr: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Valid completions, correct mask, wrong-among-valid mask (same semantics as existing aux code)."""
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

    def _pivot_aux_evaluate(self, inputs: dict[str, torch.Tensor]) -> tuple[bool, str, dict[str, float]]:
        """Whether HiPO extras may run this step; stats for logging."""
        rewards = inputs.get("rewards_per_func")
        cm = inputs.get("completion_mask")
        if rewards is None or rewards.numel() == 0 or cm is None:
            return False, "no_rewards", {}
        corr_idx, corr_thr = self._corr_spec()
        is_valid, is_correct, wrong = self._completion_correctness_masks(
            rewards, cm, corr_idx=corr_idx, corr_thr=corr_thr
        )
        n_valid = int(is_valid.sum().item())
        n_correct = int(is_correct.sum().item())
        n_wrong = int(wrong.sum().item())
        p_w = float(n_wrong) / float(n_valid) if n_valid > 0 else 0.0
        p_c = float(n_correct) / float(n_valid) if n_valid > 0 else 0.0
        stats: dict[str, float] = {
            "n_valid": float(n_valid),
            "n_correct": float(n_correct),
            "n_wrong": float(n_wrong),
            "p_wrong": p_w,
            "p_correct": p_c,
        }
        if not bool(getattr(self.args, "pivot_aux_enable_hard_gate", True)):
            return True, "disabled", stats
        bad = corr_idx < 0 or corr_idx >= rewards.shape[1]
        ok, reason = self.pivot_aux_gate_decision(
            bad_corr_idx=bad,
            n_valid=n_valid,
            n_correct=n_correct,
            n_wrong=n_wrong,
            min_valid=int(getattr(self.args, "pivot_aux_min_valid_samples", 4)),
            min_ratio=float(getattr(self.args, "pivot_aux_min_class_ratio", 0.1)),
        )
        return ok, reason, stats

    def _log_pivot_aux_metrics(self, mode: str, allowed: bool, reason: str, stats: dict[str, float]) -> None:
        m = self._metrics[mode]
        m.setdefault("hipo/pivot_valid_count", []).append(stats.get("n_valid", 0.0))
        m.setdefault("hipo/pivot_correct_count", []).append(stats.get("n_correct", 0.0))
        m.setdefault("hipo/pivot_wrong_count", []).append(stats.get("n_wrong", 0.0))
        m.setdefault("hipo/pivot_wrong_ratio", []).append(stats.get("p_wrong", 0.0))
        m.setdefault("hipo/pivot_has_pivot", []).append(1.0 if allowed else 0.0)
        m.setdefault("hipo/pivot_aux_skip_reason", []).append(reason)
        for tag in (
            "ok",
            "disabled",
            "no_rewards",
            "bad_corr_idx",
            "too_few_valid",
            "all_correct",
            "all_wrong",
            "sparse_correct",
            "sparse_wrong",
        ):
            m.setdefault(f"hipo/pivot_skip_{tag}", []).append(1.0 if reason == tag else 0.0)

    def _corr_spec(self) -> tuple[int, float]:
        return int(getattr(self.args, "correctness_reward_func_index", 0)), float(
            getattr(self.args, "correctness_threshold", 0.5)
        )

    def _compute_grpo_group_advantages(
        self,
        rewards_per_func: torch.Tensor,
        completion_mask: torch.Tensor,
        num_generations: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Wrong-only length shaping was removed from HiPO. Length is controlled via token budgets
        # (e.g., `max_completion_length` / domain budgets at eval) instead.
        return super()._compute_grpo_group_advantages(
            rewards_per_func=rewards_per_func,
            completion_mask=completion_mask,
            num_generations=num_generations,
            device=device,
        )

    def _aux_completion_last_hidden(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        *,
        first_n_field: str = "fft",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Forward pass for aux losses: last-layer completion hidden states + mask + rewards matrix."""
        rewards_per_func = inputs.get("rewards_per_func")
        if rewards_per_func is None or rewards_per_func.numel() == 0:
            return None

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        token_mask = completion_mask if "tool_mask" not in inputs else (completion_mask * inputs["tool_mask"])
        if first_n_field == "dir":
            fn = int(getattr(self.args, "hidden_direction_aux_first_n_completion_tokens", 0) or 0)
        else:
            fn = int(getattr(self.args, "fft_trace_aux_first_n_completion_tokens", 0) or 0)
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

    def _aux_completion_hidden_logits_entropy(
        self, model, inputs: dict[str, torch.Tensor], *, first_n_field: str = "dir"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Like `_aux_completion_last_hidden` but also returns completion logits + per-seq mean entropy."""
        rewards_per_func = inputs.get("rewards_per_func")
        if rewards_per_func is None or rewards_per_func.numel() == 0:
            return None

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        token_mask = completion_mask if "tool_mask" not in inputs else (completion_mask * inputs["tool_mask"])
        if first_n_field == "dir":
            fn = int(getattr(self.args, "hidden_direction_aux_first_n_completion_tokens", 0) or 0)
        else:
            fn = int(getattr(self.args, "fft_trace_aux_first_n_completion_tokens", 0) or 0)
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
        logits = outputs.logits[:, -completion_ids.size(1) :, :]
        ent = entropy_from_logits(logits.float())
        m = token_mask.unsqueeze(-1).float()
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        mean_ent = (ent * m).sum(dim=1) / denom.squeeze(-1)
        return last_hidden, token_mask, rewards_per_func, logits, mean_ent

    @staticmethod
    def _fft_trace_features(last_hidden: torch.Tensor, token_mask: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Return per-sequence FFT trace features from completion hidden states."""
        # Do spectral feature extraction in fp32 for numerical stability.
        x = last_hidden.float() * token_mask.unsqueeze(-1).float()
        z = torch.fft.rfft(x, dim=1)
        power = z.real.square() + z.imag.square()  # [B, F, H]
        feat = torch.log1p(power.mean(dim=2))  # [B, F]
        # Sequence lengths vary, so FFT frequency bins vary too. Pool to fixed width for stable prototypes.
        feat = F.adaptive_avg_pool1d(feat.unsqueeze(1), output_size=max(int(num_bins), 1)).squeeze(1)
        return F.normalize(feat, p=2, dim=1, eps=1e-8)

    @staticmethod
    def _fft_mag_trace_features(last_hidden: torch.Tensor, token_mask: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Exp39 a2: log-magnitude spectrum pooled across heads (real FFT, no log-power compression)."""
        x = last_hidden.float() * token_mask.unsqueeze(-1).float()
        z = torch.fft.rfft(x, dim=1)
        mag = torch.sqrt(z.real.square() + z.imag.square()).mean(dim=2)
        feat = F.adaptive_avg_pool1d(mag.unsqueeze(1), output_size=max(int(num_bins), 1)).squeeze(1)
        return F.normalize(F.log1p(feat), p=2, dim=1, eps=1e-8)

    @staticmethod
    def _fft_phase_trace_features(last_hidden: torch.Tensor, token_mask: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Exp39 a3: cos/sin embedding of mean complex angle per frequency bin."""
        x = last_hidden.float() * token_mask.unsqueeze(-1).float()
        z = torch.fft.rfft(x, dim=1)
        agg = z.mean(dim=2)
        phase = torch.angle(agg)
        ph = F.adaptive_avg_pool1d(phase.unsqueeze(1), output_size=max(int(num_bins), 1)).squeeze(1)
        stacked = torch.cat([torch.cos(ph), torch.sin(ph)], dim=1)
        return F.normalize(stacked, p=2, dim=1, eps=1e-8)

    def _fft_aux_pair_loss(self, wrong_feat: torch.Tensor, proto: torch.Tensor, distance: str) -> torch.Tensor:
        if distance == "mse":
            return (wrong_feat - proto.unsqueeze(0)).square().mean()
        if distance != "cosine":
            raise ValueError(f"Unknown fft_trace_aux_distance={distance!r}; expected 'cosine' or 'mse'.")
        # 1 - cosine similarity, bounded and less scale-sensitive than MSE.
        cos = F.cosine_similarity(wrong_feat, proto.unsqueeze(0), dim=1, eps=1e-8)
        return (1.0 - cos).mean()

    def _compute_pivot_shake_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp70: entropy-gated hidden-state perturbation + FFT prototype pull.

        Injects gaussian noise at the top-k% highest-entropy token positions of wrong traces
        before FFT feature extraction. This adds diversity at 'confused' decision points
        without disturbing correct-trace prototypes.
        """
        fwd = self._aux_completion_hidden_logits_entropy(model, inputs, first_n_field="fft")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func, logits, _mean_ent = fwd

        pivot_topk = float(getattr(self.args, "pivot_shake_topk", 0.05))
        pivot_scale = float(getattr(self.args, "pivot_shake_scale", 0.05))

        # Per-token entropy [B, L]
        ent = entropy_from_logits(logits.float())  # [B, L]

        # Build pivot mask: top-k% highest-entropy valid positions per sequence.
        # Mask out padding positions by setting them to -inf before topk selection.
        ent_masked = ent * token_mask.float() + (-1e9) * (1.0 - token_mask.float())
        valid_count = token_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        k_count = (pivot_topk * valid_count).ceil().long().clamp(min=1, max=token_mask.shape[1])
        sorted_ent, _ = ent_masked.sort(dim=1, descending=True)
        # Threshold: value at position k-1 (last selected rank)
        k_idx = (k_count - 1).clamp(min=0, max=sorted_ent.shape[1] - 1)
        thresholds = sorted_ent.gather(1, k_idx)  # [B, 1]
        pivot_mask = (ent_masked >= thresholds) & token_mask.bool()  # [B, L]

        # Identify wrong traces to perturb.
        corr_idx, corr_thr = self._corr_spec()
        if corr_idx >= 0 and corr_idx < rewards_per_func.shape[1]:
            correctness = rewards_per_func[:, corr_idx]
            is_valid = token_mask.sum(dim=1) > 0
            is_wrong = torch.isfinite(correctness) & (correctness < corr_thr) & is_valid
        else:
            is_wrong = torch.zeros(last_hidden.shape[0], dtype=torch.bool, device=last_hidden.device)

        # Inject noise at pivot positions of wrong traces only.
        if is_wrong.any():
            noise = torch.randn_like(last_hidden) * pivot_scale
            pivot_expand = pivot_mask.unsqueeze(-1).float()  # [B, L, 1]
            wrong_expand = is_wrong.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1]
            last_hidden = last_hidden + noise * pivot_expand * wrong_expand

        # Compute FFT features from (possibly perturbed) hidden states and return pair loss.
        num_bins = int(getattr(self.args, "fft_trace_aux_num_bins", 32))
        feat = self._fft_trace_features(last_hidden, token_mask, num_bins=num_bins)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("hipo/pivot_shake_topk_tokens_mean", []).append(
            float(pivot_mask.float().sum(dim=1).mean().detach().cpu())
        )

        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        detach_correct = bool(getattr(self.args, "fft_trace_aux_detach_correct", True))
        use_batch_fallback = bool(getattr(self.args, "fft_trace_aux_batch_fallback", True))
        distance = str(getattr(self.args, "fft_trace_aux_distance", "cosine"))

        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0

        losses = []
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
            for i in range(n_groups):
                c = correct_g[i]
                w = ~c
                if bool(c.any()) and bool(w.any()):
                    proto = feat_g[i][c].mean(dim=0)
                    if detach_correct:
                        proto = proto.detach()
                    losses.append(self._fft_aux_pair_loss(feat_g[i][w], proto, distance=distance))

        if not losses and use_batch_fallback:
            all_correct = feat[is_correct]
            all_wrong = feat[~is_correct]
            if all_correct.numel() > 0 and all_wrong.numel() > 0:
                proto = all_correct.mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                losses.append(self._fft_aux_pair_loss(all_wrong, proto, distance=distance))

        if not losses:
            return torch.zeros((), device=feat.device)
        return torch.stack(losses).mean()

    def _compute_fft_trace_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pull wrong traces toward correct FFT prototypes (group-first, batch fallback)."""
        fwd = self._aux_completion_last_hidden(model, inputs)
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        num_bins = int(getattr(self.args, "fft_trace_aux_num_bins", 32))
        feat = self._fft_trace_features(last_hidden, token_mask, num_bins=num_bins)

        mode = "train" if self.model.training else "eval"
        masked_token_mean = float(token_mask.sum(dim=1).float().mean().detach().cpu())
        self._metrics[mode].setdefault("hipo/fft_aux_masked_completion_tokens_mean", []).append(masked_token_mean)

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            self._metrics[mode].setdefault("hipo/fft_aux_primary_path", []).append(0.0)
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval

        detach_correct = bool(getattr(self.args, "fft_trace_aux_detach_correct", True))
        use_batch_fallback = bool(getattr(self.args, "fft_trace_aux_batch_fallback", True))
        distance = str(getattr(self.args, "fft_trace_aux_distance", "cosine"))
        proto_momentum = float(getattr(self.args, "fft_trace_aux_proto_momentum", 0.95))
        proto_warmup = int(getattr(self.args, "fft_trace_aux_proto_warmup_correct", 1))
        self._metrics[mode].setdefault("hipo/fft_aux_batch_size", []).append(float(n))
        self._metrics[mode].setdefault("hipo/fft_aux_num_generations", []).append(float(num_generations))

        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
        else:
            feat_g = None
            correct_g = None

        losses = []
        groups_used = 0
        wrong_samples_used = 0
        if can_use_contiguous_grouping and feat_g is not None and correct_g is not None:
            for i in range(n_groups):
                c = correct_g[i]
                w = ~c
                if bool(c.any()) and bool(w.any()):
                    proto = feat_g[i][c].mean(dim=0)
                    if detach_correct:
                        proto = proto.detach()
                    wrong_feat = feat_g[i][w]
                    losses.append(self._fft_aux_pair_loss(wrong_feat, proto, distance=distance))
                    groups_used += 1
                    wrong_samples_used += int(wrong_feat.shape[0])

        batch_fallback_used = 0.0
        if not losses and use_batch_fallback:
            all_correct = feat[is_correct]
            all_wrong = feat[~is_correct]
            if all_correct.numel() > 0 and all_wrong.numel() > 0:
                proto = all_correct.mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                losses.append(self._fft_aux_pair_loss(all_wrong, proto, distance=distance))
                batch_fallback_used = 1.0
                wrong_samples_used += int(all_wrong.shape[0])

        # Cross-step prototype memory: needed when micro-batch size is tiny (often B=1), where
        # in-batch mixed correct/wrong pairs almost never occur.
        current_correct = feat[is_correct]
        if current_correct.numel() > 0:
            cur_proto = current_correct.mean(dim=0).detach()
            if self._fft_correct_proto is None:
                self._fft_correct_proto = cur_proto
            else:
                m = min(max(proto_momentum, 0.0), 0.9999)
                self._fft_correct_proto = F.normalize(
                    m * self._fft_correct_proto + (1.0 - m) * cur_proto,
                    p=2,
                    dim=0,
                    eps=1e-8,
                )
            self._fft_correct_proto_updates += 1

        memory_fallback_used = 0.0
        if not losses and self._fft_correct_proto is not None and self._fft_correct_proto_updates >= max(proto_warmup, 1):
            all_wrong = feat[~is_correct]
            if all_wrong.numel() > 0:
                proto = self._fft_correct_proto
                if not detach_correct:
                    # Memory proto is detached state by design, so this remains detached either way.
                    proto = proto
                losses.append(self._fft_aux_pair_loss(all_wrong, proto, distance=distance))
                memory_fallback_used = 1.0
                wrong_samples_used += int(all_wrong.shape[0])

        if not losses:
            self._metrics[mode].setdefault("hipo/fft_aux_primary_path", []).append(0.0)
            self._metrics[mode].setdefault("hipo/fft_aux_groups_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/fft_aux_wrong_samples_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/fft_aux_batch_fallback_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/fft_aux_memory_fallback_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/fft_aux_proto_updates", []).append(float(self._fft_correct_proto_updates))
            return torch.zeros((), device=feat.device)

        if groups_used > 0:
            primary_path = 1.0
        elif batch_fallback_used > 0:
            primary_path = 2.0
        elif memory_fallback_used > 0:
            primary_path = 3.0
        else:
            primary_path = 0.0
        self._metrics[mode].setdefault("hipo/fft_aux_primary_path", []).append(primary_path)

        self._metrics[mode].setdefault("hipo/fft_aux_groups_used", []).append(float(groups_used))
        self._metrics[mode].setdefault("hipo/fft_aux_wrong_samples_used", []).append(float(wrong_samples_used))
        self._metrics[mode].setdefault("hipo/fft_aux_batch_fallback_used", []).append(float(batch_fallback_used))
        self._metrics[mode].setdefault("hipo/fft_aux_memory_fallback_used", []).append(float(memory_fallback_used))
        self._metrics[mode].setdefault("hipo/fft_aux_proto_updates", []).append(float(self._fft_correct_proto_updates))
        return torch.stack(losses).mean()

    def _compute_exp39_spectral_aux_loss(self, model, inputs: dict[str, torch.Tensor], *, spectrum_kind: str) -> torch.Tensor:
        """Exp39 a2 (magnitude) / a3 (phase): FFT-domain prototype pull with alternate features."""
        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="fft")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        num_bins = int(getattr(self.args, "fft_trace_aux_num_bins", 32))
        if spectrum_kind == "mag":
            feat = self._fft_mag_trace_features(last_hidden, token_mask, num_bins=num_bins)
        elif spectrum_kind == "phase":
            feat = self._fft_phase_trace_features(last_hidden, token_mask, num_bins=num_bins)
        else:
            raise ValueError(f"Unknown spectrum_kind={spectrum_kind!r}")

        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("hipo/exp39_spec_masked_completion_tokens_mean", []).append(
            float(token_mask.sum(dim=1).float().mean().detach().cpu())
        )

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            self._metrics[mode].setdefault("hipo/exp39_spec_primary_path", []).append(0.0)
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        detach_correct = bool(getattr(self.args, "fft_trace_aux_detach_correct", True))
        use_batch_fallback = bool(getattr(self.args, "fft_trace_aux_batch_fallback", True))
        distance = str(getattr(self.args, "fft_trace_aux_distance", "cosine"))
        proto_momentum = float(getattr(self.args, "fft_trace_aux_proto_momentum", 0.95))
        proto_warmup = int(getattr(self.args, "fft_trace_aux_proto_warmup_correct", 1))

        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
        else:
            feat_g = None
            correct_g = None

        losses = []
        groups_used = 0
        wrong_samples_used = 0
        if can_use_contiguous_grouping and feat_g is not None and correct_g is not None:
            for i in range(n_groups):
                c = correct_g[i]
                w = ~c
                if bool(c.any()) and bool(w.any()):
                    proto = feat_g[i][c].mean(dim=0)
                    if detach_correct:
                        proto = proto.detach()
                    wrong_feat = feat_g[i][w]
                    losses.append(self._fft_aux_pair_loss(wrong_feat, proto, distance=distance))
                    groups_used += 1
                    wrong_samples_used += int(wrong_feat.shape[0])

        batch_fallback_used = 0.0
        if not losses and use_batch_fallback:
            all_correct = feat[is_correct]
            all_wrong = feat[~is_correct]
            if all_correct.numel() > 0 and all_wrong.numel() > 0:
                proto = all_correct.mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                losses.append(self._fft_aux_pair_loss(all_wrong, proto, distance=distance))
                batch_fallback_used = 1.0
                wrong_samples_used += int(all_wrong.shape[0])

        current_correct = feat[is_correct]
        if current_correct.numel() > 0:
            cur_proto = current_correct.mean(dim=0).detach()
            if self._exp39_spec_proto is None:
                self._exp39_spec_proto = cur_proto
            else:
                m = min(max(proto_momentum, 0.0), 0.9999)
                self._exp39_spec_proto = F.normalize(
                    m * self._exp39_spec_proto + (1.0 - m) * cur_proto,
                    p=2,
                    dim=0,
                    eps=1e-8,
                )
            self._exp39_spec_proto_updates += 1

        memory_fallback_used = 0.0
        if not losses and self._exp39_spec_proto is not None and self._exp39_spec_proto_updates >= max(proto_warmup, 1):
            all_wrong = feat[~is_correct]
            if all_wrong.numel() > 0:
                proto = self._exp39_spec_proto
                losses.append(self._fft_aux_pair_loss(all_wrong, proto, distance=distance))
                memory_fallback_used = 1.0
                wrong_samples_used += int(all_wrong.shape[0])

        if not losses:
            self._metrics[mode].setdefault("hipo/exp39_spec_primary_path", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp39_spec_groups_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp39_spec_wrong_samples_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp39_spec_proto_updates", []).append(float(self._exp39_spec_proto_updates))
            return torch.zeros((), device=feat.device)

        if groups_used > 0:
            primary_path = 1.0
        elif batch_fallback_used > 0:
            primary_path = 2.0
        elif memory_fallback_used > 0:
            primary_path = 3.0
        else:
            primary_path = 0.0
        self._metrics[mode].setdefault("hipo/exp39_spec_primary_path", []).append(primary_path)
        self._metrics[mode].setdefault("hipo/exp39_spec_groups_used", []).append(float(groups_used))
        self._metrics[mode].setdefault("hipo/exp39_spec_wrong_samples_used", []).append(float(wrong_samples_used))
        self._metrics[mode].setdefault("hipo/exp39_spec_proto_updates", []).append(float(self._exp39_spec_proto_updates))
        return torch.stack(losses).mean()

    def _compute_exp39_fft_mmd_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp39 a8: linear MMD^2 on pooled FFT features between correct and wrong completions."""
        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="fft")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        num_bins = int(getattr(self.args, "fft_trace_aux_num_bins", 32))
        feat = self._fft_trace_features(last_hidden, token_mask, num_bins=num_bins)

        mode = "train" if self.model.training else "eval"
        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            self._metrics[mode].setdefault("hipo/exp39_mmd_active", []).append(0.0)
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        fc = feat[is_correct]
        fw = feat[~is_correct]
        if fc.shape[0] == 0 or fw.shape[0] == 0:
            self._metrics[mode].setdefault("hipo/exp39_mmd_active", []).append(0.0)
            return torch.zeros((), device=feat.device)

        mmd = (fc.mean(dim=0) - fw.mean(dim=0)).square().sum()
        self._metrics[mode].setdefault("hipo/exp39_mmd_active", []).append(1.0)
        return mmd

    def _compute_exp39_subspace_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp39 a4: pull wrong traces toward a 2D subspace spanned by two correct directions (when available)."""
        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="dir")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        feat = self._hidden_direction_features(last_hidden, token_mask)
        mode = "train" if self.model.training else "eval"

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        distance = str(getattr(self.args, "hidden_direction_aux_distance", "cosine"))
        detach_correct = bool(getattr(self.args, "hidden_direction_aux_detach_correct", True))

        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0
        losses = []
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
            for i in range(n_groups):
                c = correct_g[i]
                w = ~c
                if not (bool(c.any()) and bool(w.any())):
                    continue
                correct_idx = torch.nonzero(c, as_tuple=False).squeeze(-1)
                proto1 = feat_g[i][c].mean(dim=0)
                if detach_correct:
                    proto1 = proto1.detach()
                proto1_u = F.normalize(proto1, p=2, dim=0, eps=1e-8)
                if int(correct_idx.numel()) >= 2:
                    c0 = feat_g[i][correct_idx[0]]
                    c1 = feat_g[i][correct_idx[1]]
                    u2 = c1 - (c1 @ proto1_u) * proto1_u
                    if float(u2.norm().detach().cpu()) < 1e-5:
                        u2 = c0 - (c0 @ proto1_u) * proto1_u
                    proto2 = F.normalize(u2, p=2, dim=0, eps=1e-8)
                    wrong_feat = feat_g[i][w]
                    l1 = self._direction_aux_pair_loss(wrong_feat, proto1_u, distance=distance)
                    l2 = self._direction_aux_pair_loss(wrong_feat, proto2, distance=distance)
                    losses.append(0.5 * (l1 + l2))
                else:
                    wrong_feat = feat_g[i][w]
                    losses.append(self._direction_aux_pair_loss(wrong_feat, proto1_u, distance=distance))

        if not losses:
            all_correct = feat[is_correct]
            all_wrong = feat[~is_correct]
            if all_correct.numel() > 0 and all_wrong.numel() > 0:
                proto1 = all_correct.mean(dim=0)
                if detach_correct:
                    proto1 = proto1.detach()
                proto1_u = F.normalize(proto1, p=2, dim=0, eps=1e-8)
                if all_correct.shape[0] >= 2:
                    c0, c1 = all_correct[0], all_correct[1]
                    u2 = c1 - (c1 @ proto1_u) * proto1_u
                    if float(u2.norm().detach().cpu()) < 1e-5:
                        u2 = c0 - (c0 @ proto1_u) * proto1_u
                    proto2 = F.normalize(u2, p=2, dim=0, eps=1e-8)
                    l1 = self._direction_aux_pair_loss(all_wrong, proto1_u, distance=distance)
                    l2 = self._direction_aux_pair_loss(all_wrong, proto2, distance=distance)
                    losses.append(0.5 * (l1 + l2))
                else:
                    losses.append(self._direction_aux_pair_loss(all_wrong, proto1_u, distance=distance))

        self._metrics[mode].setdefault("hipo/exp39_subspace_groups", []).append(float(len(losses)))
        if not losses:
            return torch.zeros((), device=feat.device)
        return torch.stack(losses).mean()

    def _compute_exp39_pairwise_margin_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp39 a5: hinge margin separating wrong traces from a correct prototype vs. other wrongs."""
        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="dir")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        feat = self._hidden_direction_features(last_hidden, token_mask)
        mode = "train" if self.model.training else "eval"
        m = float(getattr(self.args, "exp39_pairwise_margin_m", 0.15) or 0.0)

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0
        losses = []
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
            for i in range(n_groups):
                c = correct_g[i]
                w = ~c
                if not (bool(c.any()) and bool(w.any())):
                    continue
                proto = F.normalize(feat_g[i][c].mean(dim=0), p=2, dim=0, eps=1e-8).detach()
                wrong_feat = feat_g[i][w]
                nw = int(wrong_feat.shape[0])
                if nw <= 0:
                    continue
                pos_cos = F.cosine_similarity(wrong_feat, proto.unsqueeze(0), dim=1, eps=1e-8)
                if nw == 1:
                    hinge = F.relu(m - pos_cos)
                else:
                    neg_sum = wrong_feat.sum(dim=0)
                    neg_stack = (neg_sum.unsqueeze(0) - wrong_feat) / max(nw - 1, 1)
                    neg = F.normalize(neg_stack, p=2, dim=1, eps=1e-8)
                    neg_cos = F.cosine_similarity(wrong_feat, neg, dim=1, eps=1e-8)
                    hinge = F.relu(m - pos_cos + neg_cos)
                losses.append(hinge.mean())

        if not losses:
            all_correct = feat[is_correct]
            all_wrong = feat[~is_correct]
            if all_correct.numel() > 0 and all_wrong.numel() > 0:
                proto = F.normalize(all_correct.mean(dim=0), p=2, dim=0, eps=1e-8).detach()
                nw = int(all_wrong.shape[0])
                pos_cos = F.cosine_similarity(all_wrong, proto.unsqueeze(0), dim=1, eps=1e-8)
                if nw == 1:
                    losses.append(F.relu(m - pos_cos).mean())
                else:
                    neg_sum = all_wrong.sum(dim=0)
                    neg = F.normalize((neg_sum.unsqueeze(0) - all_wrong) / max(nw - 1, 1), p=2, dim=1, eps=1e-8)
                    neg_cos = F.cosine_similarity(all_wrong, neg, dim=1, eps=1e-8)
                    losses.append(F.relu(m - pos_cos + neg_cos).mean())

        self._metrics[mode].setdefault("hipo/exp39_margin_groups", []).append(float(len(losses)))
        if not losses:
            return torch.zeros((), device=feat.device)
        return torch.stack(losses).mean()

    def _compute_exp39_entropy_gated_dir_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp39 a6: hidden-direction pull weighted higher on high-entropy completions."""
        packed = self._aux_completion_hidden_logits_entropy(model, inputs, first_n_field="dir")
        if packed is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func, _logits, mean_ent = packed
        feat = self._hidden_direction_features(last_hidden, token_mask)
        mode = "train" if self.model.training else "eval"
        k_gate = float(getattr(self.args, "exp39_entropy_gate_std", 0.5) or 0.0)

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid
        wrong_only = bool(getattr(self.args, "hidden_direction_aux_wrong_only", True))
        use_mask = ~is_correct if wrong_only else is_valid

        med = torch.median(mean_ent.detach())
        std = torch.std(mean_ent.detach(), unbiased=False).clamp(min=1e-6)
        thr = med + k_gate * std
        gate = (mean_ent > thr).float()

        distance = str(getattr(self.args, "hidden_direction_aux_distance", "cosine"))
        detach_correct = bool(getattr(self.args, "hidden_direction_aux_detach_correct", True))
        use_batch_fallback = bool(getattr(self.args, "hidden_direction_aux_batch_fallback", True))
        proto_momentum = float(getattr(self.args, "hidden_direction_aux_proto_momentum", 0.95))
        proto_warmup = int(getattr(self.args, "hidden_direction_aux_proto_warmup_correct", 1))

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0

        losses = []
        groups_used = 0
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
            use_g = use_mask.view(n_groups, num_generations)
            gate_g = gate.view(n_groups, num_generations)
            for i in range(n_groups):
                c = correct_g[i]
                moved = use_g[i]
                if not (bool(c.any()) and bool(moved.any())):
                    continue
                proto = feat_g[i][c].mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                moved_feat = feat_g[i][moved]
                g = gate_g[i][moved]
                per = 1.0 - F.cosine_similarity(moved_feat, proto.unsqueeze(0), dim=1, eps=1e-8)
                wsum = g.sum().clamp(min=1e-6)
                losses.append((per * g).sum() / wsum)
                groups_used += 1

        batch_fallback_used = 0.0
        if not losses and use_batch_fallback:
            all_correct = feat[is_correct]
            all_moved = feat[use_mask]
            g2 = gate[use_mask]
            if all_correct.numel() > 0 and all_moved.numel() > 0:
                proto = all_correct.mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                per = 1.0 - F.cosine_similarity(all_moved, proto.unsqueeze(0), dim=1, eps=1e-8)
                wsum = g2.sum().clamp(min=1e-6)
                losses.append((per * g2).sum() / wsum)
                batch_fallback_used = 1.0

        current_correct = feat[is_correct]
        if current_correct.numel() > 0:
            cur_proto = current_correct.mean(dim=0).detach()
            if self._dir_correct_proto is None:
                self._dir_correct_proto = cur_proto
            else:
                m = min(max(proto_momentum, 0.0), 0.9999)
                self._dir_correct_proto = F.normalize(
                    m * self._dir_correct_proto + (1.0 - m) * cur_proto,
                    p=2,
                    dim=0,
                    eps=1e-8,
                )
            self._dir_correct_proto_updates += 1

        memory_fallback_used = 0.0
        if not losses and self._dir_correct_proto is not None and self._dir_correct_proto_updates >= max(proto_warmup, 1):
            all_moved = feat[use_mask]
            g2 = gate[use_mask]
            if all_moved.numel() > 0:
                proto = self._dir_correct_proto
                per = 1.0 - F.cosine_similarity(all_moved, proto.unsqueeze(0), dim=1, eps=1e-8)
                wsum = g2.sum().clamp(min=1e-6)
                losses.append((per * g2).sum() / wsum)
                memory_fallback_used = 1.0

        frac_high = float(gate[use_mask].mean().detach().cpu()) if bool(use_mask.any()) else 0.0
        self._metrics[mode].setdefault("hipo/exp39_entropy_gate_frac", []).append(frac_high)
        if not losses:
            return torch.zeros((), device=feat.device)
        return torch.stack(losses).mean()

    def _compute_exp39_length_bucket_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp39 a7: prompt-length bucket prototypes (cheap proxy for puzzle 'type')."""
        rewards_per_func = inputs.get("rewards_per_func")
        if rewards_per_func is None or rewards_per_func.numel() == 0:
            return torch.zeros((), device=inputs["completion_ids"].device)

        prompt_mask = inputs["prompt_mask"]
        prompt_lens = prompt_mask.sum(dim=1).float()
        q = torch.quantile(
            prompt_lens,
            torch.tensor([0.33, 0.66], device=prompt_lens.device, dtype=prompt_lens.dtype),
        )
        bucket = (prompt_lens > q[0]).long() + (prompt_lens > q[1]).long()

        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="dir")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        feat = self._hidden_direction_features(last_hidden, token_mask)
        mode = "train" if self.model.training else "eval"

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid

        distance = str(getattr(self.args, "hidden_direction_aux_distance", "cosine"))
        detach_correct = bool(getattr(self.args, "hidden_direction_aux_detach_correct", True))
        proto_momentum = float(getattr(self.args, "hidden_direction_aux_proto_momentum", 0.95))

        for bb in range(3):
            sel = is_correct & (bucket == bb)
            if not bool(sel.any()):
                continue
            cur = feat[sel].mean(dim=0).detach()
            if self._exp39_len_proto[bb] is None:
                self._exp39_len_proto[bb] = F.normalize(cur, p=2, dim=0, eps=1e-8)
            else:
                m = min(max(proto_momentum, 0.0), 0.9999)
                self._exp39_len_proto[bb] = F.normalize(
                    m * self._exp39_len_proto[bb] + (1.0 - m) * cur,
                    p=2,
                    dim=0,
                    eps=1e-8,
                )

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0
        losses = []
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
            bucket_g = bucket.view(n_groups, num_generations)
            for i in range(n_groups):
                c = correct_g[i]
                w = ~c
                if not (bool(c.any()) and bool(w.any())):
                    continue
                bb = int(bucket_g[i, 0].item())
                proto_g = feat_g[i][c].mean(dim=0)
                if detach_correct:
                    proto_g = proto_g.detach()
                mem = self._exp39_len_proto[bb]
                if mem is not None:
                    proto_use = mem.detach()
                else:
                    proto_use = F.normalize(proto_g, p=2, dim=0, eps=1e-8)
                wrong_feat = feat_g[i][w]
                losses.append(self._direction_aux_pair_loss(wrong_feat, proto_use, distance=distance))

        if not losses:
            all_wrong = feat[~is_correct]
            b_wrong = bucket[~is_correct]
            if all_wrong.numel() > 0:
                for bb in range(3):
                    selw = b_wrong == bb
                    if not bool(selw.any()):
                        continue
                    mem = self._exp39_len_proto[bb]
                    if mem is None:
                        continue
                    wf = all_wrong[selw]
                    losses.append(self._direction_aux_pair_loss(wf, mem.detach(), distance=distance))

        self._metrics[mode].setdefault("hipo/exp39_len_bucket_groups", []).append(float(len(losses)))
        if not losses:
            return torch.zeros((), device=feat.device)
        return torch.stack(losses).mean()

    @staticmethod
    def _hidden_direction_features(last_hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Return per-sequence hidden direction feature from completion states."""
        m = token_mask.unsqueeze(-1).float()
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        feat = (last_hidden.float() * m).sum(dim=1) / denom
        return F.normalize(feat, p=2, dim=1, eps=1e-8)

    def _direction_aux_pair_loss(self, moved_feat: torch.Tensor, proto: torch.Tensor, distance: str) -> torch.Tensor:
        if distance == "mse":
            return (moved_feat - proto.unsqueeze(0)).square().mean()
        if distance != "cosine":
            raise ValueError(f"Unknown hidden_direction_aux_distance={distance!r}; expected 'cosine' or 'mse'.")
        cos = F.cosine_similarity(moved_feat, proto.unsqueeze(0), dim=1, eps=1e-8)
        return (1.0 - cos).mean()

    def _compute_hidden_direction_aux_loss(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pull selected traces toward correct hidden prototype in representation space."""
        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="dir")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        feat = self._hidden_direction_features(last_hidden, token_mask)

        mode = "train" if self.model.training else "eval"
        masked_token_mean = float(token_mask.sum(dim=1).float().mean().detach().cpu())
        self._metrics[mode].setdefault("hipo/dir_aux_masked_completion_tokens_mean", []).append(masked_token_mean)

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            self._metrics[mode].setdefault("hipo/dir_aux_primary_path", []).append(0.0)
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        is_valid = token_mask.sum(dim=1) > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid
        wrong_only = bool(getattr(self.args, "hidden_direction_aux_wrong_only", True))
        if wrong_only:
            use_mask = ~is_correct
        else:
            use_mask = is_valid

        n = feat.shape[0]
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        detach_correct = bool(getattr(self.args, "hidden_direction_aux_detach_correct", True))
        use_batch_fallback = bool(getattr(self.args, "hidden_direction_aux_batch_fallback", True))
        distance = str(getattr(self.args, "hidden_direction_aux_distance", "cosine"))
        proto_momentum = float(getattr(self.args, "hidden_direction_aux_proto_momentum", 0.95))
        proto_warmup = int(getattr(self.args, "hidden_direction_aux_proto_warmup_correct", 1))
        self._metrics[mode].setdefault("hipo/dir_aux_batch_size", []).append(float(n))
        self._metrics[mode].setdefault("hipo/dir_aux_num_generations", []).append(float(num_generations))

        can_use_contiguous_grouping = num_generations > 1 and n % num_generations == 0
        n_groups = n // num_generations if can_use_contiguous_grouping else 0
        if can_use_contiguous_grouping:
            feat_g = feat.view(n_groups, num_generations, -1)
            correct_g = is_correct.view(n_groups, num_generations)
            use_g = use_mask.view(n_groups, num_generations)
        else:
            feat_g = None
            correct_g = None
            use_g = None

        losses = []
        groups_used = 0
        used_samples = 0
        if can_use_contiguous_grouping and feat_g is not None and correct_g is not None and use_g is not None:
            for i in range(n_groups):
                c = correct_g[i]
                moved = use_g[i]
                if bool(c.any()) and bool(moved.any()):
                    proto = feat_g[i][c].mean(dim=0)
                    if detach_correct:
                        proto = proto.detach()
                    moved_feat = feat_g[i][moved]
                    losses.append(self._direction_aux_pair_loss(moved_feat, proto, distance=distance))
                    groups_used += 1
                    used_samples += int(moved_feat.shape[0])

        batch_fallback_used = 0.0
        if not losses and use_batch_fallback:
            all_correct = feat[is_correct]
            all_moved = feat[use_mask]
            if all_correct.numel() > 0 and all_moved.numel() > 0:
                proto = all_correct.mean(dim=0)
                if detach_correct:
                    proto = proto.detach()
                losses.append(self._direction_aux_pair_loss(all_moved, proto, distance=distance))
                batch_fallback_used = 1.0
                used_samples += int(all_moved.shape[0])

        current_correct = feat[is_correct]
        if current_correct.numel() > 0:
            cur_proto = current_correct.mean(dim=0).detach()
            if self._dir_correct_proto is None:
                self._dir_correct_proto = cur_proto
            else:
                m = min(max(proto_momentum, 0.0), 0.9999)
                self._dir_correct_proto = F.normalize(
                    m * self._dir_correct_proto + (1.0 - m) * cur_proto,
                    p=2,
                    dim=0,
                    eps=1e-8,
                )
            self._dir_correct_proto_updates += 1

        memory_fallback_used = 0.0
        if not losses and self._dir_correct_proto is not None and self._dir_correct_proto_updates >= max(proto_warmup, 1):
            all_moved = feat[use_mask]
            if all_moved.numel() > 0:
                proto = self._dir_correct_proto
                losses.append(self._direction_aux_pair_loss(all_moved, proto, distance=distance))
                memory_fallback_used = 1.0
                used_samples += int(all_moved.shape[0])

        if not losses:
            self._metrics[mode].setdefault("hipo/dir_aux_primary_path", []).append(0.0)
            self._metrics[mode].setdefault("hipo/dir_aux_groups_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/dir_aux_used_samples", []).append(0.0)
            self._metrics[mode].setdefault("hipo/dir_aux_batch_fallback_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/dir_aux_memory_fallback_used", []).append(0.0)
            self._metrics[mode].setdefault("hipo/dir_aux_proto_updates", []).append(float(self._dir_correct_proto_updates))
            return torch.zeros((), device=feat.device)

        if groups_used > 0:
            primary_path = 1.0
        elif batch_fallback_used > 0:
            primary_path = 2.0
        elif memory_fallback_used > 0:
            primary_path = 3.0
        else:
            primary_path = 0.0
        self._metrics[mode].setdefault("hipo/dir_aux_primary_path", []).append(primary_path)
        self._metrics[mode].setdefault("hipo/dir_aux_groups_used", []).append(float(groups_used))
        self._metrics[mode].setdefault("hipo/dir_aux_used_samples", []).append(float(used_samples))
        self._metrics[mode].setdefault("hipo/dir_aux_batch_fallback_used", []).append(float(batch_fallback_used))
        self._metrics[mode].setdefault("hipo/dir_aux_memory_fallback_used", []).append(float(memory_fallback_used))
        self._metrics[mode].setdefault("hipo/dir_aux_proto_updates", []).append(float(self._dir_correct_proto_updates))
        return torch.stack(losses).mean()

    def _compute_exp40_wrong_short_penalty(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp40: penalize wrong completions shorter than a token threshold (no extra forward)."""
        rewards_per_func = inputs.get("rewards_per_func")
        if rewards_per_func is None or rewards_per_func.numel() == 0:
            return torch.zeros((), device=inputs["completion_ids"].device)

        completion_mask = inputs["completion_mask"]
        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=completion_mask.device)

        correctness = rewards_per_func[:, corr_idx]
        lens = completion_mask.float().sum(dim=1)
        is_valid = lens > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid
        wrong = ~is_correct & is_valid

        min_t = max(int(getattr(self.args, "exp40_wrong_short_min_tokens", 32) or 32), 1)
        shape = str(getattr(self.args, "exp40_wrong_short_shape", "hinge") or "hinge").lower()
        shortfall = (float(min_t) - lens).clamp(min=0.0) / float(min_t)
        if shape == "linear":
            per = shortfall
        else:
            per = shortfall.square()

        if not bool(wrong.any()):
            return torch.zeros((), device=completion_mask.device, dtype=per.dtype)
        return per[wrong].mean()

    def _compute_exp40_freq_collapse_penalty(self, model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Exp40: wrong-only proxy for spectral energy in low FFT bins (early training only)."""
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        warmup = int(getattr(self.args, "exp40_freq_warmup_steps", 0) or 0)
        lam = float(getattr(self.args, "exp40_freq_lambda", 0.0) or 0.0)
        if lam <= 0.0 or step > warmup:
            return torch.zeros((), device=inputs["completion_ids"].device)

        rewards_per_func = inputs.get("rewards_per_func")
        if rewards_per_func is None or rewards_per_func.numel() == 0:
            return torch.zeros((), device=inputs["completion_ids"].device)

        fwd = self._aux_completion_last_hidden(model, inputs, first_n_field="fft")
        if fwd is None:
            return torch.zeros((), device=inputs["completion_ids"].device)
        last_hidden, token_mask, rewards_per_func = fwd
        num_bins = int(getattr(self.args, "fft_trace_aux_num_bins", 32))
        feat = self._fft_trace_features(last_hidden, token_mask, num_bins=num_bins)

        ratio = float(getattr(self.args, "exp40_freq_lowbin_ratio", 0.25) or 0.25)
        ratio = min(max(ratio, 0.0), 1.0)
        k = max(1, int(num_bins * ratio))
        k = min(k, feat.shape[1])

        corr_idx, corr_thr = self._corr_spec()
        if corr_idx < 0 or corr_idx >= rewards_per_func.shape[1]:
            return torch.zeros((), device=feat.device)

        correctness = rewards_per_func[:, corr_idx]
        lens = token_mask.float().sum(dim=1)
        is_valid = lens > 0
        is_correct = torch.isfinite(correctness) & (correctness >= corr_thr) & is_valid
        wrong = ~is_correct & is_valid

        if not bool(wrong.any()):
            return torch.zeros((), device=feat.device, dtype=feat.dtype)

        abs_f = feat.abs()
        low_mass = abs_f[:, :k].sum(dim=1) / abs_f.sum(dim=1).clamp(min=1e-8)
        # Store diagnostic on module for compute_loss metrics (scalar tensor).
        self._exp40_last_freq_lowbin_mean = low_mass[wrong].mean()
        return self._exp40_last_freq_lowbin_mean

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        fft_aux_lambda = float(getattr(self.args, "fft_trace_aux_lambda", 0.0) or 0.0)
        dir_aux_lambda = float(getattr(self.args, "hidden_direction_aux_lambda", 0.0) or 0.0)
        arm = str(getattr(self.args, "exp39_arm", "") or "").strip().lower()
        mode = "train" if self.model.training else "eval"
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)

        pivot_ok, pivot_reason, pivot_stats = self._pivot_aux_evaluate(inputs)
        self._log_pivot_aux_metrics(mode, pivot_ok, pivot_reason, pivot_stats)

        exp39_unscaled = zero
        fft_aux = zero
        dir_aux = zero
        exp39_scaled_metric = 0.0

        wants_hipo_aux = (arm != "a0") and (
            (arm in _EXP39_SPECIAL_ARMS) or (fft_aux_lambda > 0.0) or (dir_aux_lambda > 0.0)
        )

        if wants_hipo_aux and pivot_ok:
            if arm == "a2":
                exp39_unscaled = self._compute_exp39_spectral_aux_loss(model, inputs, spectrum_kind="mag")
            elif arm == "a3":
                exp39_unscaled = self._compute_exp39_spectral_aux_loss(model, inputs, spectrum_kind="phase")
            elif arm == "a4":
                exp39_unscaled = self._compute_exp39_subspace_aux_loss(model, inputs)
            elif arm == "a5":
                exp39_unscaled = self._compute_exp39_pairwise_margin_aux_loss(model, inputs)
            elif arm == "a6":
                exp39_unscaled = self._compute_exp39_entropy_gated_dir_aux_loss(model, inputs)
            elif arm == "a7":
                exp39_unscaled = self._compute_exp39_length_bucket_aux_loss(model, inputs)
            elif arm == "a8":
                exp39_unscaled = self._compute_exp39_fft_mmd_aux_loss(model, inputs)
            else:
                fft_aux = (
                    self._compute_fft_trace_aux_loss(model, inputs)
                    if fft_aux_lambda > 0.0
                    else zero
                )
                dir_aux = (
                    self._compute_hidden_direction_aux_loss(model, inputs)
                    if dir_aux_lambda > 0.0
                    else zero
                )

        if arm in _EXP39_SPECIAL_ARMS:
            coeff = fft_aux_lambda if arm in ("a2", "a3", "a8") else dir_aux_lambda
            extra = coeff * exp39_unscaled
            exp39_scaled_metric = float((coeff * exp39_unscaled).detach().cpu())
        elif arm == "a0":
            extra = zero
        else:
            extra = fft_aux_lambda * fft_aux + dir_aux_lambda * dir_aux

        # Exp70: entropy-gated pivot hidden-state shake (additive, independent of fft_aux_lambda).
        pivot_shake_enable = bool(getattr(self.args, "pivot_shake_enable", False))
        pivot_shake_lambda = float(getattr(self.args, "fft_trace_aux_lambda", 0.0) or 0.0)
        pivot_shake_aux = zero
        if pivot_shake_enable and pivot_ok:
            pivot_shake_aux = self._compute_pivot_shake_aux_loss(model, inputs)
            extra = extra + pivot_shake_lambda * pivot_shake_aux
        self._metrics[mode].setdefault("hipo/pivot_shake_unscaled", []).append(float(pivot_shake_aux.detach().cpu()))
        self._metrics[mode].setdefault("hipo/pivot_shake_scaled", []).append(float((pivot_shake_lambda * pivot_shake_aux).detach().cpu()))

        self._metrics[mode].setdefault("hipo/fft_aux_unscaled", []).append(float(fft_aux.detach().cpu()))
        self._metrics[mode].setdefault("hipo/fft_aux_scaled", []).append(float((fft_aux_lambda * fft_aux).detach().cpu()))
        self._metrics[mode].setdefault("hipo/dir_aux_unscaled", []).append(float(dir_aux.detach().cpu()))
        self._metrics[mode].setdefault("hipo/dir_aux_scaled", []).append(float((dir_aux_lambda * dir_aux).detach().cpu()))
        self._metrics[mode].setdefault("hipo/exp39_aux_unscaled", []).append(float(exp39_unscaled.detach().cpu()))
        self._metrics[mode].setdefault("hipo/exp39_aux_scaled", []).append(exp39_scaled_metric)

        exp40_enable = bool(getattr(self.args, "exp40_enable", False))
        lam_s = float(getattr(self.args, "exp40_wrong_short_lambda", 0.0) or 0.0)
        lam_f = float(getattr(self.args, "exp40_freq_lambda", 0.0) or 0.0)
        exp40_extra = zero
        if exp40_enable and (lam_s > 0.0 or lam_f > 0.0):
            if pivot_ok:
                short_u = (
                    self._compute_exp40_wrong_short_penalty(inputs)
                    if lam_s > 0.0
                    else torch.zeros((), device=loss.device, dtype=loss.dtype)
                )
                freq_u = (
                    self._compute_exp40_freq_collapse_penalty(model, inputs)
                    if lam_f > 0.0
                    else torch.zeros((), device=loss.device, dtype=loss.dtype)
                )
                exp40_extra = lam_s * short_u + lam_f * freq_u

                rewards_per_func = inputs.get("rewards_per_func")
                completion_mask = inputs["completion_mask"]
                short_rate = 0.0
                if rewards_per_func is not None and rewards_per_func.numel() > 0 and completion_mask is not None:
                    corr_idx, corr_thr = self._corr_spec()
                    lens = completion_mask.float().sum(dim=1)
                    _is_valid, _is_correct, wrong = self._completion_correctness_masks(
                        rewards_per_func, completion_mask, corr_idx=corr_idx, corr_thr=corr_thr
                    )
                    min_t = max(int(getattr(self.args, "exp40_wrong_short_min_tokens", 32) or 32), 1)
                    if bool(wrong.any()):
                        short_rate = float((lens[wrong] < float(min_t)).float().mean().detach().cpu())

                self._metrics[mode].setdefault("hipo/exp40_wrong_short_unscaled", []).append(float(short_u.detach().cpu()))
                self._metrics[mode].setdefault("hipo/exp40_wrong_short_scaled", []).append(
                    float((lam_s * short_u).detach().cpu())
                )
                self._metrics[mode].setdefault("hipo/exp40_wrong_short_rate", []).append(short_rate)
                self._metrics[mode].setdefault("hipo/exp40_freq_unscaled", []).append(float(freq_u.detach().cpu()))
                self._metrics[mode].setdefault("hipo/exp40_freq_scaled", []).append(float((lam_f * freq_u).detach().cpu()))
                lowbin_m = (
                    float(self._exp40_last_freq_lowbin_mean.detach().cpu())
                    if self._exp40_last_freq_lowbin_mean is not None
                    else float(freq_u.detach().cpu())
                )
                self._metrics[mode].setdefault("hipo/exp40_freq_lowbin_ratio_mean", []).append(lowbin_m)
                self._exp40_last_freq_lowbin_mean = None
            else:
                self._metrics[mode].setdefault("hipo/exp40_wrong_short_unscaled", []).append(0.0)
                self._metrics[mode].setdefault("hipo/exp40_wrong_short_scaled", []).append(0.0)
                self._metrics[mode].setdefault("hipo/exp40_wrong_short_rate", []).append(0.0)
                self._metrics[mode].setdefault("hipo/exp40_freq_unscaled", []).append(0.0)
                self._metrics[mode].setdefault("hipo/exp40_freq_scaled", []).append(0.0)
                self._metrics[mode].setdefault("hipo/exp40_freq_lowbin_ratio_mean", []).append(0.0)
        else:
            self._metrics[mode].setdefault("hipo/exp40_wrong_short_unscaled", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp40_wrong_short_scaled", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp40_wrong_short_rate", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp40_freq_unscaled", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp40_freq_scaled", []).append(0.0)
            self._metrics[mode].setdefault("hipo/exp40_freq_lowbin_ratio_mean", []).append(0.0)

        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        return (loss + extra + exp40_extra) / normalizer
