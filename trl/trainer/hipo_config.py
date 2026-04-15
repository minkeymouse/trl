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

"""HiPO (group-sampled policy optimization): extends [`GRPOConfig`] with representation auxiliaries."""

from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class HiPOConfig(GRPOConfig):
    r"""
    Configuration for [`HiPOTrainer`] — same as [`GRPOConfig`] plus optional representation auxiliaries
    (no human labels, no teacher).

    **Default aggregation** is GDPO-style ``normalize_then_sum`` (see ``GRPOConfig.multi_objective_aggregation``):
    each reward function is normalized within the prompt group before the weighted sum (unlike vanilla GRPO's
    default ``sum_then_normalize``). Override with ``multi_objective_aggregation=\"sum_then_normalize\"`` if desired.

    Note: this trainer no longer includes a wrong-length shaping term; length control should be done via
    token budgets (e.g., ``max_completion_length``) and evaluation budgets.
    """

    multi_objective_aggregation: str = field(
        default="normalize_then_sum",
        metadata={
            "help": "HiPO default: `'normalize_then_sum'` (GDPO / group-wise per-reward normalization before weighted "
            "sum). Set to `'sum_then_normalize'` to match vanilla GRPO aggregation."
        },
    )
    correctness_reward_func_index: int = field(
        default=0,
        metadata={
            "help": "Column index in `rewards_per_func` used as correctness signal "
            "(>= `correctness_threshold` counts as correct)."
        },
    )
    correctness_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold on the correctness reward column for the wrong/correct split."},
    )
    fft_trace_aux_lambda: float = field(
        default=0.02,
        metadata={
            "help": "HiPO-defining FFT auxiliary loss coefficient: pull wrong traces toward the in-group correct "
            "trace FFT prototype. Must be > 0."
        },
    )
    fft_trace_aux_detach_correct: bool = field(
        default=True,
        metadata={
            "help": "If True, stop gradient through correct-trace prototype in FFT aux loss "
            "(only wrong traces are directly pulled)."
        },
    )
    fft_trace_aux_batch_fallback: bool = field(
        default=True,
        metadata={
            "help": "If True, when no mixed correct/wrong prompt-group is available in the current micro-batch, "
            "fallback to a batch-level correct prototype so FFT aux still contributes gradients."
        },
    )
    fft_trace_aux_distance: str = field(
        default="cosine",
        metadata={
            "help": "Distance for FFT aux pull. One of: `cosine`, `mse`. `cosine` is more scale-robust for "
            "normalized spectral features."
        },
    )
    fft_trace_aux_proto_momentum: float = field(
        default=0.95,
        metadata={
            "help": "EMA momentum for cross-step correct-trace prototype memory used when in-batch pairing is not "
            "available."
        },
    )
    fft_trace_aux_proto_warmup_correct: int = field(
        default=1,
        metadata={
            "help": "Minimum number of prototype updates before using memory-prototype FFT aux on wrong samples."
        },
    )
    fft_trace_aux_num_bins: int = field(
        default=32,
        metadata={
            "help": "Number of spectral bins after adaptive pooling. Keeps FFT feature dimension fixed across "
            "variable completion lengths."
        },
    )
    fft_trace_aux_first_n_completion_tokens: int = field(
        default=0,
        metadata={
            "help": "If > 0, FFT auxiliary loss only uses the first N **completion** tokens per sequence "
            "(masking later tokens). 0 disables (use full completion)."
        },
    )
    hidden_direction_aux_lambda: float = field(
        default=0.0,
        metadata={
            "help": "Optional direct hidden-direction auxiliary coefficient. If > 0, wrong traces are pulled "
            "toward a correct hidden prototype in representation space."
        },
    )
    hidden_direction_aux_distance: str = field(
        default="cosine",
        metadata={
            "help": "Distance for hidden-direction aux pull. One of: `cosine`, `mse`."
        },
    )
    hidden_direction_aux_wrong_only: bool = field(
        default=True,
        metadata={
            "help": "If True, apply hidden-direction aux only on wrong traces. If False, all traces can be used."
        },
    )
    hidden_direction_aux_first_n_completion_tokens: int = field(
        default=0,
        metadata={
            "help": "If > 0, hidden-direction aux only uses first N completion tokens per sequence (0 = full)."
        },
    )
    hidden_direction_aux_detach_correct: bool = field(
        default=True,
        metadata={
            "help": "If True, stop gradient through correct hidden prototype in hidden-direction aux."
        },
    )
    hidden_direction_aux_batch_fallback: bool = field(
        default=True,
        metadata={
            "help": "If True, when no mixed prompt-group pairs are available, use a batch-level prototype fallback."
        },
    )
    hidden_direction_aux_proto_momentum: float = field(
        default=0.95,
        metadata={
            "help": "EMA momentum for cross-step correct hidden prototype memory in hidden-direction aux."
        },
    )
    hidden_direction_aux_proto_warmup_correct: int = field(
        default=1,
        metadata={
            "help": "Minimum number of prototype updates before memory fallback is used in hidden-direction aux."
        },
    )
    exp39_arm: str = field(
        default="",
        metadata={
            "help": "Exp39 auxiliary matrix arm id: a0..a8. Empty string keeps legacy fft_trace + hidden_direction composition."
        },
    )
    exp39_pairwise_margin_m: float = field(
        default=0.15,
        metadata={"help": "Exp39 a5: hinge margin m in relu(m - cos(w,c) + cos(w,n)) for pairwise direction aux."},
    )
    exp39_entropy_gate_std: float = field(
        default=0.5,
        metadata={
            "help": "Exp39 a6: keep dir-style pull only for sequences with mean completion entropy above "
            "median + this many (batch) std deviations."
        },
    )
    exp40_enable: bool = field(
        default=False,
        metadata={"help": "Exp40: enable wrong-only short penalty + early frequency-collapse penalty."},
    )
    exp40_wrong_short_lambda: float = field(
        default=0.0,
        metadata={"help": "Exp40: scale for wrong-only short-completion penalty (0 disables that term)."},
    )
    exp40_wrong_short_min_tokens: int = field(
        default=32,
        metadata={"help": "Exp40: wrong completions shorter than this many tokens incur penalty."},
    )
    exp40_wrong_short_shape: str = field(
        default="hinge",
        metadata={"help": "Exp40: 'hinge' or 'linear' shortfall shape vs min_tokens."},
    )
    exp40_freq_lambda: float = field(
        default=0.0,
        metadata={"help": "Exp40: scale for low-frequency energy concentration on wrong traces (0 disables)."},
    )
    exp40_freq_warmup_steps: int = field(
        default=200,
        metadata={"help": "Exp40: apply frequency penalty only for global_step <= this value."},
    )
    exp40_freq_lowbin_ratio: float = field(
        default=0.25,
        metadata={
            "help": "Exp40: fraction of FFT bins (from start) used as 'low frequency' for collapse proxy [0,1]."
        },
    )
    pivot_aux_enable_hard_gate: bool = field(
        default=True,
        metadata={
            "help": "If True, skip HiPO auxiliary losses (FFT/dir/Exp39/Exp40 extras) when the micro-batch has no "
            "usable correct/wrong pivot (all one class, too few valid samples, or class ratio below threshold)."
        },
    )
    pivot_aux_min_valid_samples: int = field(
        default=4,
        metadata={
            "help": "Minimum number of valid completions (non-empty completion_mask) required before aux runs."
        },
    )
    pivot_aux_min_class_ratio: float = field(
        default=0.10,
        metadata={
            "help": "Minimum fraction of valid samples that must be correct and wrong each (e.g. 0.1 => both classes "
            ">=10% of valid). Prevents aux on extremely imbalanced batches."
        },
    )
    pivot_shake_enable: bool = field(
        default=False,
        metadata={
            "help": "Exp70: entropy-gated hidden-state perturbation. Adds gaussian noise at the top-k% highest "
            "entropy token positions of wrong traces before FFT feature extraction. Adds diversity specifically "
            "at 'confused' decision points without disturbing correct traces."
        },
    )
    pivot_shake_topk: float = field(
        default=0.05,
        metadata={
            "help": "Exp70: fraction of completion tokens (per sequence) selected as high-entropy pivot positions "
            "for hidden-state perturbation. E.g. 0.05 = top 5% highest-entropy tokens."
        },
    )
    pivot_shake_scale: float = field(
        default=0.05,
        metadata={
            "help": "Exp70: gaussian noise scale (std) injected at pivot token hidden states. "
            "Relative to unit normal; keep small (0.01–0.10) to avoid destabilizing representations."
        },
    )
