# HiPO (Hidden Preference Optimization) — Nemotron Kaggle research extension for TRL DPO.
# Time-axis low-frequency energy share on last-layer hidden states (completion span only).
# Mirrors numerics in ``nemotron_lab.feasibility.hidden_fft`` / precompute ``scripts/exp34_precompute_spectral_targets.py``.
from __future__ import annotations

import torch


def _rfft_time_power(x: torch.Tensor) -> torch.Tensor:
    z = torch.fft.rfft(x, dim=0)
    return z.abs() ** 2


def _power_sum_nonfreq_bins(power: torch.Tensor) -> torch.Tensor:
    if power.ndim == 1:
        return power
    return power.sum(dim=tuple(range(1, power.ndim)))


def _low_high_split(
    power: torch.Tensor,
    *,
    low_freq_frac: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_bins = int(power.shape[0])
    if n_bins <= 1:
        k_cut = 1
    else:
        k_cut = max(1, min(n_bins, int(float(low_freq_frac) * n_bins)))
    low = power[:k_cut].sum()
    high = power[k_cut:].sum()
    return low, high


def low_freq_ratio(x: torch.Tensor, *, low_freq_frac: float, eps: float = 1e-8) -> torch.Tensor:
    power = _rfft_time_power(x)
    power_1d = _power_sum_nonfreq_bins(power)
    low, high = _low_high_split(power_1d, low_freq_frac=low_freq_frac, eps=eps)
    return low / (low + high + eps)


def batch_completion_low_freq_r(
    hidden_last: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    low_freq_frac: float,
) -> torch.Tensor:
    """Per-sequence scalar R on completion tokens; hidden_last [B,L,H], mask [B,L] (1 on completion)."""
    b, _, _ = hidden_last.shape
    out = []
    for i in range(b):
        m = completion_mask[i].bool()
        hc = hidden_last[i][m]
        if hc.shape[0] < 2:
            out.append(hidden_last.new_tensor(0.5))
        else:
            out.append(low_freq_ratio(hc, low_freq_frac=low_freq_frac))
    return torch.stack(out, dim=0)


def batch_completion_mean_hidden_l2(
    hidden_last: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence mean L2 norm over completion token hidden states [B,L,H], mask [B,L]."""
    b, _, _ = hidden_last.shape
    out: list[torch.Tensor] = []
    for i in range(b):
        m = completion_mask[i].bool()
        hc = hidden_last[i][m]
        if hc.shape[0] == 0:
            out.append(hidden_last.new_tensor(0.0))
        else:
            norms = torch.linalg.vector_norm(hc, dim=-1)
            out.append(norms.mean())
    return torch.stack(out, dim=0)
