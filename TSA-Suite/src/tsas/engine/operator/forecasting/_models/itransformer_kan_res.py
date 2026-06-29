# -*- coding: utf-8 -*-

"""
iTransformer regressor with KAN prediction head and residual prediction strategy.

This variant uses the NPU-compatible dense encoder (MoE ablated by ``num_experts=1``)
and applies target flatline masking to prevent autoregressive copying.

Adapted from the HBHD industrial forecasting project.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsas.engine.operator.forecasting._models.npu_transformer import (
    NPUMoETransformerEncoder,
    NPUMoETransformerEncoderLayer,
)

__all__ = [
    'KANLinear',
    'KAN',
    'LagAwareRefiner',
    'ITransformerRegressor',
]


class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network linear layer."""

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output


class KAN(nn.Module):
    """Multi-layer KAN."""

    def __init__(self, layers_hidden, grid_size=5, spline_order=3, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    **kwargs
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


def shift_pad_left(x: torch.Tensor, lag: int) -> torch.Tensor:
    """Shift sequence left by ``lag`` steps and zero-pad the beginning."""
    if lag == 0:
        return x
    B, L, F = x.shape
    z = x.new_zeros(B, lag, F)
    return torch.cat([z, x[:, :L - lag, :]], dim=1)


class LagAwareRefiner(nn.Module):
    """Cross-correlation-based lag-aware target token refiner."""

    def __init__(self, d_model: int, nhead: int, seq_len: int, num_features: int,
                 token_proj: nn.Linear, feature_embed: nn.Parameter,
                 target_idx: int, max_lag: int = 16,
                 bias_scale: float = 2.0, attn_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.target_idx = int(target_idx)
        self.max_lag = int(max_lag)
        self.bias_scale = float(bias_scale)
        self.token_proj = token_proj
        self.feature_embed = feature_embed
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(attn_dropout)

    def _build_shifted_tokens(self, x_seq: torch.Tensor) -> torch.Tensor:
        toks = []
        for lag in range(self.max_lag + 1):
            xs = shift_pad_left(x_seq, lag)
            xs = xs.permute(0, 2, 1)
            tok = self.token_proj(xs) + self.feature_embed
            toks.append(tok)
        return torch.stack(toks, dim=1)

    @torch.no_grad()
    def _xcorr_prior(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, L, F = x_seq.shape
        x_target = x_seq[:, :, self.target_idx]
        x_target_centered = x_target - x_target.mean(dim=1, keepdim=True)
        norm_target = torch.sqrt((x_target_centered ** 2).sum(dim=1, keepdim=True) + 1e-6)
        priors = []
        for lag in range(self.max_lag + 1):
            x_shifted = shift_pad_left(x_seq, lag)
            x_shifted_centered = x_shifted - x_shifted.mean(dim=1, keepdim=True)
            norm_shifted = torch.sqrt((x_shifted_centered ** 2).sum(dim=1, keepdim=True) + 1e-6)
            numerator = (x_target_centered.unsqueeze(-1) * x_shifted_centered).sum(dim=1)
            denominator = (norm_target * norm_shifted.squeeze(1) + 1e-6)
            r = (numerator / denominator).clamp(-1.0, 1.0)
            priors.append(r)
        return torch.stack(priors, dim=1)

    def forward(self, x_seq: torch.Tensor, enc_tokens: torch.Tensor, tgt_vec: torch.Tensor) -> torch.Tensor:
        B, L, F = x_seq.shape
        K_lag_total = self.max_lag + 1
        tok_shifted = self._build_shifted_tokens(x_seq)
        K = self.k_proj(tok_shifted)
        V = self.v_proj(tok_shifted)
        Q = self.q_proj(tgt_vec).view(B, self.nhead, 1, self.d_head)
        Kh = K.view(B, K_lag_total * F, self.nhead, self.d_head).transpose(1, 2)
        Vh = V.view(B, K_lag_total * F, self.nhead, self.d_head).transpose(1, 2)
        logits = torch.matmul(Q, Kh.transpose(-2, -1)) / np.sqrt(self.d_head)
        prior = self._xcorr_prior(x_seq)
        prior_bias = (self.bias_scale * prior).view(B, 1, 1, K_lag_total * F)
        logits = logits + prior_bias
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, Vh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, self.d_model)
        out = self.o_proj(ctx)
        tgt_new = self.ln(tgt_vec + out)
        return tgt_new


class ITransformerRegressor(nn.Module):
    """iTransformer regressor with dense NPU encoder, KAN head, and residual prediction."""

    def __init__(self, seq_len: int, num_features: int, out_dim: int, target_idx: int,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.2,
                 step_cond_head: bool = True, lag_aware: bool = True, lag_max: int = 16,
                 lag_bias_scale: float = 2.0, lag_dropout: float = 0.1,
                 kan_grid_size: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.target_idx = int(target_idx)
        self.step_cond_head = step_cond_head
        self.lag_aware = lag_aware

        # Tokenization and embedding
        self.token_proj = nn.Linear(seq_len, d_model)
        self.feature_embed = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

        # Dense NPU-compatible encoder (MoE ablated with num_experts=1)
        enc_layer = NPUMoETransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            num_experts=1,
            top_k=1,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu"
        )
        self.encoder = NPUMoETransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        # Lag-aware refiner for the target token
        if lag_aware:
            self.lag_refiner = LagAwareRefiner(
                d_model=d_model, nhead=nhead, seq_len=seq_len, num_features=num_features,
                token_proj=self.token_proj, feature_embed=self.feature_embed,
                target_idx=self.target_idx, max_lag=lag_max,
                bias_scale=lag_bias_scale, attn_dropout=lag_dropout
            )

        # KAN prediction head
        if step_cond_head:
            self.step_embed = nn.Parameter(torch.zeros(out_dim, d_model))
            nn.init.normal_(self.step_embed, std=0.02)
            self.head_shared = KAN(
                layers_hidden=[d_model, d_model // 2, 1],
                grid_size=kan_grid_size
            )
        else:
            self.head = KAN(
                layers_hidden=[d_model, d_model // 2, out_dim],
                grid_size=kan_grid_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Target flatline masking: destroy autoregressive shortcut
        base_value = x[:, -1, self.target_idx]
        x_masked = x.clone()
        x_masked[:, :, self.target_idx] = base_value.unsqueeze(1).expand(-1, self.seq_len)

        # Encode
        xT = x_masked.permute(0, 2, 1)
        tok = self.token_proj(xT)
        tok = tok + self.feature_embed
        enc = self.encoder(tok)

        # Extract and refine target token
        tgt_vec = enc[:, self.target_idx, :]
        if self.lag_aware:
            tgt_vec = self.lag_refiner(x, enc, tgt_vec)

        # Predict residual
        if self.step_cond_head:
            B = x.size(0)
            H = tgt_vec.unsqueeze(1) + self.step_embed.unsqueeze(0)
            residual = self.head_shared(H.reshape(-1, H.size(-1))).view(B, -1)
        else:
            residual = self.head(tgt_vec)

        return residual
