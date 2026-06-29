# -*- coding: utf-8 -*-

"""
NPU-compatible Transformer components.

Avoids ``torch.nn.MultiheadAttention`` entirely and uses only basic ops:
``Linear``, ``matmul``, ``softmax``, ``dropout``, ``LayerNorm``.

This module is adapted from the HBHD industrial forecasting project.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'NPUMultiheadAttention',
    'NPUSparseMoE',
    'NPUMoETransformerEncoderLayer',
    'NPUMoETransformerEncoder',
]


class NPUMultiheadAttention(nn.Module):
    """Multi-Head Self-Attention built from basic ops only.

    Supports optional attention mask and key_padding_mask.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.batch_first = batch_first
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query, key, value: ``[B, T, D]`` (batch_first=True)
            attn_mask: ``[T, T]`` or ``[B, H, T, T]`` optional
            key_padding_mask: ``[B, T]`` bool mask (True = padded position)

        Returns:
            output: ``[B, T, D]``
            attn_weights: ``[B, H, T, T]``
        """
        B = query.size(0)

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape to [B, H, T, d_head]
        Q = Q.view(B, -1, self.nhead, self.d_head).transpose(1, 2)
        K = K.view(B, -1, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(B, -1, self.nhead, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, T, T]

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)           # [B, H, T, d_head]
        out = out.transpose(1, 2).contiguous()        # [B, T, H, d_head]
        out = out.view(B, -1, self.d_model)           # [B, T, D]

        output = self.o_proj(out)
        return output, attn_weights


class NPUSparseMoE(nn.Module):
    """Sparse Mixture of Experts for Transformer FFN."""

    def __init__(self, d_model, num_experts=8, top_k=2,
                 dim_feedforward=512, dropout=0.1, activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        act_func = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                act_func,
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)

        router_logits = self.router(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            token_indices, k_indices = torch.where(expert_mask)

            if len(token_indices) > 0:
                expert_inputs = x_flat[token_indices]
                expert_outputs = expert(expert_inputs)
                weights = routing_weights[token_indices, k_indices].unsqueeze(-1)
                final_output[token_indices] += expert_outputs * weights

        return final_output.view(*orig_shape)


class NPUMoETransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with MoE FFN and NPU-compatible attention.

    Uses Pre-LN structure for training stability.
    """

    def __init__(self, d_model, nhead, num_experts=8, top_k=2,
                 dim_feedforward=512, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = NPUMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = NPUSparseMoE(d_model, num_experts, top_k, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2,
                                  attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.moe(src2)
        src = src + src2
        return src


class NPUMoETransformerEncoder(nn.Module):
    """Stack of ``NPUMoETransformerEncoderLayer``."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        if self.norm is not None:
            src = self.norm(src)
        return src
