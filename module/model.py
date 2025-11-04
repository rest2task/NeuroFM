"""Core model definition for the NeuroFM foundation model.

The implementation below mirrors the system described in the accompanying
manuscript.  It contains:

* Modality specific encoders for structural MRI, resting-state fMRI and
  task fMRI inputs.
* A mixture-of-experts encoder for task fMRI where a gating network chooses
  between multiple preprocessing pipelines.
* A differentiable hemodynamic response (HRF) module that aligns event
  regressors with the BOLD signal.
* A transformer backbone with a learnable subject token that fuses
  information across modalities and events.
* A reasoning layer that produces an event-by-factor matrix capturing the
  latent cognitive factors associated with each event.
* A lightweight calibration head that can be trained in a few-shot manner to
  predict subject level outcomes from the shared representation.

The design favours clarity and modularity so that new datasets or
preprocessing streams can be connected with minimal code changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility structures
# ---------------------------------------------------------------------------


@dataclass
class EventDesign:
    """Container for event regressors.

    Attributes
    ----------
    boxcars:
        Tensor of shape ``(batch, num_events, time_points)`` describing the
        event on/off timings before HRF convolution.
    parametric:
        Tensor of shape ``(batch, num_events, num_parametric)`` containing
        optional parametric modulators (e.g. task difficulty).
    """

    boxcars: torch.Tensor
    parametric: torch.Tensor


# ---------------------------------------------------------------------------
# Positional encoding utilities
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer inputs."""

    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input sequence.

        Parameters
        ----------
        x:
            Input tensor with shape ``(seq_len, batch, embed_dim)``.
        """

        seq_len = x.size(0)
        return x + self.pe[:seq_len]


# ---------------------------------------------------------------------------
# HRF modelling
# ---------------------------------------------------------------------------


class CanonicalHRF(nn.Module):
    """Canonical hemodynamic response function used for convolution."""

    def __init__(self, time_steps: int, dt: float = 0.5):
        super().__init__()
        self.dt = dt
        time = torch.arange(0, time_steps * dt, dt)
        # Canonical HRF approximated via a difference of gammas
        peak_values = self._gamma_pdf(time, 6)
        undershoot = self._gamma_pdf(time, 16)
        hrf = peak_values - 0.35 * undershoot
        hrf /= hrf.sum() + 1e-8
        self.register_buffer("kernel", hrf.view(1, 1, -1))

    @staticmethod
    def _gamma_pdf(x: torch.Tensor, shape: float) -> torch.Tensor:
        return x.pow(shape - 1) * torch.exp(-x) / (math.gamma(shape))

    def forward(self, boxcars: torch.Tensor) -> torch.Tensor:
        """Apply HRF convolution to event boxcars.

        Parameters
        ----------
        boxcars:
            Tensor of shape ``(batch, num_events, time_points)``.
        """

        batch, num_events, time_points = boxcars.shape
        signal = boxcars.view(batch * num_events, 1, time_points)
        conv = F.conv1d(signal, self.kernel, padding=self.kernel.size(-1) - 1)
        conv = conv[:, :, :time_points]
        return conv.view(batch, num_events, time_points)


# ---------------------------------------------------------------------------
# Expert encoders and gating
# ---------------------------------------------------------------------------


class ExpertEncoder(nn.Module):
    """Simple temporal encoder used by each task-fMRI expert."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, time, features)``.
        """

        _, hidden = self.rnn(x)
        return self.proj(hidden.squeeze(0))


class GatingNetwork(nn.Module):
    """Metadata driven gating for the mixture-of-experts module."""

    def __init__(self, metadata_dim: int, num_experts: int):
        super().__init__()
        hidden = max(metadata_dim // 2, 1)
        self.layers = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts)
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        logits = self.layers(metadata)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Core NeuroFM model
# ---------------------------------------------------------------------------


class NeuroFMModel(nn.Module):
    """Mixture-of-experts foundation model for multi-modal fMRI."""

    def __init__(
        self,
        struct_dim: int,
        rest_dim: int,
        task_dim: int,
        event_param_dim: int,
        time_points: int,
        num_events: int,
        metadata_dim: int,
        embed_dim: int = 128,
        expert_hidden: int = 128,
        num_experts: int = 5,
        num_factors: int = 16,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_events = num_events
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_factors = num_factors

        # Structural encoder (voxel based structural context)
        self.structural_encoder = nn.Sequential(
            nn.Linear(struct_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Resting-state encoder (captures spontaneous dynamics)
        self.rest_encoder = nn.GRU(rest_dim, embed_dim, batch_first=True)

        # Task mixture-of-experts encoders
        self.experts = nn.ModuleList(
            [ExpertEncoder(task_dim, expert_hidden, embed_dim) for _ in range(num_experts)]
        )
        self.gating = GatingNetwork(metadata_dim, num_experts)

        # Event encoders and HRF modelling
        self.hrf = CanonicalHRF(time_points)
        event_input_dim = time_points + event_param_dim
        self.event_encoder = nn.Sequential(
            nn.Linear(event_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Transformer backbone with subject token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.positional = SinusoidalPositionalEncoding(embed_dim, max_len=num_events + 4)
        self.subject_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Reasoning layer producing event-by-factor matrix
        self.factor_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_factors)
        )

        # Few-shot calibration head operating on event factors
        self.calibration_head = nn.Sequential(
            nn.Linear(num_factors, num_factors),
            nn.GELU(),
            nn.Linear(num_factors, output_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.subject_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _encode_rest(self, rest: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rest_encoder(rest)
        return hidden.squeeze(0)

    def _encode_events(self, design: EventDesign) -> torch.Tensor:
        hrf_aligned = self.hrf(design.boxcars)
        # Combine HRF aligned response with parametric modulators
        summary = hrf_aligned  # (batch, events, time)
        summary = summary.view(summary.size(0), summary.size(1), -1)
        features = torch.cat([summary, design.parametric], dim=-1)
        batch, events, _ = features.shape
        features = features.view(batch * events, -1)
        encoded = self.event_encoder(features)
        return encoded.view(batch, events, -1)

    # ------------------------------------------------------------------
    # Forward API
    # ------------------------------------------------------------------

    def forward(
        self,
        structural: torch.Tensor,
        resting: torch.Tensor,
        task_experts: Sequence[torch.Tensor],
        metadata: torch.Tensor,
        events: EventDesign,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for NeuroFM.

        Parameters
        ----------
        structural:
            Tensor of shape ``(batch, struct_dim)``.
        resting:
            Tensor of shape ``(batch, time_steps, rest_dim)``.
        task_experts:
            Sequence containing ``num_experts`` tensors each of shape
            ``(batch, time_steps, task_dim)`` representing different
            preprocessing streams.
        metadata:
            Tensor of shape ``(batch, metadata_dim)`` describing acquisition
            parameters used by the gating network.
        events:
            :class:`EventDesign` describing task events.
        """

        batch_size = structural.size(0)
        assert len(task_experts) == self.num_experts, (
            f"Expected {self.num_experts} expert inputs, received {len(task_experts)}"
        )

        # Encode modalities
        structural_embed = self.structural_encoder(structural)
        rest_embed = self._encode_rest(resting)

        expert_embeddings = [expert(x) for expert, x in zip(self.experts, task_experts)]
        expert_stack = torch.stack(expert_embeddings, dim=1)
        weights = self.gating(metadata).unsqueeze(-1)
        task_embed = torch.sum(expert_stack * weights, dim=1)

        event_tokens = self._encode_events(events)

        # Assemble transformer tokens: subject, structural, resting, task, events
        subject_token = self.subject_token.expand(-1, batch_size, -1)
        structural_token = structural_embed.unsqueeze(0)
        rest_token = rest_embed.unsqueeze(0)
        task_token = task_embed.unsqueeze(0)
        event_tokens_seq = event_tokens.permute(1, 0, 2)  # (events, batch, embed)

        transformer_input = torch.cat(
            [subject_token, structural_token, rest_token, task_token, event_tokens_seq],
            dim=0,
        )
        transformer_input = self.positional(transformer_input)
        encoded = self.transformer(transformer_input)

        # Extract outputs
        subject_representation = encoded[0]
        event_representations = encoded[4:]

        factors = self.factor_head(event_representations)
        # Aggregate for prediction (mean pooling across events)
        factor_summary = factors.mean(dim=0)
        prediction = self.calibration_head(factor_summary)

        return {
            "subject_embedding": subject_representation,
            "event_factors": factors.permute(1, 0, 2),  # (batch, events, factors)
            "prediction": prediction,
        }
