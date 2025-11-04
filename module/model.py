"""Core model definition for the NeuroFM foundation model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EventDesign:
    """Container for event regressors."""

    boxcars: torch.Tensor
    parametric: torch.Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer inputs."""

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class CanonicalHRF(nn.Module):
    """Canonical hemodynamic response function used for convolution."""

    def __init__(self, time_steps: int, dt: float = 0.5) -> None:
        super().__init__()
        self.dt = dt
        time = torch.arange(0, time_steps * dt, dt)
        peak_values = self._gamma_pdf(time, 6)
        undershoot = self._gamma_pdf(time, 16)
        hrf = peak_values - 0.35 * undershoot
        hrf /= hrf.sum() + 1e-8
        self.register_buffer("kernel", hrf.view(1, 1, -1))

    @staticmethod
    def _gamma_pdf(x: torch.Tensor, shape: float) -> torch.Tensor:
        return x.pow(shape - 1) * torch.exp(-x) / math.gamma(shape)

    def forward(self, boxcars: torch.Tensor) -> torch.Tensor:
        batch, num_events, time_points = boxcars.shape
        signal = boxcars.view(batch * num_events, 1, time_points)
        conv = F.conv1d(signal, self.kernel, padding=self.kernel.size(-1) - 1)
        conv = conv[:, :, :time_points]
        return conv.view(batch, num_events, time_points)


class TimeseriesEncoder(nn.Module):
    """Encoder for high-dimensional timeseries."""

    def __init__(self, input_dim: int, reduction_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, reduction_dim)
        self.norm = nn.LayerNorm(reduction_dim)
        self.temporal = nn.GRU(reduction_dim, embed_dim, batch_first=True)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        # Input expected in shape (batch, features, time)
        series = series.permute(0, 2, 1)
        projected = self.project(series)
        projected = self.norm(projected)
        _, hidden = self.temporal(projected)
        return hidden.squeeze(0)


class SwiFTExpert(nn.Module):
    """Profile-specific expert used by the SwiFT backbone."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embed_dim: int,
        dilation: int,
        profile: str,
    ) -> None:
        super().__init__()
        groups = math.gcd(hidden_channels, 16) or 1
        self.profile = profile
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.norm = nn.GroupNorm(groups, hidden_channels)
        self.activation = nn.GELU()
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_dim),
        )
        self.frequency_mixer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
        )
        self.temporal_head: nn.Module | None = None
        self.motion_head: nn.Module | None = None
        self.residual_head: nn.Module | None = None
        self.parcellation_head: nn.Module | None = None
        self.smoother: nn.Module | None = None

        if profile == "hcp_minimal":
            self.smoother = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=5,
                padding=2,
                groups=in_channels,
                bias=False,
            )
        elif profile == "hrf_optimized":
            self.temporal_head = nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4),
                nn.GELU(),
                nn.Conv1d(hidden_channels, embed_dim, kernel_size=1),
            )
        elif profile == "motion_denoised":
            self.motion_head = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, embed_dim),
            )
        elif profile == "glm_residual":
            self.residual_head = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, embed_dim),
            )
        elif profile == "parcellation_based":
            roi_dim = in_channels * 64
            self.parcellation_head = nn.Sequential(
                nn.Linear(roi_dim, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, embed_dim),
            )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        source = volume
        if self.smoother is not None:
            source = self.smoother(volume)

        local = self.depthwise(source)
        local = self.pointwise(local)
        local = self.norm(local)
        local = self.activation(local)
        pooled = F.adaptive_avg_pool3d(local, 1).flatten(1)

        freq = torch.fft.rfftn(source, dim=(-3, -2, -1), norm="ortho")
        freq_descriptor = freq.abs().mean(dim=(-3, -2, -1))
        features = pooled + self.frequency_mixer(freq_descriptor)

        if self.temporal_head is not None:
            temporal_series = volume.mean(dim=(-3, -2, -1)).unsqueeze(1)
            temporal_features = self.temporal_head(temporal_series).squeeze(1)
            features = features + temporal_features

        if self.motion_head is not None:
            high_pass = volume - F.avg_pool3d(volume, kernel_size=5, stride=1, padding=2)
            motion_summary = high_pass.abs().mean(dim=(-3, -2, -1))
            features = features + self.motion_head(motion_summary)

        if self.residual_head is not None:
            baseline = volume.mean(dim=1, keepdim=True)
            residual = volume - baseline
            residual_summary = residual.mean(dim=(-3, -2, -1))
            features = features + self.residual_head(residual_summary)

        if self.parcellation_head is not None:
            roi_grid = F.adaptive_avg_pool3d(volume, output_size=(4, 4, 4))
            roi_descriptor = roi_grid.flatten(start_dim=1)
            features = features + self.parcellation_head(roi_descriptor)

        return self.channel_mixer(features)


class SwiFTBackbone(nn.Module):
    """Mixture-of-experts backbone combining spatial and frequency cues."""

    DEFAULT_PROFILES = [
        "hcp_minimal",
        "hrf_optimized",
        "motion_denoised",
        "glm_residual",
        "parcellation_based",
    ]

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        expert_profiles: list[str] | None = None,
    ) -> None:
        super().__init__()
        if expert_profiles is None:
            expert_profiles = self.DEFAULT_PROFILES

        hidden_channels = max(embed_dim * 2, 128)
        dilation_map = {
            "hcp_minimal": 1,
            "hrf_optimized": 1,
            "motion_denoised": 2,
            "glm_residual": 3,
            "parcellation_based": 1,
        }

        self.experts = nn.ModuleList(
            SwiFTExpert(
                in_channels,
                hidden_channels,
                embed_dim,
                dilation_map.get(profile, 1),
                profile,
            )
            for profile in expert_profiles
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d(1)
        self.gating = nn.Sequential(
            nn.Linear(in_channels * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, len(self.experts)),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        spatial_summary = self.spatial_pool(volume).flatten(1)
        temporal_summary = volume.reshape(volume.size(0), volume.size(1), -1).std(dim=-1)
        gate_input = torch.cat([spatial_summary, temporal_summary], dim=-1)
        gate_logits = self.gating(gate_input)
        weights = gate_logits.softmax(dim=-1)

        expert_outputs = torch.stack([expert(volume) for expert in self.experts], dim=1)
        weighted = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return weighted


class SwiFTVolumeEncoder(nn.Module):
    """SwiFT-based encoder specialised for 4D fMRI volumes."""

    def __init__(self, time_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.backbone = SwiFTBackbone(time_channels, embed_dim)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        # Input expected in shape (batch, X, Y, Z, time)
        if volume.dim() != 5:
            raise ValueError(
                "SwiFTVolumeEncoder expects inputs shaped (batch, X, Y, Z, time)"
            )
        volume = volume.permute(0, 4, 1, 2, 3)
        return self.backbone(volume)


class MetadataEncoder(nn.Module):
    """Simple feed-forward encoder for acquisition metadata."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.network(metadata)


class NeuroFMModel(nn.Module):
    """Mixture-of-modalities foundation model for multi-modal fMRI."""

    def __init__(
        self,
        hcp_dim: int,
        parcellation_dim: int,
        metadata_dim: int,
        event_param_dim: int,
        num_events: int,
        time_points: int,
        volume_time_channels: int,
        embed_dim: int = 256,
        num_factors: int = 16,
    ) -> None:
        super().__init__()

        hcp_reduction = max(128, hcp_dim // 64)
        parcel_reduction = max(128, parcellation_dim // 4)

        self.hcp_encoder = TimeseriesEncoder(hcp_dim, hcp_reduction, embed_dim)
        self.parcellation_encoder = TimeseriesEncoder(
            parcellation_dim, parcel_reduction, embed_dim
        )
        self.volume_encoder = SwiFTVolumeEncoder(volume_time_channels, embed_dim)
        self.metadata_encoder = MetadataEncoder(metadata_dim, embed_dim)

        self.hrf = CanonicalHRF(time_points)
        self.event_encoder = nn.Sequential(
            nn.Linear(time_points + event_param_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.positional = SinusoidalPositionalEncoding(embed_dim, max_len=num_events + 5)
        self.subject_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.factor_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_factors),
        )
        self.calibration_head = nn.Sequential(
            nn.Linear(num_factors, num_factors),
            nn.GELU(),
            nn.Linear(num_factors, 1),
        )

        self.num_events = num_events
        self._reset_parameters()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_events(self, design: EventDesign) -> torch.Tensor:
        hrf_aligned = self.hrf(design.boxcars)
        summary = hrf_aligned.view(hrf_aligned.size(0), hrf_aligned.size(1), -1)
        features = torch.cat([summary, design.parametric], dim=-1)
        batch, events, _ = features.shape
        encoded = self.event_encoder(features.view(batch * events, -1))
        return encoded.view(batch, events, -1)

    def _assemble_tokens(
        self,
        subject_token: torch.Tensor,
        metadata_token: torch.Tensor,
        hcp_token: torch.Tensor,
        parcel_token: torch.Tensor,
        volume_token: torch.Tensor,
        event_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = [
            subject_token,
            metadata_token.unsqueeze(0),
            hcp_token.unsqueeze(0),
            parcel_token.unsqueeze(0),
            volume_token.unsqueeze(0),
            event_tokens.permute(1, 0, 2),
        ]
        return torch.cat(tokens, dim=0)

    # ------------------------------------------------------------------
    # Forward API
    # ------------------------------------------------------------------

    def forward(
        self,
        hcp: torch.Tensor,
        parcellation: torch.Tensor,
        volume: torch.Tensor,
        metadata: torch.Tensor,
        events: EventDesign,
    ) -> Dict[str, torch.Tensor]:
        batch_size = hcp.size(0)

        hcp_embed = self.hcp_encoder(hcp)
        parcel_embed = self.parcellation_encoder(parcellation)
        volume_embed = self.volume_encoder(volume)
        metadata_embed = self.metadata_encoder(metadata)
        event_tokens = self._encode_events(events)

        subject_token = self.subject_token.expand(-1, batch_size, -1)
        transformer_input = self._assemble_tokens(
            subject_token,
            metadata_embed,
            hcp_embed,
            parcel_embed,
            volume_embed,
            event_tokens,
        )
        transformer_input = self.positional(transformer_input)
        encoded = self.transformer(transformer_input)

        subject_representation = encoded[0]
        event_representations = encoded[5:]

        factors = self.factor_head(event_representations)
        factor_summary = factors.mean(dim=0)
        prediction = self.calibration_head(factor_summary)

        return {
            "subject_embedding": subject_representation,
            "event_factors": factors.permute(1, 0, 2),
            "prediction": prediction,
        }

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.subject_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

