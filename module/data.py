"""Dataset and dataloader utilities for NeuroFM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .model import EventDesign
from .utils import ensure_float_tensor, load_array, read_manifest, stack_tensors


@dataclass
class SamplePaths:
    """Simple container describing the files required for a single sample."""

    hcp: Path
    parcellation: Path
    volume: Path
    metadata: Path
    event_boxcars: Path
    event_parametric: Path
    target: Path

    @classmethod
    def from_dict(cls, item: Dict[str, str]) -> "SamplePaths":
        return cls(
            hcp=Path(item["hcp"]),
            parcellation=Path(item["parcellation"]),
            volume=Path(item["volume"]),
            metadata=Path(item["metadata"]),
            event_boxcars=Path(item["event_boxcars"]),
            event_parametric=Path(item["event_parametric"]),
            target=Path(item["target"]),
        )


class NeuroFMDataset(Dataset):
    """Dataset backed by a JSON manifest describing each sample."""

    def __init__(self, manifest_path: Path) -> None:
        super().__init__()
        manifest = read_manifest(manifest_path)
        self.samples: List[SamplePaths] = [SamplePaths.from_dict(item) for item in manifest]

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.samples)

    def _load_events(self, sample: SamplePaths) -> EventDesign:
        boxcars = ensure_float_tensor(load_array(sample.event_boxcars))
        parametric = ensure_float_tensor(load_array(sample.event_parametric))
        return EventDesign(boxcars=boxcars, parametric=parametric)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        hcp = ensure_float_tensor(load_array(sample.hcp))
        parcellation = ensure_float_tensor(load_array(sample.parcellation))
        volume = ensure_float_tensor(load_array(sample.volume))
        metadata = ensure_float_tensor(load_array(sample.metadata))
        target = ensure_float_tensor(load_array(sample.target))
        events = self._load_events(sample)
        return {
            "hcp": hcp,
            "parcellation": parcellation,
            "volume": volume,
            "metadata": metadata,
            "events": events,
            "target": target,
        }


def collate_samples(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate samples returned by :class:`NeuroFMDataset`."""

    hcp = stack_tensors(item["hcp"] for item in batch)
    parcellation = stack_tensors(item["parcellation"] for item in batch)
    volume = stack_tensors(item["volume"] for item in batch)
    metadata = stack_tensors(item["metadata"] for item in batch)
    targets = stack_tensors(item["target"] for item in batch)

    boxcars = stack_tensors(item["events"].boxcars for item in batch)
    parametric = stack_tensors(item["events"].parametric for item in batch)

    return {
        "hcp": hcp,
        "parcellation": parcellation,
        "volume": volume,
        "metadata": metadata,
        "events": EventDesign(boxcars=boxcars, parametric=parametric),
        "target": targets,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create a dataloader with the repository-wide collate function."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_samples,
    )

