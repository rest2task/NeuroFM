"""Utility helpers for NeuroFM.

The utilities centralise common tensor manipulation and IO routines used
throughout the project.  Keeping them in a single module ensures the model
and training code stay focused on high level logic instead of bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def load_array(path: Path) -> torch.Tensor:
    """Load an array from ``.pt`` or ``.npy`` files.

    Parameters
    ----------
    path:
        Location of the tensor on disk.
    """

    suffix = path.suffix
    if suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if suffix == ".npy":
        data = np.load(path)
        return torch.from_numpy(data)
    raise ValueError(f"Unsupported tensor format: {path}")


def ensure_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Cast tensors to ``float32`` for model consumption."""

    if tensor.dtype in {torch.float16, torch.float32, torch.float64}:
        return tensor.float()
    return tensor.float()


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load the dataset manifest describing the location of every sample."""

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("The dataset manifest must be a list of samples")
    return data


def stack_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Stack tensors ensuring they share the same shape."""

    tensors = list(tensors)
    reference_shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != reference_shape:
            raise ValueError(
                "All tensors in a batch must share the same shape. "
                f"Got {tensor.shape} and {reference_shape}."
            )
    return torch.stack(tensors, dim=0)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a dictionary of tensors to the requested device."""

    return {key: value.to(device) for key, value in batch.items()}


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

