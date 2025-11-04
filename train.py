"""Training entry point for NeuroFM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from module.data import NeuroFMDataset, create_dataloader
from module.model import EventDesign, NeuroFMModel
from module.utils import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NeuroFM model")
    parser.add_argument("train_manifest", type=Path, help="Path to the training manifest")
    parser.add_argument("val_manifest", type=Path, help="Path to the validation manifest")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-factors", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def infer_model_kwargs(sample: Dict[str, torch.Tensor]) -> Dict[str, int]:
    events: EventDesign = sample["events"]
    return {
        "hcp_dim": sample["hcp"].shape[0],
        "parcellation_dim": sample["parcellation"].shape[0],
        "metadata_dim": sample["metadata"].shape[0],
        "event_param_dim": events.parametric.shape[-1],
        "num_events": events.boxcars.shape[0],
        "time_points": events.boxcars.shape[-1],
        "volume_time_channels": sample["volume"].shape[-1],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = NeuroFMDataset(args.train_manifest)
    val_dataset = NeuroFMDataset(args.val_manifest)

    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model_kwargs = infer_model_kwargs(train_dataset[0])
    model = NeuroFMModel(
        embed_dim=args.embed_dim,
        num_factors=args.num_factors,
        **model_kwargs,
    )
    model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        num_train = 0
        for batch in train_loader:
            batch = {
                "hcp": batch["hcp"].to(device),
                "parcellation": batch["parcellation"].to(device),
                "volume": batch["volume"].to(device),
                "metadata": batch["metadata"].to(device),
                "events": EventDesign(
                    boxcars=batch["events"].boxcars.to(device),
                    parametric=batch["events"].parametric.to(device),
                ),
                "target": batch["target"].to(device),
            }

            optimizer.zero_grad()
            outputs = model(
                hcp=batch["hcp"],
                parcellation=batch["parcellation"],
                volume=batch["volume"],
                metadata=batch["metadata"],
                events=batch["events"],
            )
            loss = criterion(outputs["prediction"], batch["target"])
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch["hcp"].size(0)
            num_train += batch["hcp"].size(0)

        model.eval()
        val_loss = 0.0
        num_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    "hcp": batch["hcp"].to(device),
                    "parcellation": batch["parcellation"].to(device),
                    "volume": batch["volume"].to(device),
                    "metadata": batch["metadata"].to(device),
                    "events": EventDesign(
                        boxcars=batch["events"].boxcars.to(device),
                        parametric=batch["events"].parametric.to(device),
                    ),
                    "target": batch["target"].to(device),
                }
                outputs = model(
                    hcp=batch["hcp"],
                    parcellation=batch["parcellation"],
                    volume=batch["volume"],
                    metadata=batch["metadata"],
                    events=batch["events"],
                )
                loss = criterion(outputs["prediction"], batch["target"])
                val_loss += loss.item() * batch["hcp"].size(0)
                num_val += batch["hcp"].size(0)

        train_loss /= max(1, num_train)
        val_loss /= max(1, num_val)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "neurofm_model.pth")
    print("Training completed. Model saved to neurofm_model.pth")


if __name__ == "__main__":
    main()

