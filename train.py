from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from module.model import EventDesign, NeuroFMModel


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------


def build_synthetic_dataset(
    num_samples: int,
    struct_dim: int,
    rest_dim: int,
    rest_time: int,
    task_dim: int,
    task_time: int,
    num_experts: int,
    num_events: int,
    event_param_dim: int,
    metadata_dim: int,
):
    """Create a synthetic dataset that exercises the full NeuroFM pipeline."""

    structural = torch.randn(num_samples, struct_dim)
    resting = torch.randn(num_samples, rest_time, rest_dim)

    # Construct expert streams with a shared signal on expert 0 to create
    # learnable structure.
    task_streams = []
    base_signal = torch.randn(num_samples, task_time, task_dim)
    for expert_id in range(num_experts):
        noise = torch.randn(num_samples, task_time, task_dim) * 0.1
        if expert_id == 0:
            task_streams.append(base_signal + noise)
        else:
            task_streams.append(torch.randn_like(base_signal) + noise)

    # Event design: each sample contains ``num_events`` stimuli whose
    # parametric modulator is correlated with the target label.
    boxcars = torch.zeros(num_samples, num_events, task_time)
    parametric = torch.zeros(num_samples, num_events, event_param_dim)
    for idx in range(num_samples):
        for e in range(num_events):
            onset = torch.randint(0, task_time // 2, (1,)).item()
            duration = torch.randint(task_time // 8, task_time // 4, (1,)).item()
            duration = min(duration, task_time - onset)
            boxcars[idx, e, onset:onset + duration] = 1.0
            parametric[idx, e, 0] = torch.sin(0.1 * (onset + duration))
    design = EventDesign(boxcars=boxcars, parametric=parametric)

    # Metadata encodes acquisition traits with a subtle dependency on the base
    # signal to drive the gating network.
    metadata = torch.randn(num_samples, metadata_dim)
    metadata[:, 0] = base_signal.mean(dim=(1, 2))

    # Targets correlate with the combination of structural signal and the
    # parametric modulators of the first event.
    targets = (
        structural.mean(dim=1, keepdim=True)
        + parametric[:, 0, 0].unsqueeze(1)
        + base_signal.mean(dim=(1, 2), keepdim=True)
    )
    targets = torch.tanh(targets)

    return {
        "structural": structural,
        "resting": resting,
        "task_streams": task_streams,
        "design": design,
        "metadata": metadata,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_EXPERTS = 5
STRUCT_DIM = 128
REST_DIM = 64
REST_TIME = 20
TASK_DIM = 32
TASK_TIME = 32
NUM_EVENTS = 6
EVENT_PARAM_DIM = 1
METADATA_DIM = 4
EMBED_DIM = 96
NUM_FACTORS = 12

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 16


model = NeuroFMModel(
    struct_dim=STRUCT_DIM,
    rest_dim=REST_DIM,
    task_dim=TASK_DIM,
    event_param_dim=EVENT_PARAM_DIM,
    time_points=TASK_TIME,
    num_events=NUM_EVENTS,
    metadata_dim=METADATA_DIM,
    embed_dim=EMBED_DIM,
    num_experts=NUM_EXPERTS,
    num_factors=NUM_FACTORS,
    output_dim=1,
)
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


dataset = build_synthetic_dataset(
    num_samples=120,
    struct_dim=STRUCT_DIM,
    rest_dim=REST_DIM,
    rest_time=REST_TIME,
    task_dim=TASK_DIM,
    task_time=TASK_TIME,
    num_experts=NUM_EXPERTS,
    num_events=NUM_EVENTS,
    event_param_dim=EVENT_PARAM_DIM,
    metadata_dim=METADATA_DIM,
)

indices = torch.randperm(dataset["structural"].size(0))
train_indices = indices[:100]
val_indices = indices[100:]


def slice_dataset(data, idx):
    return {
        "structural": data["structural"][idx],
        "resting": data["resting"][idx],
        "task_streams": [stream[idx] for stream in data["task_streams"]],
        "design": EventDesign(
            boxcars=data["design"].boxcars[idx],
            parametric=data["design"].parametric[idx],
        ),
        "metadata": data["metadata"][idx],
        "targets": data["targets"][idx],
    }


train_data = slice_dataset(dataset, train_indices)
val_data = slice_dataset(dataset, val_indices)


def iterate_batches(data, batch_size):
    total = data["structural"].size(0)
    permutation = torch.randperm(total)
    for start in range(0, total, batch_size):
        batch_idx = permutation[start:start + batch_size]
        yield {
            "structural": data["structural"][batch_idx].to(DEVICE),
            "resting": data["resting"][batch_idx].to(DEVICE),
            "task_streams": [stream[batch_idx].to(DEVICE) for stream in data["task_streams"]],
            "design": EventDesign(
                boxcars=data["design"].boxcars[batch_idx].to(DEVICE),
                parametric=data["design"].parametric[batch_idx].to(DEVICE),
            ),
            "metadata": data["metadata"][batch_idx].to(DEVICE),
            "targets": data["targets"][batch_idx].to(DEVICE),
        }


for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_losses: List[float] = []
    for batch in iterate_batches(train_data, BATCH_SIZE):
        optimizer.zero_grad()
        outputs = model(
            structural=batch["structural"],
            resting=batch["resting"],
            task_experts=batch["task_streams"],
            metadata=batch["metadata"],
            events=batch["design"],
        )
        loss = criterion(outputs["prediction"], batch["targets"])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_batch = {
            "structural": val_data["structural"].to(DEVICE),
            "resting": val_data["resting"].to(DEVICE),
            "task_streams": [stream.to(DEVICE) for stream in val_data["task_streams"]],
            "design": EventDesign(
                boxcars=val_data["design"].boxcars.to(DEVICE),
                parametric=val_data["design"].parametric.to(DEVICE),
            ),
            "metadata": val_data["metadata"].to(DEVICE),
            "targets": val_data["targets"].to(DEVICE),
        }
        val_outputs = model(
            structural=val_batch["structural"],
            resting=val_batch["resting"],
            task_experts=val_batch["task_streams"],
            metadata=val_batch["metadata"],
            events=val_batch["design"],
        )
        val_loss = criterion(val_outputs["prediction"], val_batch["targets"]).item()

    mean_train_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch {epoch:02d} | Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}")


torch.save(model.state_dict(), "neurofm_model.pth")
print("Training completed. Model saved to neurofm_model.pth")
