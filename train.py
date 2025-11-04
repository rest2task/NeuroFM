import torch
import torch.nn as nn
import torch.optim as optim
from module.model import NeuroFMModel

# Hyperparameters
input_dim = 100    # dimensionality of each input version (feature size per expert input)
embed_dim = 64     # embedding dimension for each expert's output
num_experts = 5    # number of expert inputs (preprocessed versions)
output_dim = 1     # output dimension (e.g., 1 for binary classification or regression)
learning_rate = 1e-3
num_epochs = 20
batch_size = 16

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create the NeuroFM mixture-of-experts model and move it to the device
model = NeuroFMModel(input_dim=input_dim, embed_dim=embed_dim, num_experts=num_experts, output_dim=output_dim)
model.to(device)

# Loss and optimizer (using Binary Cross Entropy with Logits for binary classification)
criterion = nn.BCEWithLogitsLoss()  # appropriate since our output_dim=1 (binary logit)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate a synthetic dataset for demonstration
# We will create N samples, each with 5 input versions and a binary label.
N = 100  # number of samples
# Initialize random inputs for each expert version
# For demonstration, let's introduce a pattern: the first expert input will have information correlated with the label.
X = [None] * num_experts  # will hold synthetic data for each expert
for i in range(num_experts):
    # For each expert, create a tensor of shape (N, input_dim)
    X[i] = torch.randn(N, input_dim)  # start with random normal data
# Create labels (0 or 1)
y = torch.zeros(N, 1)
# Introduce a simple rule for labels based on expert 0's data (for learning signal):
# If the sum of features in expert0 > 0, label = 1, else 0 (this will create a pattern to learn).
sum_expert0 = X[0].sum(dim=1, keepdim=True)
y[sum_expert0 > 0] = 1.0  # set label 1 for those samples where expert0's features sum to positive

# Split into train and validation sets (e.g., 80/20 split)
train_size = int(0.8 * N)
indices = torch.randperm(N)
train_idx, val_idx = indices[:train_size], indices[train_size:]
# Prepare training data
X_train = [x[train_idx] for x in X]  # list of tensors for each expert
y_train = y[train_idx]
# Prepare validation data
X_val = [x[val_idx] for x in X]
y_val = y[val_idx]

# Training loop
for epoch in range(1, num_epochs+1):
    model.train()
    # Mini-batch training
    permutation = torch.randperm(train_size)
    batch_losses = []
    for i in range(0, train_size, batch_size):
        batch_indices = permutation[i:i+batch_size]
        # Collect batch data for each expert and move to device
        batch_inputs = [x[batch_indices].to(device) for x in X_train]  # list of tensors
        batch_labels = y_train[batch_indices].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)            # forward pass (list of inputs to model)
        loss = criterion(outputs, batch_labels)  # compute loss
        loss.backward()                          # backpropagation
        optimizer.step()                         # update parameters
        
        batch_losses.append(loss.item())
    # Compute average loss for epoch
    avg_loss = sum(batch_losses) / len(batch_losses)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_inputs = [x.to(device) for x in X_val]
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, y_val.to(device)).item()
        # Calculate accuracy on validation for insight (for binary classification)
        preds = torch.sigmoid(val_outputs).cpu()  # convert logits to probabilities on CPU
        predicted_labels = (preds >= 0.5).float()
        accuracy = (predicted_labels == y_val).float().mean().item()
    
    print(f"Epoch {epoch:02d}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {accuracy:.2f}")

# Save the trained model parameters
torch.save(model.state_dict(), "neurofm_model.pth")
print("Training completed. Model saved to neurofm_model.pth")
