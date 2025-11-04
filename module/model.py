import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertEncoder(nn.Module):
    """
    An expert network that processes one version of the input.
    This could be a small feedforward network or a CNN, depending on data shape.
    For simplicity, we use a two-layer MLP to produce an embedding.
    """
    def __init__(self, input_dim, embed_dim):
        super(ExpertEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, max(input_dim, embed_dim))  # hidden layer
        self.fc2 = nn.Linear(max(input_dim, embed_dim), embed_dim)  # output embedding layer

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)  # out shape: (batch_size, embed_dim)
        return out

class GatingNetwork(nn.Module):
    """
    Gating network that produces weights for each expert.
    It takes a concatenated feature vector (or other metadata) and outputs a weight for each expert.
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        # Simple two-layer MLP for gating (can be more complex if needed)
        self.fc1 = nn.Linear(input_dim, input_dim // 2 if input_dim // 2 > 0 else 1)
        self.fc2 = nn.Linear(input_dim // 2 if input_dim // 2 > 0 else 1, num_experts)

    def forward(self, x):
        # x shape: (batch_size, input_dim), representing combined features or metadata
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)             # logits for each expert, shape: (batch_size, num_experts)
        weights = F.softmax(logits, dim=1)  # convert to softmax weights across experts
        return weights  # shape: (batch_size, num_experts)

class NeuroFMModel(nn.Module):
    """
    The NeuroFM mixture-of-experts model. 
    It contains multiple expert encoders and a gating mechanism to combine their outputs.
    Optionally, a final head is used for prediction (e.g., classification or regression).
    """
    def __init__(self, input_dim, embed_dim, num_experts=5, output_dim=1):
        super(NeuroFMModel, self).__init__()
        self.num_experts = num_experts
        # Initialize each expert encoder (one per preprocessed version of input).
        # Each expert might be specialized; here we use identical structure for simplicity.
        self.experts = nn.ModuleList([
            ExpertEncoder(input_dim, embed_dim) for _ in range(num_experts)
        ])
        # (For clarity, each expert could be commented as:
        # Expert 1: e.g., Minimal Preprocessing (ICA-FIX denoised)
        # Expert 2: HRF-Optimized input
        # Expert 3: Motion-Denoised input
        # Expert 4: GLM-Residual input
        # Expert 5: Parcellation-Based input
        # In this implementation, all experts share the same architecture but have separate parameters.)
        
        # Gating network to combine expert outputs.
        # Gating input dim = num_experts * embed_dim (concatenated experts' embeddings).
        self.gating = GatingNetwork(input_dim=num_experts * embed_dim, num_experts=num_experts)
        # Final prediction head: maps the combined expert embedding to desired output dimension.
        self.final_head = nn.Linear(embed_dim, output_dim)
        
    def forward(self, inputs):
        """
        Forward pass for the NeuroFM model.
        :param inputs: list or tuple of length `num_experts`, each a tensor of shape (batch, input_dim)
                       representing one version of the preprocessed input.
        """
        assert len(inputs) == self.num_experts, f"Expected {self.num_experts} input versions, got {len(inputs)}"
        
        # Get each expert's output embedding
        expert_outputs = []  # will hold tensors of shape (batch, embed_dim) from each expert
        for expert, x in zip(self.experts, inputs):
            out = expert(x)              # shape: (batch, embed_dim)
            expert_outputs.append(out)
        # Stack expert outputs into a single tensor of shape (batch, num_experts, embed_dim)
        expert_outputs_tensor = torch.stack(expert_outputs, dim=1)  # shape: (batch, 5, embed_dim)
        
        # Flatten the expert outputs for gating input (concatenate along feature dim)
        # shape after view: (batch, num_experts * embed_dim)
        batch_size = expert_outputs_tensor.size(0)
        # We use reshape to (batch, num_experts*embed_dim)
        gating_input = expert_outputs_tensor.view(batch_size, -1)
        
        # Compute gating weights for each expert
        weights = self.gating(gating_input)  # shape: (batch, num_experts)
        # Reshape weights for broadcasting: (batch, num_experts, 1)
        weights_expanded = weights.unsqueeze(2)  # prepare for weighted sum
        
        # Compute weighted sum of expert outputs
        # Multiply each expert output by its weight and sum across experts (dim=1)
        combined_embedding = torch.sum(expert_outputs_tensor * weights_expanded, dim=1)  # shape: (batch, embed_dim)
        
        # Final prediction using the combined embedding
        output = self.final_head(combined_embedding)  # shape: (batch, output_dim)
        # Note: For classification (binary), output is a logit; use sigmoid in loss. For regression, output is direct value.
        return output
