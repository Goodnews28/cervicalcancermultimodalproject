import torch
import torch.nn as nn

class FusionTabularMLP(nn.Module):
    def __init__(self, input_dim: int = 17, emb_dim: int = 128, hidden_dims: list = [256, 128], dropout: float = 0.3):
        super(FusionTabularMLP, self).__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, emb_dim))  # Final projection to emb_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)