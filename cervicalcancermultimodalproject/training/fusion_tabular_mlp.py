import torch
import torch.nn as nn

class FusionTabularMLP(nn.Module):
    def __init__(self, input_dim: int = 17, emb_dim: int = 128, hidden_dims: list = [256, 128], dropout: float = 0.3):
        super(FusionTabularMLP, self).__init__()

        layers = []
        # track the current input width so each hidden layer can chain correctly.
        in_dim = input_dim

        # Build MLP layers based on hidden_dims list
        #hidden_dims is a list of integers specifying the number of neurons in each hidden layer.
        for hidden_dim in hidden_dims:
            # learns nonlinear patterns with regularization.
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final layer to project to embedding dimension
        layers.append(nn.Linear(in_dim, emb_dim)) 

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # pass the whole tabular tensor through the stacked MLP blocks in one call.
        return self.mlp(x)
