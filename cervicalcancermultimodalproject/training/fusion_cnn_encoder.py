import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=128, return_activations=False):
        super(FusionCNNEncoder, self).__init__()
        self.return_activations = return_activations

        # Define convolutional layers with increasing feature maps and downsampling
        self.conv1 = nn.Sequential(
            # Input: [B, 3, H, W]
            # This block extracts low-level features and reduces spatial size by half.
            # Kernel_size=3 & padding=1 ensures that the output spatial dimensions=input before pooling.
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  
            #Output: [B, 32, H/2, W/2]
        )

        self.conv2 = nn.Sequential(
            # Input: [B, 32, H/2, W/2]
            # This block captures more complex patterns and further reduces spatial size.
            # Num. of channels doubles to capture more complex features.
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  
            # 0utput: [B, 64, H/4, W/4]
        )

        self.conv3 = nn.Sequential(
            # Input: [B, 64, H/4, W/4]
            # Captures high-level features and reduces spatial size to 1x1.
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  
            # Output: [B, 128, 1, 1]
            # AdaptiveAvgPool2d allows the model to handle variable input sizes by always 
            # outputting a 1x1 spatial dimension, effectively summarizing the feature maps 
            # into a single vector per channel.
        )

        # flattened CNN features into a compact embedding vector.
        self.fc = nn.Linear(128, embedding_dim)

    # processes the input through the convolutional layers and outputs the final embedding.
    def forward(self, x, return_activations=False):
        self.return_activations = return_activations
        a1 = self.conv1(x)
        a2 = self.conv2(a1)
        a3 = self.conv3(a2)
        flat = a3.view(a3.size(0), -1)
        out = self.fc(flat)

        if return_activations:
            return out, {'conv1': a1, 'conv2': a2, 'conv3': a3}
        return out
