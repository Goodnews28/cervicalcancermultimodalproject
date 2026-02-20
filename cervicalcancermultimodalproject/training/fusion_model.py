import torch
import torch.nn as nn
from fusion_cnn_encoder import FusionCNNEncoder
from fusion_tabular_mlp import FusionTabularMLP

class FusionMultimodalModel(nn.Module):
    def __init__(self, 
                 image_input_shape=(3, 224, 224), 
                 tabular_input_dim=17, 
                 fusion_output_dim=128, 
                 #change to 2 for binary classification with CrossEntropyLoss, or keep 1 for BCEWithLogitsLoss??
                 num_classes=1,
                 tabular_hidden_dims=[256, 128],
                 dropout=0.3):
        super(FusionMultimodalModel, self).__init__()

        # Image Encoder
        # I only need channel count from image_input_shape because height/width are handled inside the CNN.
        self.image_encoder = FusionCNNEncoder(in_channels=image_input_shape[0], embedding_dim=fusion_output_dim)

        # Tabular Encoder (MLP)
        self.tabular_encoder = FusionTabularMLP(
            input_dim=tabular_input_dim,
            hidden_dims=tabular_hidden_dims,
            emb_dim=fusion_output_dim,
            dropout=dropout
        )

        # Fusion + Classification Head
        # Two embeddings are concatenated, so classifier input is 2 * fusion_output_dim.
        self.classifier = nn.Sequential(
            nn.Linear(2 * fusion_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):
        #returns intermediate CNN activations for explainability tools.
        if self.image_encoder.return_activations:
            image_feat, activations = self.image_encoder(image, return_activations=True)
        else:
            image_feat = self.image_encoder(image)
            activations = None

        tabular_feat = self.tabular_encoder(tabular)
        # fuse by concatenating feature vectors, not averaging, so each modality keeps its own signal.
        fusion_feat = torch.cat((image_feat, tabular_feat), dim=1)
        output = self.classifier(fusion_feat)

        if self.image_encoder.return_activations:
            return output, activations
        return output
