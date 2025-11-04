import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
CSV_PATH = './processed_data/Herlev/herlev_labels_with_simulated.csv'
IMAGE_FEATURE_OUTPUT = './processed_data/Herlev/herlev_image_features.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
IMAGE_SIZE = 224

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# === CUSTOM DATASET ===
class HerlevDataset(Dataset):
    def __init__(self, csv_df):
        self.df = csv_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, idx

# === LOAD METADATA CSV ===
df = pd.read_csv(CSV_PATH)
df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)

# === MODEL: ResNet50 w/o classification head ===
resnet = models.resnet50(pretrained=True)
# Replace the final classifier with a Linear layer that outputs the same input (identity)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, resnet.fc.in_features)
resnet.fc.weight.data = torch.eye(resnet.fc.in_features)
resnet.fc.bias.data.zero_()
resnet = resnet.to(DEVICE)
resnet.eval()

# === DATA LOADER ===
dataset = HerlevDataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === FEATURE EXTRACTION ===
features = np.zeros((len(df), 2048))
with torch.no_grad():
    for images, indices in tqdm(loader, desc="Extracting features"):
        images = images.to(DEVICE)
        output = resnet(images).cpu().numpy()
        features[indices.numpy()] = output

# === SAVE TO CSV ===
feature_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
final_df = pd.concat([df.drop(columns=['width', 'height']), feature_df], axis=1)

# Export
final_df.to_csv(IMAGE_FEATURE_OUTPUT, index=False)
print(f"\nFeature extraction complete. Saved to: {IMAGE_FEATURE_OUTPUT}")


# === CONFIG ===
FEATURE_CSV = './processed_data/Herlev/herlev_image_features.csv'
LABEL_COL = 'label'  # Normal or Abnormal
OPTIONAL_METADATA_COL = 'pelvic_pain' 

# === Load Data ===
df = pd.read_csv(FEATURE_CSV)

# Extract features
feature_cols = [col for col in df.columns if col.startswith('feat_')]
X = df[feature_cols].values

# Labels
labels = df[LABEL_COL]
optional = df[OPTIONAL_METADATA_COL]

# Encode for plotting
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(labels)

# === Run t-SNE ===
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X)

# === Add to DataFrame ===
df['tsne_1'] = X_embedded[:, 0]
df['tsne_2'] = X_embedded[:, 1]

# === Plot ===
plt.figure(figsize=(10, 7))
sns.set(style='whitegrid')

palette = {'Normal': 'skyblue', 'Abnormal': 'salmon'}
markers = {'Yes': 's', 'No': 'o'}

for marker_val in df[OPTIONAL_METADATA_COL].unique():
    subset = df[df[OPTIONAL_METADATA_COL] == marker_val]
    sns.scatterplot(
        data=subset,
        x='tsne_1', y='tsne_2',
        hue=LABEL_COL,
        palette=palette,
        style=OPTIONAL_METADATA_COL,
        markers=markers,
        s=80,
        edgecolor='black'
    )


plt.title('t-SNE of Herlev Image Features\nColor = Label | Shape = Pelvic Pain')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# Save the plot
plt.savefig('./processed_data/Herlev/herlev_tsne_plot.png', dpi=300)
