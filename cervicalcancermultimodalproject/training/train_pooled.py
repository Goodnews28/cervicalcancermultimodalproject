# train_pooled.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap
from fusion_model import FusionMultimodalModel

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
IMAGE_DIR = "processed_data/Herlev/herlev_processed_images"
CSV_PATH = "processed_data/Herlev/herlev_image_features.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load and preprocess CSV
# ----------------------------
df = pd.read_csv(CSV_PATH)

binary_cols = ['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['hiv_status'] = df['hiv_status'].map({'Positive': 1, 'Negative': 0})
df['socioeconomic_status'] = df['socioeconomic_status'].map({'Low': 0, 'Medium': 1, 'High': 2})

scaler = StandardScaler()
num_cols = ['number_of_sexual_partners', 'parity'] + [f'feat_{i}' for i in range(2042)]  # Up to feat_2041 for 2051 total
df[num_cols] = scaler.fit_transform(df[num_cols])

df['label'] = df['label'].map({'Normal': 0.0, 'Abnormal': 1.0})

# Normalize and validate image paths
missing_files = []
valid_rows = []
for idx, row in df.iterrows():
    image_path = os.path.normpath(row['image_path'])
    if os.path.exists(image_path):
        valid_rows.append(row)
    else:
        missing_files.append(image_path)

df = pd.DataFrame(valid_rows, columns=df.columns)
labels = df['label'].to_numpy()

# Log missing files
if missing_files:
    print(f"Warning: {len(missing_files)} image files not found. Saving list to 'results/pooled_fusion/missing_files.txt'")
    os.makedirs("results/pooled_fusion", exist_ok=True)
    with open("results/pooled_fusion/missing_files.txt", "w") as f:
        f.write(f"Missing files (checked at {pd.Timestamp.now()}):\n")
        f.write("\n".join(missing_files))

# Check if DataFrame is empty after filtering
if df.empty:
    raise ValueError(f"No valid image paths found in the dataset. Please check the CSV file ({CSV_PATH}) and image directory ({IMAGE_DIR}).")

# ----------------------------
# Dataset
# ----------------------------
class MultimodalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        image_path = os.path.normpath(row['image_path'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
        image = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)

        drop_cols = ['image_path', 'original_class', 'label', 'is_augmented']
        numeric_cols = ['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening',
                       'hiv_status', 'socioeconomic_status', 'number_of_sexual_partners', 'parity'] + \
                      [f'feat_{i}' for i in range(2042)]  # Up to feat_2041
        row_dropped = row.drop(columns=drop_cols, errors='ignore')
        tabular_tensor = torch.tensor(row_dropped[numeric_cols].values.astype(float), dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.float)

        return image, tabular_tensor, label

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create parent results directory
os.makedirs("results/pooled_fusion", exist_ok=True)

# ----------------------------
# Cross-Validation Loop
# ----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []
all_fold_aucs, all_fold_f1s = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
    print(f"\n--- Fold {fold + 1} ---")
    fold_dir = f"results/pooled_fusion/fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = MultimodalDataset(train_df, transform)
    val_dataset = MultimodalDataset(val_df, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionMultimodalModel(tabular_input_dim=2051)  # Match the 2051 numeric features
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)  # Increased weight_decay from 1e-4

    train_losses, val_losses, val_aucs = [], [], []
    best_val_loss = float('inf')
    patience = 5  # Reduced from 10
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        model.image_encoder.return_activations = False
        running_loss = 0.0
        for images, tabular, labels in train_loader:
            images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, tabular).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_scores = [], []
        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
                model.image_encoder.return_activations = False
                outputs = model(images, tabular).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(torch.sigmoid(outputs).cpu().numpy())
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_auc = roc_auc_score(y_true, y_scores)
        val_aucs.append(val_auc)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 0.02:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Plot training and validation loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Curves")
    plt.legend()
    plt.savefig(f"{fold_dir}/loss_curve.png")
    plt.close()

    # Plot validation AUC
    plt.plot(val_aucs, label='Val AUC')
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"Fold {fold} Validation AUC")
    plt.legend()
    plt.savefig(f"{fold_dir}/auc_curve.png")
    plt.close()

    # Final validation metrics
    model.eval()
    y_true, y_pred, y_scores, features = [], [], [], []
    with torch.no_grad():
        for images, tabular, labels in val_loader:
            images, tabular = images.to(DEVICE), tabular.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            model.image_encoder.return_activations = False
            logits = model(images, tabular).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            # Extract fusion features for UMAP
            model.image_encoder.return_activations = False
            image_feat = model.image_encoder(images)
            tabular_feat = model.tabular_encoder(tabular)
            fusion_feat = torch.cat((image_feat, tabular_feat), dim=1)
            features.extend(fusion_feat.cpu().numpy())

            # Visualize first batch's activations
            model.image_encoder.return_activations = True
            _, activations = model.image_encoder(images, return_activations=True)
            act = activations.get('conv2')
            if act is not None:
                act = act[0]
                for i in range(min(6, act.shape[0])):
                    plt.imshow(act[i].detach().cpu(), cmap='viridis')
                    plt.title(f"Feature Map {i} from conv2")
                    plt.axis('off')
                    plt.savefig(f"{fold_dir}/activation_map_{i}.png")
                    plt.close()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    # Metrics
    auc = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    all_fold_aucs.append(auc)
    all_fold_f1s.append(f1)

    # Save metrics
    metrics = {
        "fold": fold + 1,
        "AUC": auc,
        "F1": f1,
        "Sensitivity (Recall)": recall,
        "Precision": precision
    }
    fold_metrics.append(metrics)

    with open(f"{fold_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    torch.save(model.state_dict(), f"{fold_dir}/model.pt")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.savefig(f"{fold_dir}/conf_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Fold {fold} ROC Curve")
    plt.legend()
    plt.savefig(f"{fold_dir}/roc_curve.png")
    plt.close()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"Fold {fold} Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.savefig(f"{fold_dir}/calibration_curve.png")
    plt.close()

    # UMAP
    features = np.array(features)
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(features)
    embeddings = np.asarray(embeddings)

    plt.figure(figsize=(6, 5))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y_true, cmap='coolwarm', s=10)
    plt.title(f"Fold {fold} UMAP Projection")
    plt.colorbar()
    plt.savefig(f"{fold_dir}/umap.png")
    plt.close()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----------------------------
# Summary
# ----------------------------
print("\nCross-Validation Results:")
for fm in fold_metrics:
    print(f"Fold {fm['fold']}: AUC={fm['AUC']:.3f}, F1={fm['F1']:.3f}, Sens={fm['Sensitivity (Recall)']:.3f}, Precision={fm['Precision']:.3f}")

# Average Metrics
avg_metrics = {
    "AUC": np.mean([m["AUC"] for m in fold_metrics]),
    "F1": np.mean([m["F1"] for m in fold_metrics]),
    "Sensitivity": np.mean([m["Sensitivity (Recall)"] for m in fold_metrics]),
    "Precision": np.mean([m["Precision"] for m in fold_metrics]),
}
print("\nAverage CV Metrics:")
for k, v in avg_metrics.items():
    print(f"{k}: {v:.3f}")

# Save average metrics
with open("results/pooled_fusion/average_metrics.json", "w") as f:
    json.dump(avg_metrics, f, indent=2)

# Plot bar charts for fold-wise metrics
folds = list(range(1, 5 + 1))
auc_scores = [m['AUC'] for m in fold_metrics]
f1_scores = [m['F1'] for m in fold_metrics]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=folds, y=auc_scores)
plt.title("Fold-wise AUC")
plt.xlabel("Fold")
plt.ylabel("AUC")

plt.subplot(1, 2, 2)
sns.barplot(x=folds, y=f1_scores)
plt.title("Fold-wise F1")
plt.xlabel("Fold")
plt.ylabel("F1 Score")

plt.tight_layout()
plt.savefig("results/pooled_fusion/fold_metric_bars.png")
plt.close()