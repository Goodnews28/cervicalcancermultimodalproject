# type: ignore
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
import time
matplotlib.use('Agg')  # Set non-interactive backend globally
from fusion_model import FusionMultimodalModel  # Ensure this matches your model file

# --- Heartbeat Function ---
def print_heartbeat(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S EDT')}] {message}")

# --- Dataset ---
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = "processed_data/Herlev/herlev_processed_images"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract filename from the first column, assuming it might contain full path
        img_name = os.path.basename(self.dataframe.iloc[idx, 0])  # Get only the filename
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Define numeric columns
        numeric_cols = ['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening',
                       'hiv_status', 'socioeconomic_status', 'number_of_sexual_partners', 'parity'] + \
                      [f'feat_{i}' for i in range(2042)]  # Up to feat_2041
        
        # Preprocess binary and categorical columns
        row = self.dataframe.iloc[idx].copy()
        binary_cols = ['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening']
        for col in binary_cols:
            row[col] = 1 if row[col] == 'Yes' else 0 if row[col] == 'No' else row[col]
        row['hiv_status'] = 1 if row['hiv_status'] == 'Positive' else 0 if row['hiv_status'] == 'Negative' else row['hiv_status']
        row['socioeconomic_status'] = {'Low': 0, 'Medium': 1, 'High': 2}.get(row['socioeconomic_status'], row['socioeconomic_status'])

        tabular = row[numeric_cols].values.astype(np.float32)
        
        # Convert label to float with safe default
        label_value = row['label']
        label_float = {'Normal': 0.0, 'Abnormal': 1.0}.get(label_value, 0.0)  # Default to 0.0 if unexpected
        label = torch.tensor(float(label_float), dtype=torch.float)  # Ensure conversion to float
        return image, tabular, label

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load dataset ---
df = pd.read_csv("processed_data/Herlev/herlev_image_features.csv")
np.random.seed(42)
df['age'] = np.random.randint(20, 81, size=len(df))
df['region'] = np.random.randint(1, 6, size=len(df))

# Preprocess the entire DataFrame to ensure consistency
binary_cols = ['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['hiv_status'] = df['hiv_status'].map({'Positive': 1, 'Negative': 0})
df['socioeconomic_status'] = df['socioeconomic_status'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['label'] = df['label'].map({'Normal': 0.0, 'Abnormal': 1.0})  # Ensure label is preprocessed

# Check overall label distribution
print_heartbeat(f"Overall label distribution: {df['label'].value_counts()}")

# --- Stratified Cross-Validation ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(df.index, df['label'])):
    print_heartbeat(f"Starting analysis for Fold {fold}")
    model_path = f"results/pooled_fusion/fold_{fold}/model.pt"
    result_dir = f"results/pooled_fusion/fold_{fold}/"
    os.makedirs(result_dir, exist_ok=True)

    # Load model
    model = FusionMultimodalModel(tabular_input_dim=2051)  # Match the 2051 numeric features
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Validation split
    val_df = df.iloc[val_idx].copy()
    val_dataset = MultimodalDataset(val_df, transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Check validation label distribution
    val_labels = val_df['label']
    print_heartbeat(f"Fold {fold} validation label distribution: {val_labels.value_counts()}")

    # --- SHAP ---
    try:
        print_heartbeat(f"Starting SHAP analysis for Fold {fold}")
        start_time = time.time()
        tabular_data = []
        labels = []
        for i, (_, tabular, label) in enumerate(val_loader):
            tabular_data.append(tabular.numpy())
            labels.append(label.numpy())
            if i % 10 == 0:  # Heartbeat every 10 batches
                print_heartbeat(f"Processing SHAP batch {i}/{len(val_loader)} for Fold {fold}")
        tabular_data = np.vstack(tabular_data)
        labels = np.concatenate(labels)

        def model_predict(x):
            batch_size = x.shape[0]
            mock_image = torch.zeros(1, 3, 8, 8)  # 8x8 image
            mock_images = mock_image.expand(batch_size, -1, -1, -1)  # Broadcast to batch size
            tabular = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(mock_images, tabular).detach().numpy()
                if np.allclose(outputs, 0) or outputs.size == 0:
                    print_heartbeat(f"Warning: Model output is zero or empty for batch size {batch_size} in Fold {fold}")
                return outputs

        background = shap.sample(tabular_data, 5)  # Keep background samples low
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(tabular_data, batch_size=4)  # Reduced to batch_size 4
        shap_values = np.squeeze(shap_values, axis=2) if shap_values is not None else np.zeros((len(val_idx), 2051))
        print_heartbeat(f"SHAP values shape after squeeze: {shap_values.shape} for Fold {fold}")
        print_heartbeat(f"Tabular data shape: {tabular_data.shape} for Fold {fold}")
        print_heartbeat(f"SHAP values valid: {np.any(shap_values)} and not all NaN: {not np.all(np.isnan(shap_values))} for Fold {fold}")

        plt.ioff()  # Ensure non-interactive mode
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, tabular_data, feature_names=['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening',
                           'hiv_status', 'socioeconomic_status', 'number_of_sexual_partners', 'parity'] + \
                          [f'feat_{i}' for i in range(2042)], plot_type='bar', show=False)
        plt.title(f"SHAP Summary - Fold {fold}")
        try:
            plt.savefig(f"{result_dir}/shap_summary.png")
            print_heartbeat(f"Saved shap_summary.png to {result_dir}")
            if os.path.exists(f"{result_dir}/shap_summary.png"):
                print_heartbeat(f"Verified shap_summary.png exists at {result_dir}")
            else:
                print_heartbeat(f"Failed to verify shap_summary.png at {result_dir}")
        except Exception as e:
            print_heartbeat(f"Error saving shap_summary.png: {str(e)} for Fold {fold}")
        plt.close()
        print_heartbeat(f"SHAP analysis completed for Fold {fold} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print_heartbeat(f"Error in SHAP analysis for Fold {fold}: {str(e)}")
        continue

    # --- LIME ---
    try:
        print_heartbeat(f"Starting LIME analysis for Fold {fold}")
        start_time = time.time()
        def predict_tabular(x):
            with torch.no_grad():
                images = torch.zeros(x.shape[0], 3, 8, 8)
                tabular = torch.tensor(x, dtype=torch.float32)
                outputs = model(images, tabular).squeeze()
                probabilities = torch.sigmoid(outputs)
                return np.stack([1 - probabilities.numpy(), probabilities.numpy()], axis=1)

        lime_explainer = LimeTabularExplainer(
            training_data=tabular_data,
            feature_names=['pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker', 'prior_screening',
                           'hiv_status', 'socioeconomic_status', 'number_of_sexual_partners', 'parity'] + \
                          [f'feat_{i}' for i in range(2042)],
            class_names=["Normal", "Abnormal"],
            mode="classification"
        )

        lime_results = lime_explainer.explain_instance(
            data_row=tabular_data[0],
            predict_fn=predict_tabular,
            num_features=6
        )

        lime_results.save_to_file(f"{result_dir}/lime_explanation.html")
        print_heartbeat(f"Saved lime_explanation.html to {result_dir}")
        print_heartbeat(f"LIME analysis completed for Fold {fold} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print_heartbeat(f"Error in LIME analysis for Fold {fold}: {str(e)}")
        continue

    # --- Grad-CAM ---
    try:
        print_heartbeat(f"Starting Grad-CAM analysis for Fold {fold}")
        start_time = time.time()
        model.eval()
        grad_cam = GradCAM(model=model, target_layers=[model.image_encoder.conv3])

        # Get predictions for the validation set
        all_images, all_tabular, all_labels = [], [], []
        for images, tabular, labels in val_loader:
            all_images.append(images)
            all_tabular.append(tabular)
            all_labels.append(labels)
        all_images = torch.cat(all_images).to("cpu")
        all_tabular = torch.cat(all_tabular).to("cpu")
        all_labels = torch.cat(all_labels).to("cpu")
        with torch.no_grad():
            outputs = model(all_images, all_tabular).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            print_heartbeat(f"Predictions: {predictions.numpy()}, Labels: {all_labels.numpy()}")
            print_heartbeat(f"Predictions shape: {predictions.shape}, Labels shape: {all_labels.shape}")

        # Identify TP and FN indices with debug
        tp_indices = (predictions == 1) & (all_labels == 1)
        fn_indices = (predictions == 0) & (all_labels == 1)
        print_heartbeat(f"TP indices count debug: {tp_indices.sum().item()}, FN indices count debug: {fn_indices.sum().item()}")
        tp_images = all_images[tp_indices][:5]  # Limit to 5 for TP
        fn_images = all_images[fn_indices][:5]  # Limit to 5 for FN
        print_heartbeat(f"TP images shape: {tp_images.shape if tp_images.shape else 'None'}, FN images shape: {fn_images.shape if fn_images.shape else 'None'}")

        # Process TP cases
        if tp_images.shape[0] > 0:
            def model_forward_tp(images):
                tp_tabular = all_tabular[tp_indices]
                if images.shape[0] > tp_tabular.shape[0]:
                    tp_tabular = tp_tabular[:images.shape[0]]
                return model(images, tp_tabular)
            original_forward = model.forward
            model.forward = model_forward_tp.__get__(model, model.__class__)
            targets = [ClassifierOutputTarget(1)]  # type: ignore[reportArgumentType, reportGeneralTypeIssues]
            grayscale_cam_tp = grad_cam(input_tensor=tp_images, targets=targets)
            model.forward = original_forward
            cam_images_tp = []
            for i in range(min(5, tp_images.shape[0])):
                img = tp_images[i].permute(1, 2, 0).numpy()
                cam = show_cam_on_image(img, grayscale_cam_tp[i], use_rgb=True)
                cam_images_tp.append(cam)
                print_heartbeat(f"TP cam image {i} shape: {cam.shape if cam is not None else 'None'}")
            plt.figure(figsize=(15, 3))
            for i, cam in enumerate(cam_images_tp):
                plt.subplot(1, 5, i + 1)
                plt.imshow(cam)
                plt.axis('off')
            try:
                plt.savefig(f"{result_dir}/gradcam_tp.png")
                print_heartbeat(f"Saved gradcam_tp.png to {result_dir}")
                if os.path.exists(f"{result_dir}/gradcam_tp.png"):
                    print_heartbeat(f"Verified gradcam_tp.png exists at {result_dir}")
                else:
                    print_heartbeat(f"Failed to verify gradcam_tp.png at {result_dir}")
            except Exception as e:
                print_heartbeat(f"Error saving gradcam_tp.png for Fold {fold}: {str(e)}")
            plt.close()
        else:
            print_heartbeat(f"No TP images found for Fold {fold}")

        # Process FN cases
        if fn_images.shape[0] > 0:
            def model_forward_fn(images):
                fn_tabular = all_tabular[fn_indices]
                if images.shape[0] > fn_tabular.shape[0]:
                    fn_tabular = fn_tabular[:images.shape[0]]
                return model(images, fn_tabular)
            original_forward = model.forward
            model.forward = model_forward_fn.__get__(model, model.__class__)
            targets = [ClassifierOutputTarget(1)]  # type: ignore[reportArgumentType, reportGeneralTypeIssues]
            grayscale_cam_fn = grad_cam(input_tensor=fn_images, targets=targets)
            model.forward = original_forward
            cam_images_fn = []
            for i in range(min(5, fn_images.shape[0])):
                img = fn_images[i].permute(1, 2, 0).numpy()
                cam = show_cam_on_image(img, grayscale_cam_fn[i], use_rgb=True)
                cam_images_fn.append(cam)
                print_heartbeat(f"FN cam image {i} shape: {cam.shape if cam is not None else 'None'}")
            plt.figure(figsize=(15, 3))
            for i, cam in enumerate(cam_images_fn):
                plt.subplot(1, 5, i + 1)
                plt.imshow(cam)
                plt.axis('off')
            try:
                plt.savefig(f"{result_dir}/gradcam_fn.png")
                print_heartbeat(f"Saved gradcam_fn.png to {result_dir}")
                if os.path.exists(f"{result_dir}/gradcam_fn.png"):
                    print_heartbeat(f"Verified gradcam_fn.png exists at {result_dir}")
                else:
                    print_heartbeat(f"Failed to verify gradcam_fn.png at {result_dir}")
            except Exception as e:
                print_heartbeat(f"Error saving gradcam_fn.png for Fold {fold}: {str(e)}")
            plt.close()
        else:
            print_heartbeat(f"No FN images found for Fold {fold}")

        print_heartbeat(f"Grad-CAM analysis completed for Fold {fold} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print_heartbeat(f"Error in Grad-CAM analysis for Fold {fold}: {str(e)}")
        continue

    # --- Fairness AUC across demographic bins ---
    try:
        print_heartbeat(f"Starting Fairness analysis for Fold {fold}")
        start_time = time.time()
        predictions = []
        for i, (images, tabular, _) in enumerate(val_loader):
            with torch.no_grad():
                outputs = model(images.to("cpu"), tabular.to("cpu")).squeeze()
                predictions.extend(torch.sigmoid(outputs).numpy())
            if i % 10 == 0:
                print_heartbeat(f"Processing Fairness batch {i}/{len(val_loader)} for Fold {fold}")

        val_labels = val_df['label'].to_numpy()
        print_heartbeat(f"Val labels shape: {val_labels.shape}, Predictions length: {len(predictions)}")
        if len(val_labels) != len(predictions):
            print_heartbeat(f"Warning: Label-prediction length mismatch in Fold {fold}: {len(val_labels)} vs {len(predictions)}")
            min_len = min(len(val_labels), len(predictions))
            val_labels = val_labels[:min_len]
            predictions = predictions[:min_len]
        predictions_binary = (np.array(predictions) > 0.5).astype(int)
        print_heartbeat(f"Predictions binary: {np.unique(predictions_binary, return_counts=True)}")

        # Compute age decade bins
        val_df_subset = val_df.copy()
        val_df_subset['age_decade'] = (val_df_subset['age'] // 10 * 10)

        # Group by age decade and region
        groups = val_df_subset.groupby(['age_decade', 'region'])
        fairness_data = []
        for name, group in groups:
            idx = group.index
            valid_idx = [i for i in range(len(val_labels)) if i in idx]
            print_heartbeat(f"Group {name}: Raw idx = {idx.tolist()}, Valid idx = {valid_idx}")
            if valid_idx:
                group_labels = val_labels[valid_idx]
                group_preds = predictions_binary[valid_idx]
                tp = np.sum((group_preds == 1) & (group_labels == 1))
                tn = np.sum((group_preds == 0) & (group_labels == 0))
                fp = np.sum((group_preds == 1) & (group_labels == 0))
                fn = np.sum((group_preds == 0) & (group_labels == 1))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                print_heartbeat(f"Group {name}: TP={tp}, TN={tn}, FP={fp}, FN={fn}, Sensitivity={sensitivity}, Specificity={specificity}")
                fairness_data.append([name[0], name[1], sensitivity, specificity])

        # Create fairness table
        fairness_df = pd.DataFrame(fairness_data, columns=['Age Decade', 'Region', 'Sensitivity', 'Specificity'])
        fairness_df.to_csv(f"{result_dir}/fairness_table.csv", index=False)
        print_heartbeat(f"Saved fairness_table.csv to {result_dir}")

        # AUC by group with debug
        auc_by_group = {}
        predictions_np = np.array(predictions)
        for name, group in groups:
            idx = group.index
            valid_idx = [i for i in range(len(val_labels)) if i in idx]
            if valid_idx:
                group_labels = val_labels[valid_idx]
                group_preds = predictions_np[valid_idx]
                try:
                    auc = roc_auc_score(group_labels, group_preds)
                    auc_by_group[name] = auc
                    print_heartbeat(f"Group {name}: AUC = {auc}, Labels = {group_labels.tolist()}, Predictions = {group_preds.tolist()}")
                except ValueError as e:
                    auc_by_group[name] = np.nan
                    print_heartbeat(f"Group {name}: AUC calculation failed: {e}, Labels = {group_labels.tolist()}, Predictions = {group_preds.tolist()}")
            else:
                auc_by_group[name] = np.nan
                print_heartbeat(f"Group {name}: No valid indices")

        if not auc_by_group:
            print_heartbeat(f"Warning: No valid AUC values for Fold {fold}")
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(auc_by_group)), list(auc_by_group.values()))
            plt.xticks(range(len(auc_by_group)), list(auc_by_group.keys()), rotation=45)
            plt.title(f"Fold {fold}: AUC by Simulated Demographic Groups")
            plt.xlabel("Age Decade, Region")
            plt.ylabel("AUC")
            plt.tight_layout()
            plt.savefig(f"{result_dir}/fairness_chart.png")
            print_heartbeat(f"Saved fairness_chart.png to {result_dir}")
            if os.path.exists(f"{result_dir}/fairness_chart.png"):
                print_heartbeat(f"Verified fairness_chart.png exists at {result_dir}")
            else:
                print_heartbeat(f"Failed to verify fairness_chart.png at {result_dir}")
            plt.close()

        print_heartbeat(f"Fairness analysis completed for Fold {fold} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print_heartbeat(f"Error in Fairness analysis for Fold {fold}: {str(e)}")
        continue

print_heartbeat("Interpretability and fairness analysis completed across all folds.")