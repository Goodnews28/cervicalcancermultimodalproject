import os
import pandas as pd
from PIL import Image
import numpy as np

# === Config ===
HERLEV_RAW_DIR = './data/cytology images/Smear2005/New database pictures'
HERLEV_PROCESSED_DIR = './processed_data/Herlev/herlev_processed_images'
CSV_OUTPUT = './processed_data/Herlev/herlev_labels.csv'
TARGET_SIZE = (224, 224)

# === Label Mapping ===
# Map the original 7 subclasses to binary labels (Normal vs Abnormal)
label_map = {
    'carcinoma_in_situ': 'Abnormal',
    'light_dysplastic': 'Abnormal',
    'moderate_dysplastic': 'Abnormal',
    'severe_dysplastic': 'Abnormal',
    'normal_columnar': 'Normal',
    'normal_intermediate': 'Normal',
    'normal_superficiel': 'Normal',
}

# === Create output dir ===
os.makedirs(HERLEV_PROCESSED_DIR, exist_ok=True)

print(f"\nLooking inside: {HERLEV_RAW_DIR}")
subclasses = os.listdir(HERLEV_RAW_DIR)
print(f"Found {len(subclasses)} subclass folders: {subclasses}")

image_data = []

# === Image Loop ===
# Loop through each subclass folder and process images
for subclass in subclasses:
    subclass_dir = os.path.join(HERLEV_RAW_DIR, subclass)
    if not os.path.isdir(subclass_dir) or subclass not in label_map:
        continue

    label = label_map[subclass]

    for filename in os.listdir(subclass_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        file_path = os.path.join(subclass_dir, filename)
        try:
            img = Image.open(file_path).convert('RGB')
            resized = img.resize(TARGET_SIZE)
            normalized = np.array(resized) / 255.0

            # Clean filename and create relative path
            save_name = f"{subclass}_{filename.replace(' ', '_')}"
            save_path = os.path.join(HERLEV_PROCESSED_DIR, save_name)

            # Save processed image
            Image.fromarray((normalized * 255).astype(np.uint8)).save(save_path)

            # Save relative path for CSV
            relative_path = os.path.relpath(save_path, start='./').replace('\\', '/')

            image_data.append({
                'image_path': relative_path,
                'original_class': subclass,
                'label': label,
                'width': TARGET_SIZE[0],
                'height': TARGET_SIZE[1],
                'is_augmented': '-d' in filename.lower()
            })

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
# === Create DataFrame and Save CSV ===
#datframe stores metadata about each image for the CSV output.
df = pd.DataFrame(image_data)

# Makes sure the directory exists for the CSV
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)

# Saves the DataFrame to CSV
df.to_csv(CSV_OUTPUT, index=False)

print(f"\nProcessed {len(df)} images.")
print(f"Metadata saved to: {CSV_OUTPUT}")

