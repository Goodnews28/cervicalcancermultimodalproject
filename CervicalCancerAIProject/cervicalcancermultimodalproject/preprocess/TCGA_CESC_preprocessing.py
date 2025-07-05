import pandas as pd
import numpy as np

# Load miRNA expression data (rows: miRNAs, columns: sample IDs)
mirna_raw = pd.read_csv('./data/biomarkers/TCGA_CESC/TCGA.CESC.sampleMap_miRNA_HiSeq_gene.gz', 
                       sep='\t', index_col=0)

# Transpose to get samples as rows
mirna_df = mirna_raw.T.reset_index().rename(columns={'index': 'sample_id'})

# Column name for hsa-miR-21-5p
mir21_col = 'MIMAT0000076'

# Check if miR-21 column exists
if mir21_col not in mirna_df.columns:
    raise ValueError(f"miRNA column '{mir21_col}' not found in expression data.")

# Keep only sample ID and miR-21 expression
mirna_df = mirna_df[['sample_id', mir21_col]].rename(columns={mir21_col: 'miR21_expression'})

# Log2 transform (add 1 to avoid log(0))
mirna_df['miR21_log2'] = pd.to_numeric(mirna_df['miR21_expression'], errors='coerce').apply(
    lambda x: np.nan if pd.isna(x) else round(np.log2(x + 1), 4)
)

# Load clinical metadata
# NOTE: your clinical matrix file has sample IDs in the first column, no transpose needed
clinical_df = pd.read_csv('./data/biomarkers/TCGA_CESC/TCGA.CESC.sampleMap_CESC_clinicalMatrix', 
                          sep='\t', low_memory=False)

# Rename sample ID column if necessary (assume first column is sampleID)
clinical_df = clinical_df.rename(columns={clinical_df.columns[0]: 'sample_id'})

# Select relevant clinical columns if they exist, else print warning
clinical_cols = ['sample_id', 'gender', 'age_at_initial_pathologic_diagnosis', 
                 'clinical_stage', 'human_papillomavirus_type', 'vital_status']

available_cols = [col for col in clinical_cols if col in clinical_df.columns]
missing_cols = set(clinical_cols) - set(available_cols)

if missing_cols:
    print(f"Warning: Clinical columns missing from data and will be skipped: {missing_cols}")

clinical_df = clinical_df[available_cols]

# Merge miRNA expression with clinical data on sample_id
merged = pd.merge(mirna_df, clinical_df, on='sample_id', how='inner')

# Drop missing values in key columns if needed
merged = merged.dropna(subset=['miR21_log2', 'clinical_stage'])

# Save processed data to CSV
merged.to_csv('./PROCESSED_data/TCGA_CESC_preprocessed.csv', index=False)

print("Preprocessed dataset saved successfully.")
