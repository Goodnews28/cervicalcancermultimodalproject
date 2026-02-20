import pandas as pd
import numpy as np

# Loads miRNA expression data
mirna_raw = pd.read_csv('./data/biomarkers/TCGA_CESC/TCGA.CESC.sampleMap_miRNA_HiSeq_gene.gz', sep='\t', index_col=0)
# I transpose expression data so each row becomes one sample.
mirna_df = mirna_raw.T.reset_index().rename(columns={'index': 'sample_id'}) #transpose and reset index for easier handling
#now the columns are features and the rows are samples

# Extracts miR-21 expression
mir21_col = 'MIMAT0000076'
if mir21_col not in mirna_df.columns:
    raise ValueError(f"miRNA column '{mir21_col}' not found in expression data.")
mirna_df = mirna_df[['sample_id', mir21_col]].rename(columns={mir21_col: 'miR21_expression'})

# Converts to numeric and apply log2 transformation
mirna_df['miR21_expression'] = pd.to_numeric(mirna_df['miR21_expression'], errors='coerce')
# I log-transform skewed expression values to reduce extreme ranges.
mirna_df['miR21_log2'] = mirna_df['miR21_expression'].apply(lambda x: np.nan if pd.isna(x) else round(np.log2(x + 1), 4))

# Z-score normalization (optional for modeling)
mirna_df['miR21_zscore'] = (mirna_df['miR21_log2'] - mirna_df['miR21_log2'].mean()) / mirna_df['miR21_log2'].std()

# Loads clinical metadata
clinical_df = pd.read_csv('./data/biomarkers/TCGA_CESC/TCGA.CESC.sampleMap_CESC_clinicalMatrix', sep='\t', low_memory=False)
clinical_df = clinical_df.rename(columns={clinical_df.columns[0]: 'sample_id'})

# Selects relevant columns
clinical_cols = ['sample_id', 'gender', 'age_at_initial_pathologic_diagnosis',
                 'clinical_stage', 'human_papillomavirus_type', 'vital_status']
available_cols = [col for col in clinical_cols if col in clinical_df.columns]
clinical_df = clinical_df[available_cols]

# Cleans HPV status: fill NaNs and normalize values
clinical_df['hpv_status_clean'] = clinical_df['human_papillomavirus_type'].fillna('Unknown').str.strip().str.lower()

# Adds binary column for early stage classification
clinical_df['early_stage'] = clinical_df['clinical_stage'].str.contains('Stage I', case=False, na=False)

# Merges datasets
merged = pd.merge(mirna_df, clinical_df, on='sample_id', how='inner')

# Drops rows with missing key values
merged = merged.dropna(subset=['miR21_log2', 'clinical_stage'])

# Saves final preprocessed data
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
merged.to_csv('./processed_data/TCGA_CESC_processed.csv', index=False)
print("Preprocessed dataset saved successfully with additional features.")
