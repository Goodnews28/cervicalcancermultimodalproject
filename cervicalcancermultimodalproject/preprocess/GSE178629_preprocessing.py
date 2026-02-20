import pandas as pd
import numpy as np

# Loads explorative and validation cohorts
explorative = pd.read_csv('./data/biomarkers/GSE178629/GSE178629_Explorative_cohort_miRNA_normRPM_matrix.txt.gz', sep='\t', index_col=0)
validation = pd.read_csv('./data/biomarkers/GSE178629/GSE178629_Validation_cohort1_miRNA_normRPM_matrix.txt.gz', sep='\t', index_col=0)

print(f"Explorative shape: {explorative.shape}")
print(f"Validation shape: {validation.shape}")

# Combines and transpose
combined = pd.concat([explorative, validation], axis=1)
combined = combined.T.reset_index().rename(columns={'index': 'sample_id'})

# Finds miR-21 columns
mir21_candidates = [col for col in combined.columns if '21' in col and 'miR' in col]
print("Possible miR-21 columns:", mir21_candidates)

# Extracts hsa-miR-21-5p
mir21_col = 'hsa-miR-21-5p'
combined['miR21_expression'] = pd.to_numeric(combined[mir21_col], errors='coerce')
# I log-transform skewed expression values to reduce extreme ranges.
combined['miR21_log2'] = combined['miR21_expression'].apply(lambda x: round(np.log2(x + 1), 4) if pd.notnull(x) else np.nan)

# Drops invalid rows (like header rows or NaN values)
combined = combined[combined['sample_id'] != 'miRNA_Precursor_ID']
combined = combined.dropna(subset=['miR21_log2'])

# Adds z-score normalized expression
combined['miR21_zscore'] = (combined['miR21_log2'] - combined['miR21_log2'].mean()) / combined['miR21_log2'].std()

# Saves cleaned output
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
combined[['sample_id', 'miR21_expression', 'miR21_log2', 'miR21_zscore']].to_csv(
    './processed_data/GSE178629_processed.csv', index=False
)

print(f"Processed {len(combined)} samples. Cleaned dataset saved.")
