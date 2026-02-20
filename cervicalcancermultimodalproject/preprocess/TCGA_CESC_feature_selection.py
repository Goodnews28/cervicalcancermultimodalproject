import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load TCGA_CESC dataset
df = pd.read_csv('./processed_data/TCGA_CESC/TCGA_CESC_with_simulated.csv')

# ---- Simulate binary target label ----
df['target'] = (df['miR21_log2'] > df['miR21_log2'].median()).astype(int)

# Define target and features
y = df['target']
non_feature_cols = ['sample_id', 'target']
X = df.drop(columns=non_feature_cols, errors='ignore')

# Encode categorical variables
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Ensure numeric only
X = X.select_dtypes(include=[np.number])

# Make sure miR21_log2 is included
mir21_col = 'miR21_log2'
mir21_val = X[mir21_col]

# 1. Variance Threshold Selection
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
selected_vt = X.columns[vt.get_support()].tolist()

if mir21_col not in selected_vt:
    selected_vt.append(mir21_col)

# I persist processed outputs so later scripts can reuse the exact same data snapshot.
pd.DataFrame({'feature': selected_vt}).to_csv(
    './processed_data/TCGA_CESC/variance_selected_features.csv', index=False)

# 2. ANOVA F-test
anova = SelectKBest(score_func=f_classif, k='all')
anova.fit(X, y)
f_scores = pd.Series(anova.scores_, index=X.columns).sort_values(ascending=False)

f_scores.reset_index().rename(columns={
    'index': 'feature', 0: 'f_score'
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
}).to_csv('./processed_data/TCGA_CESC/anova_feature_scores.csv', index=False)

# 3. Random Forest Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

importances.reset_index().rename(columns={
    'index': 'feature', 0: 'rf_importance'
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
}).to_csv('./processed_data/TCGA_CESC/random_forest_feature_importance.csv', index=False)

print("TCGA_CESC feature selection completed.")


# === Plotting ANOVA F-scores ===
anova_df = pd.read_csv('./processed_data/TCGA_CESC/anova_feature_scores.csv')
top_anova = anova_df.sort_values('f_score', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=top_anova, x='f_score', y='feature', palette='Blues_d')
plt.title('Top 10 Features (ANOVA F-score) - TCGA_CESC')
plt.tight_layout()
plt.savefig('./processed_data/TCGA_CESC/anova_top10.png')
plt.close()

# === Plotting Random Forest Importance ===
rf_df = pd.read_csv('./processed_data/TCGA_CESC/random_forest_feature_importance.csv')
top_rf = rf_df.sort_values('rf_importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=top_rf, x='rf_importance', y='feature', palette='Greens_d')
plt.title('Top 10 Features (Random Forest Importance) - TCGA_CESC')
plt.tight_layout()
plt.savefig('./processed_data/TCGA_CESC/rf_top10.png')
plt.close()

# === Correlation Heatmap ===
# Load dataset with simulated features
df = pd.read_csv('./processed_data/TCGA_CESC/TCGA_CESC_with_simulated.csv')

# Select only numeric features for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr = numeric_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap - TCGA_CESC (miR-21 + Clinical Features)', fontsize=14)
plt.tight_layout()
plt.savefig('./processed_data/TCGA_CESC/tcga_cesc_correlation_heatmap.png')
plt.close()

print("Correlation heatmap saved.")


# === Load data ===
df = pd.read_csv('./processed_data/TCGA_CESC/TCGA_CESC_with_simulated.csv')

# === Target label for color (optional) ===
df['target'] = (df['miR21_log2'] > df['miR21_log2'].median()).astype(int)

# === Drop ID & target ===
X = df.drop(columns=['sample_id', 'target'], errors='ignore')

# === Encode categorical features ===
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# === Standardize ===
X = X.select_dtypes(include=[np.number])
# I standardize numeric features to mean 0 / std 1 before modeling.
X_scaled = StandardScaler().fit_transform(X)

# === Run t-SNE ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# === Combine for plotting ===
df_tsne = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2'])
df_tsne['miR21_high'] = (df['miR21_log2'] > df['miR21_log2'].median()).map({True: 'High', False: 'Low'})

# === Plot ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='miR21_high', palette='coolwarm', alpha=0.8)
plt.title('t-SNE Projection of TCGA_CESC (omics + simulated features)')
plt.savefig('./processed_data/TCGA_CESC/tsne_projection_TCGA_CESC.png', dpi=300)
plt.show()
