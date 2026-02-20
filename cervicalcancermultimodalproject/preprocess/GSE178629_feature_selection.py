import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('./processed_data/GSE178629/GSE178629_with_simulated.csv')

# ---- Simulate a target label----
# Example: classify based on miR21_log2 being above median
df['target'] = (df['miR21_log2'] > df['miR21_log2'].median()).astype(int)

# Define target and features
y = df['target']
non_feature_cols = ['sample_id', 'target']
X = df.drop(columns=non_feature_cols, errors='ignore')

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Ensure numeric
X = X.select_dtypes(include=[np.number])

# Make sure miR21 is included
mir21_col = 'miR21_log2'
mir21_val = X[mir21_col]

# 1. Variance threshold
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
selected_vt = X.columns[vt.get_support()].tolist()

# Force miR21_log2 to be included
if mir21_col not in selected_vt:
    selected_vt.append(mir21_col)
df_vt = pd.DataFrame({'feature': selected_vt})
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
df_vt.to_csv('./processed_data/GSE178629/variance_selected_features.csv', index=False)

# 2. ANOVA F-test
anova = SelectKBest(score_func=f_classif, k='all')
anova.fit(X, y)
f_scores = pd.Series(anova.scores_, index=X.columns).sort_values(ascending=False)
df_anova = f_scores.reset_index()
df_anova.columns = ['feature', 'f_score']
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
df_anova.to_csv('./processed_data/GSE178629/anova_feature_scores.csv', index=False)

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
df_rf = importances.reset_index()
df_rf.columns = ['feature', 'rf_importance']
# I persist processed outputs so later scripts can reuse the exact same data snapshot.
df_rf.to_csv('./processed_data/GSE178629/random_forest_feature_importance.csv', index=False)

print("Feature selection completed and saved.")

# === Plotting ANOVA F-scores ===
anova_df = pd.read_csv('./processed_data/GSE178629/anova_feature_scores.csv')
top_anova = anova_df.sort_values('f_score', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=top_anova, x='f_score', y='feature', palette='Purples_d')
plt.title('Top 10 Features (ANOVA F-score) - GSE178629')
plt.tight_layout()
plt.savefig('./processed_data/GSE178629/anova_top10.png')
plt.close()

# === Plotting Random Forest Importance ===
rf_df = pd.read_csv('./processed_data/GSE178629/random_forest_feature_importance.csv')
top_rf = rf_df.sort_values('rf_importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=top_rf, x='rf_importance', y='feature', palette='Oranges_d')
plt.title('Top 10 Features (Random Forest Importance) - GSE178629')
plt.tight_layout()
plt.savefig('./processed_data/GSE178629/rf_top10.png')
plt.close()

# === Plotting Correlation Heatmap ===
# Load dataset with simulated features
df = pd.read_csv('./processed_data/GSE178629/GSE178629_with_simulated.csv')

# Select numeric columns only
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap - GSE178629 (miR-21 + Clinical Features)', fontsize=14)
plt.tight_layout()
plt.savefig('./processed_data/GSE178629/gse178629_correlation_heatmap.png')
plt.close()

print("GSE178629 correlation heatmap saved.")

# === Load data ===
df = pd.read_csv('./processed_data/GSE178629/GSE178629_with_simulated.csv')

# === Create a binary target for color (optional) ===
df['target'] = (df['miR21_log2'] > df['miR21_log2'].median()).astype(int)

# === Drop ID & target ===
X = df.drop(columns=['sample_id', 'target'], errors='ignore')

# === Encode categorical variables ===
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# === Standardize numeric values ===
X = X.select_dtypes(include=[np.number])
# I standardize numeric features to mean 0 / std 1 before modeling.
X_scaled = StandardScaler().fit_transform(X)

# === Run t-SNE ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# === Combine results for plotting ===
df_tsne = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2'])
df_tsne['miR21_high'] = (df['miR21_log2'] > df['miR21_log2'].median()).map({True: 'High', False: 'Low'})

# === Plot ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='miR21_high', palette='coolwarm', alpha=0.8)
plt.title('t-SNE Projection of GSE178629 (omics + simulated features)')
plt.savefig('./processed_data/GSE178629/tsne_projection_GSE178629.png', dpi=300)
plt.show()
