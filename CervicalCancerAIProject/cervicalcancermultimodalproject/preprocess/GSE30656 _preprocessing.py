import gzip
import pandas as pd
import numpy as np

def load_expression_data(expr_gz_path):
    with gzip.open(expr_gz_path, 'rt') as f:
        lines = f.readlines()

    try:
        start_index = next(i for i, line in enumerate(lines) if line.strip() == '!series_matrix_table_begin') + 1
    except StopIteration:
        raise ValueError("Could not find '!series_matrix_table_begin' in the expression data file.")

    header_line = lines[start_index].strip()
    header_line_stripped = header_line.replace('"', '').strip()

    if header_line_stripped.startswith('ID_REF'):
        data_start = start_index
    elif header_line_stripped.startswith('ID'):
        data_start = start_index
    else:
        raise ValueError(f"Unexpected header line at {start_index}: {header_line}")

    data_lines = lines[data_start:]
    data_str = ''.join(data_lines)

    df_expr = pd.read_csv(pd.io.common.StringIO(data_str), sep='\t', index_col=0)
    print(df_expr.head())
    print(df_expr.index[:20])
    print(df_expr.columns.tolist())

    return df_expr
# 

def load_annotation(annot_gz_path):
    with gzip.open(annot_gz_path, 'rt') as f:
        lines = f.readlines()

    # Find header line starting with "ID"
    header_line_index = next(i for i, line in enumerate(lines) if line.strip().startswith("ID"))
    df_annot = pd.read_csv(pd.io.common.StringIO(''.join(lines[header_line_index:])), sep='\t')
    print("Annotation columns:", df_annot.columns.tolist())
    return df_annot
    

def main():
    expr_path = './data/biomarkers/GSE30656/GSE30656_series_matrix.txt.gz'
    annot_path = './data/biomarkers/GSE30656/annotation_probe_names.gz'

    df_expr = load_expression_data(expr_path)
    print("Sample expression matrix row indices:")
    print(df_expr.index[:20].tolist())
    print(f"✅ Expression data shape: {df_expr.shape}")

    df_annot = load_annotation(annot_path)
    print(f"✅ Annotation data shape: {df_annot.shape}")

    print("Example expression index types:", [type(i) for i in df_expr.index[:5]])
    print("Example annotation ID types:", df_annot['ID'].head().apply(type).tolist())

    # Dynamically find miR-21 probes from annotation by searching 'GeneName' or 'miRNA_ID'
    miR21_probes = df_annot[
    df_annot['GENE_NAME'].str.contains('miR-21', case=False, na=False) |
    df_annot['miRNA_ID'].str.contains('miR-21', case=False, na=False)
    ]['ID'].astype(str).tolist()

    matched_probes = [probe for probe in miR21_probes if probe in df_expr.index]
    print(f"Found {len(miR21_probes)} probes annotated as miR-21.")

    # Only keep probes that exist in the expression data
    matched_probes = [probe for probe in miR21_probes if probe in df_expr.index]

    if not matched_probes:
        print("❌ No matching miR-21 probes found in expression data.")
        return

    print(f"✅ Found {len(matched_probes)} matching miR-21 probes in expression data.")

    # Extract expression values and compute average
    df_miR21 = df_expr.loc[matched_probes].astype(float)
    avg_expr = df_miR21.mean(axis=0)
    log2_expr = np.log2(avg_expr + 1)

    print("🔬 Log2 Expression (first 10 samples):")
    print(log2_expr.head(10))
    print(log2_expr.describe())

    output_path = './processed_data/GSE30656_miR21_expression.csv'
    log2_expr.to_frame(name='miR21_log2_expression').to_csv(output_path)
    print(f"✅ miR-21 expression data saved to: {output_path}")

if __name__ == '__main__':
    main()
