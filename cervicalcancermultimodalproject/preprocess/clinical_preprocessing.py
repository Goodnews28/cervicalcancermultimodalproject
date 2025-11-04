import pandas as pd
import numpy as np
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def simulate_clinical_features(df):
    n = len(df)

    # Simulated features
    df['pelvic_pain'] = np.random.choice(['Yes', 'No'], size=n, p=[0.3, 0.7])
    df['abnormal_bleeding'] = np.random.choice(['Yes', 'No'], size=n, p=[0.4, 0.6])
    df['hpv_vaccinated'] = np.random.choice(['Yes', 'No'], size=n, p=[0.5, 0.5])
    df['smoker'] = np.random.choice(['Yes', 'No'], size=n, p=[0.35, 0.65])
    df['number_of_sexual_partners'] = np.random.poisson(lam=3, size=n) + 1 # type: ignore
    df['prior_screening'] = np.random.choice(['Yes', 'No'], size=n, p=[0.6, 0.4])
    hiv_binary = np.random.binomial(1, 0.1, size=n)
    df['hiv_status'] = ['Positive' if x == 1 else 'Negative' for x in hiv_binary]
    df['parity'] = np.random.poisson(2, size=n).clip(0, 8)
    df['socioeconomic_status'] = np.random.choice(['Low', 'Medium', 'High'], size=n, p=[0.3, 0.5, 0.2])

    # Reorder columns: original first, simulated last
    simulated_cols = [
        'pelvic_pain', 'abnormal_bleeding', 'hpv_vaccinated', 'smoker',
        'number_of_sexual_partners', 'prior_screening', 'hiv_status',
        'parity', 'socioeconomic_status'
    ]
    original_cols = [col for col in df.columns if col not in simulated_cols]
    return df[original_cols + simulated_cols]

def process_and_simulate(dataset_name):
    base_path = f'./processed_data/{dataset_name}'
    
    if dataset_name == 'Herlev':
        input_file = os.path.join(base_path, 'herlev_labels.csv')
        output_file = os.path.join(base_path, 'herlev_labels_with_simulated.csv')
    else:
        input_file = os.path.join(base_path, f'{dataset_name}_processed.csv')
        output_file = os.path.join(base_path, f'{dataset_name}_with_simulated.csv')

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    try:
        df = pd.read_csv(input_file)
        df_sim = simulate_clinical_features(df)
        df_sim.to_csv(output_file, index=False)
        print(f"Simulated clinical data saved to {output_file}")
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    datasets = ['TCGA_CESC', 'GSE178629', 'Herlev']
    for name in datasets:
        process_and_simulate(name)
