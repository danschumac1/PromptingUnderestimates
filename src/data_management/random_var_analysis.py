'''
2026-02-18
How to run:
    python ./src/data_management/random_var_analysis.py
'''
import pandas as pd
import numpy as np
from scipy import stats

def main():
    # 1. Load the data
    df = pd.read_csv("./data/random_variant_results.tsv", sep='\t') 
    df.drop(columns=["approach"], inplace=True) # Drop approach if it's not needed for grouping 
    # 2. Clean base model name
    df['model_family'] = df['mode_name'].str.rsplit('_', n=1).str[0]

    # 3. Define the aggregation function
    def calculate_stats(x):
        n = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=1) if n > 1 else 0
        
        ci_hi = 0
        if n > 1:
            confidence = 0.95
            t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
            ci_hi = t_crit * (std / np.sqrt(n))
        
        return pd.Series({
            'Mean': mean,
            'StdDev': std,
            'Range': np.max(x) - np.min(x),
            'CI_95': ci_hi,
            'Runs': float(n) # Kept as float to match your output style
        })

    # 4. Group by Dataset, Model, AND Embedding Type
    # We include 'embedding_type' here to see vis-ust vs lets-ust
    group_cols = [
        'dataset', 
        'model_family', 
        # 'embedding_type', 
        # 'layer'
        ]
    
    # If 'approach' contains 'logistic_regression' and it's the same for all, 
    # you can leave it out or keep it. I'll include it for completeness.
    if 'approach' in df.columns:
        group_cols.append('approach')

    results = (df.groupby(group_cols)['accuracy']
               .apply(calculate_stats)
               .unstack())

    # 5. Print for visibility
    print("--- STABILITY ANALYSIS REPORT ---")
    # Using to_string() ensures the table doesn't truncate columns
    print(results.round(4).to_string())

    # save to tsv
    results.to_csv("./data/random_variant_analysis_report.tsv", sep='\t')
    print("Report saved to ./data/random_variant_analysis_report.tsv")

if __name__ == "__main__":
    main()