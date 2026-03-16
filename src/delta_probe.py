'''
How to run:
   python ./src/delta_probe.py
'''
import pandas as pd
from scipy import stats

DATAPATH = "./data/datdump.tsv"
MODEL_LIST = [
    "llama",
    "llama_random",
    "mistral",
    "mistral_random",
    "qwen",
    "qwen_random"
]
TEST_SIZE = {
    "emg": 47,
    "tee": 73,
    "har": 2947,
    "had": 7959,
    "rwc": 4501,
    "ctu": 250
}


def main():
    df = pd.read_csv(DATAPATH, sep='\t')
    df = df[(df['model'].isin(MODEL_LIST)) & (df['method'] == "logistic_regression")]

    pairs = []
    for model in MODEL_LIST:
        if model.endswith("_random"):
            pairs.append((model[:-7], model))

    results = []
    for base_model, random_model in pairs:
        base_df = df[df['model'] == base_model]
        random_df = df[df['model'] == random_model]
        
        for dataset in base_df['dataset'].unique():
            for modality in base_df['modality'].unique():
                # Extracting scores
                b_match = base_df[(base_df['dataset'] == dataset) & (base_df['modality'] == modality)]
                r_match = random_df[(random_df['dataset'] == dataset) & (random_df['modality'] == modality)]
                
                if not b_match.empty and not r_match.empty:
                    base_score = b_match['f1'].values[0]
                    random_score = r_match['f1'].values[0]
                    results.append({
                        'base_model': base_model,
                        'dataset': dataset,
                        'modality': modality,
                        'base_score': base_score,
                        'random_score': random_score,
                        'delta': base_score - random_score
                    })

    results_df = pd.DataFrame(results)
    
    # --- SIGNIFICANCE TESTING SECTION ---
    print("### STATISTICAL SIGNIFICANCE RESULTS (One-Sided Wilcoxon) ###\n")
    
    unique_models = results_df['base_model'].unique()
    alpha = 0.05
    # Bonferroni Correction: adjust alpha based on number of model families tested
    adj_alpha = alpha / len(unique_models) 

    for model in unique_models:
        model_data = results_df[results_df['base_model'] == model]
        
        # We test if the 'base_score' is significantly 'greater' than 'random_score'
        stat, p_val = stats.wilcoxon(model_data['base_score'], 
                                     model_data['random_score'], 
                                     alternative='greater')
        
        is_significant = p_val < adj_alpha
        
        print(f"Model Family: {model.upper()}")
        print(f"  Observations: {len(model_data)}")
        print(f"  Mean Delta:   {model_data['delta'].mean():.4f}")
        print(f"  P-Value:      {p_val:.5f}")
        print(f"  Significant:  {is_significant} (at adj. alpha {adj_alpha:.3f})")
        print("-" * 40)

if __name__ == "__main__":
    main()