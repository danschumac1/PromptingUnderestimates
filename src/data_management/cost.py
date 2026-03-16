'''
How to run:
   python ./src/data_management/cost.py
'''
import pandas as pd
import sys; sys.path.append("./src")
from utils.file_io import load_json
def main():

    data = load_json("data/gpt4o_cost_analysis_results.json")

    # Load data
    df = pd.DataFrame(data["results"])

    # 1. Calculate the cost for 1,000 rows for each individual entry
    df['cost_per_1k_rows'] = df['avg_cost_usd'] * 1000

    # 2. Group by 'embedding_type' and calculate the mean
    # We aggregate the token counts and the costs
    avg_df = df.groupby('embedding_type').agg({
        'avg_total_tokens': 'mean',
        'avg_cost_usd': 'mean',
        'cost_per_1k_rows': 'mean'
    }).reset_index()

    # 3. Rename columns for clarity
    # Rename columns to be explicit
    avg_df.columns = [
        'Embedding Type', 
        'Avg Tokens Per Row', 
        'Avg Cost Per Row', 
        'Projected Cost (1,000 Rows)'
    ]

    format_dict = {
        'Avg Tokens Per Row': '{:,.2f}'.format,
        # 'Avg Cost Per Row': '${:.6f}'.format, # Increased precision to show the small fraction
        'Projected Cost (1,000 Rows)': '${:.2f}'.format
    }

    # 4. Sort by cost and format
    avg_df = avg_df.sort_values('Projected Cost (1,000 Rows)', ascending=True)
    avg_df.drop(columns=['Avg Cost Per Row'], inplace=True) # Drop the per-row cost as it's less relevant for the final presentation
    
    # Use .format as the callable function for each column

    # Apply formatting and print
    print(avg_df.to_string(index=False, formatters=format_dict))

if __name__ == "__main__":
    main()